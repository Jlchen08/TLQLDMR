#!/usr/bin/env python3
"""Independently audit a validation-selected study; never fit or select a model.

Checks frozen sources/data, all candidate validation predictions and argmins,
selection locks, fitting-only preprocessing/IDs, every reported metric and
static/causal-rolling calibration. A deterministic sample of saved kernel fits
is reconstructed to verify R,c,C, primal/dual objectives and predictions. The
frozen data loader and training-only sampler are reused; the study runner and
refinement solver are not imported. Floating-point certificates are numerical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.selection_data import load_all_selection_data
from review.selection_sampling import stratified_target_indices

CAPACITY = {1:99., 2:200., 3:99., 4:66., 5:36., 6:96.}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-8, rtol=1e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, f'{label}: shape differs {a.shape} versus {b.shape}')
    check(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite value')
    check(np.allclose(a, b, atol=atol, rtol=rtol), f'{label}: numerical mismatch')


def ids_equal(actual, expected, label):
    check(np.array_equal(actual, expected), f'{label}: identifiers/order differ')


def point_metrics(y, prediction):
    e = y-prediction
    denominator = float(np.sum((y-y.mean())**2))
    return {'n':len(y), 'rmse':float(np.sqrt(np.mean(e*e))),
            'mae':float(np.mean(np.abs(e))),
            'r2':float(1-np.sum(e*e)/denominator) if denominator > 0 else None}


def interval_metrics(y, lower, upper, alpha):
    check(np.all(lower <= upper), 'Interval endpoints are reversed')
    width = upper-lower
    penalty = np.where(y < lower, lower-y, 0.) + np.where(y > upper, y-upper, 0.)
    return {'n':len(y), 'picp':float(np.count_nonzero((lower <= y) & (y <= upper))/len(y)),
            'mpiw':float(width.mean()), 'winkler':float(np.mean(width+2/alpha*penalty))}


def metrics_equal(actual, expected, label):
    check(set(actual) == set(expected), f'{label}: metric names differ')
    for key, value in expected.items():
        if value is None:
            check(actual[key] is None, f'{label}.{key}: expected undefined metric')
        else:
            close(actual[key], value, label+'.'+key)


def calibration_rank(scores, alpha):
    scores = np.asarray(scores, float)
    check(len(scores) > 0 and np.isfinite(scores).all() and np.all(scores >= 0),
          'Calibration scores must be nonnegative and finite')
    rank = math.ceil((len(scores)+1)*(1-alpha))
    check(1 <= rank <= len(scores), 'Calibration rank is not finite')
    return float(np.sort(scores)[rank-1]), rank


def causal_rolling_quantiles(calibration_scores, test_scores, window, alpha):
    """Independent prefix construction: score t is excluded at prediction t."""
    calibration_scores, test_scores = np.asarray(calibration_scores), np.asarray(test_scores)
    answer = []
    for t in range(len(test_scores)):
        available = np.concatenate((calibration_scores, test_scores[:t]))[-window:]
        answer.append(calibration_rank(available, alpha)[0])
    return np.asarray(answer)


class Verifier:
    def __init__(self, directory, allow_partial=False, sample_models=15):
        self.directory = Path(directory).resolve()
        self.allow_partial, self.sample_models = allow_partial, sample_models
        self.errors, self.pending = [], []
        self.contexts, self.validated_fits, self.final_records = {}, {}, []
        self.audit = {'candidate_validations':0, 'selections':0, 'results':0,
                      'kernel_models_reconstructed':[], 'nonconverged_kernel_fits':[],
                      'result_hashes':{}}
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        self.cfg = json.loads((self.directory/'protocol.json').read_text())
        lock_path = self.directory/'selection_lock.json'
        self.lock_at_start = json.loads(lock_path.read_text()) if lock_path.exists() else None
        if self.lock_at_start is None:
            check(not list(self.directory.glob('*/*/final_seed*/result.json')),
                  'Test results already exist without a selection lock')
        check(self.cfg == self.manifest['protocol'], 'Manifest/protocol JSON disagree')
        for name, expected in self.manifest['source_hashes'].items():
            check(digest(ROOT/name) == expected, f'Frozen source changed: {name}')
        for name, expected in self.manifest['data_hashes'].items():
            check(digest(ROOT/name) == expected, f'Frozen workbook changed: {name}')
        reference = ROOT/'review/replied_comment_runs/chronological_v1/manifest.json'
        check(digest(reference) == self.manifest['reference_manifest_sha256'], 'Reference manifest changed')
        check(allow_partial or self.manifest['status'] == 'complete', 'Study is not complete')
        self.data = load_all_selection_data(include_past_power=False)
        provenance = json.loads((self.directory/'data_provenance.json').read_text())
        for site, data in self.data.items():
            check(provenance[str(site)]['provenance'] == data.provenance, f'Farm {site}: provenance differs')
            check(provenance[str(site)]['preprocessing'] == data.preprocessing, f'Farm {site}: loader protocol differs')
            check(data.X.shape[1] == 312, f'Farm {site}: unexpected meteorology-only feature schema')
        self.tasks = {task['name']:task for task in self.cfg['tasks']}

    def attempt(self, label, function, *args):
        try:
            return function(*args)
        except Exception as error:
            self.errors.append({'item':label, 'error':f'{type(error).__name__}: {error}'})
            return None

    def context(self, task_name, method):
        target_only = method == 'TargetSVQR'
        key = (task_name, target_only)
        if key in self.contexts:
            return self.contexts[key]
        task = self.tasks[task_name]
        site, data = task['site'], self.data[task['site']]
        folder = self.directory/task_name/('target_data' if target_only else 'pooled_data')
        saved = np.load(folder/'ids.npz', allow_pickle=False)
        actual_target = saved['target_window']
        expected_target = np.flatnonzero(data.mask('train','target'))
        if task['mode'] == 'cross':
            expected_target, sampling = stratified_target_indices(data, self.cfg['cross_target_fit_budget'])
            check(json.loads((folder/'sampling.json').read_text()) == sampling, 'Training sampling audit differs')
        ids_equal(actual_target, expected_target, f'{key}: target fitting IDs')
        check(len(np.unique(actual_target)) == len(actual_target), 'Duplicate target fitting IDs')
        source_sites, source_windows, xs, ys = [], [], [], []
        if not target_only:
            sources = [site] if task['mode'] == 'within' else [s for s in self.data if s != site]
            cap = self.cfg['within_site_source_cap'] if task['mode'] == 'within' else self.cfg['source_cap_per_other_site']
            for source in sources:
                selected = np.flatnonzero(self.data[source].mask('train','source'))
                if len(selected) > cap:
                    selected = selected[np.linspace(0,len(selected)-1,cap,dtype=int)]
                source_sites.extend([source]*len(selected));source_windows.extend(selected.tolist())
                xs.append(self.data[source].X[selected]);ys.append(self.data[source].y_raw[selected]/CAPACITY[source])
        ids_equal(saved['source_site'], source_sites, f'{key}: source-site IDs')
        ids_equal(saved['source_window'], source_windows, f'{key}: source-window IDs')
        all_ids = {'train':actual_target}
        for split in ('validation','calibration','test'):
            expected = np.flatnonzero(data.mask(split,'target'))
            ids_equal(saved[split+'_window'], expected, f'{key}: {split} IDs')
            all_ids[split] = expected
        for i, (name, ids) in enumerate(all_ids.items()):
            check(len(ids) and len(np.unique(ids)) == len(ids), f'{name}: empty or duplicate IDs')
            check(np.all(data.timestamps[ids][1:] > data.timestamps[ids][:-1]), f'{name}: not chronological')
            for other in list(all_ids)[i+1:]:
                check(not np.intersect1d(ids, all_ids[other]).size, f'{name}/{other}: target IDs overlap')
        close(data.timestamps[all_ids['calibration']].astype('datetime64[ns]').astype('int64').max()
              < data.timestamps[all_ids['test']].astype('datetime64[ns]').astype('int64').min(), True,
              'Calibration outcomes must precede testing')
        raw_x = np.concatenate(xs+[data.X[actual_target]])
        raw_y = np.concatenate(ys+[data.y_raw[actual_target]/CAPACITY[site]])
        finite = np.isfinite(raw_x)
        means = np.array([raw_x[finite[:,j],j].mean(dtype=float) if finite[:,j].any() else 0.
                          for j in range(raw_x.shape[1])])
        filled = np.where(finite, raw_x, means)
        scales = filled.std(axis=0);scales[scales == 0] = 1.
        y_mean, y_scale = raw_y.mean(), raw_y.std() or 1.
        state = json.loads((folder/'preprocessing.json').read_text())
        for name, value in [('mean',means),('scale',scales),('label_mean',y_mean),('label_scale',y_scale)]:
            close(state[name], value, f'{key}: fitting-only {name}')
        x, y = (filled-means)/scales, (raw_y-y_mean)/y_scale
        counts = json.loads((folder/'counts.json').read_text())
        check(counts['n_source'] == len(source_windows) and counts['n_target_fit'] == len(actual_target),
              'Fitting count differs from IDs')
        for split in ('validation','calibration','test'):
            check(counts['n_'+split] == len(all_ids[split]), f'{split}: count differs')
            for event, mask in data.event_masks.items():
                check(counts['events'][split][event] == int(mask[all_ids[split]].sum()), f'{split}/{event}: count differs')
        answer = {'folder':folder, 'data':data, 'site':site, 'ids':all_ids, 'state':state,
                  'x':x, 'y':y, 'source_n':len(source_windows), 'counts':counts}
        self.contexts[key] = answer
        return answer

    def fit_metadata(self, folder):
        relative = str(folder.relative_to(self.directory))
        if relative in self.validated_fits:
            return self.validated_fits[relative]
        record = json.loads((folder/'fit.json').read_text())
        task, method = record['task'], record['method']
        ctx = self.context(task, method)
        check(record['counts'] == ctx['counts'], f'{relative}: fit counts mismatch')
        check(record['site'] == ctx['site'], f'{relative}: wrong site')
        grid = self.cfg['tree_candidates' if method == 'HistGBR' else 'kernel_candidates']
        check(record['candidate'] == grid[record['candidate_index']], f'{relative}: candidate differs')
        check(record['seed'] in ([0] if method == 'HistGBR' else self.cfg['seeds']), 'Unexpected seed')
        check(record['tau'] in self.cfg['taus'], 'Unexpected quantile')
        check(record['data_folder'] == str(ctx['folder'].relative_to(self.directory)), 'Wrong preprocessing folder')
        if method != 'HistGBR':
            diagnostics = record['diagnostics']
            P, D = diagnostics['primal'], diagnostics['dual_bound']
            close(diagnostics['gap'], max(0.,P-D), 'Recorded absolute gap')
            scale = max(1.,abs(P),abs(D))
            close(diagnostics['relative_gap'], max(0.,P-D)/scale, 'Recorded relative gap')
            check(D <= P+1e-8*scale, 'Dual bound exceeds primal')
            achieved = diagnostics['gap'] <= diagnostics['gap_atol']+diagnostics['gap_rtol']*scale
            check(bool(diagnostics['converged']) == achieved, 'Convergence flag disagrees with objective gap')
            if not diagnostics['converged']:
                self.audit['nonconverged_kernel_fits'].append(relative)
        self.validated_fits[relative] = record
        return record

    def candidate(self, task, method, index):
        folder = self.directory/task/method/'candidates'/f'c{index:02d}'
        if not all((folder/name).is_file() for name in ('fit.json','validation.csv','validation_metrics.json')):
            self.pending.append(str(folder.relative_to(self.directory)));return None
        ctx = self.context(task,method)
        fit = self.fit_metadata(folder)
        check(fit['candidate_index'] == index and fit['seed'] == 0 and fit['tau'] == .5,
              'Candidate index/seed/quantile differs')
        frame = pd.read_csv(folder/'validation.csv')
        ids_equal(frame['window_index'], ctx['ids']['validation'], 'Candidate validation IDs')
        close(frame['y_true'], ctx['data'].y_raw[ctx['ids']['validation']], 'Candidate raw MW labels')
        close(frame['prediction'],frame['prediction'],'Candidate predictions finite')
        metrics = point_metrics(frame['y_true'].to_numpy(),frame['prediction'].to_numpy())
        metrics_equal(json.loads((folder/'validation_metrics.json').read_text()), metrics, 'Validation metric')
        self.audit['candidate_validations'] += 1
        return {'candidate_index':index, 'validation':metrics, 'diagnostics':fit['diagnostics']}

    def selection(self, task, method, candidates):
        path = self.directory/task/method/'selection.json'
        if not path.exists():
            self.pending.append(str(path.relative_to(self.directory)));return None
        check(all(candidate is not None for candidate in candidates), 'Selection saved before every candidate is complete')
        record = json.loads(path.read_text())
        check(len(record['candidates']) == len(candidates), 'Some candidate rows were omitted')
        for actual, expected in zip(record['candidates'],candidates):
            check(actual['candidate_index'] == expected['candidate_index'], 'Candidate order differs')
            metrics_equal(actual['validation'],expected['validation'],'Selection candidate metric')
            check(actual['diagnostics'] == expected['diagnostics'],'Selection candidate diagnostics differ')
        best = min(candidates,key=lambda row:(row['validation']['rmse'],row['candidate_index']))
        check(record['selected_index'] == best['candidate_index'], 'Selection is not Q1 RMSE argmin with declared tie-breaking')
        metrics_equal(record['selected_validation'],best['validation'],'Selected validation metric')
        self.audit['selections'] += 1
        return record

    def result(self, task, method, seed, selection):
        folder = self.directory/task/method/f'final_seed{seed}'
        path = folder/'result.json'
        if not path.exists():
            self.pending.append(str(path.relative_to(self.directory)));return None
        check(selection is not None,'Final result exists without complete selection')
        result = json.loads(path.read_text())
        check((result['task'],result['method'],result['seed']) == (task,method,seed),'Result identity differs')
        check(result['selected_index'] == selection['selected_index'],'Final model did not use locked candidate')
        ctx = self.context(task,method)
        check(result['counts'] == ctx['counts'], 'Final counts differ')
        frame, cal = pd.read_csv(folder/'predictions.csv'), pd.read_csv(folder/'calibration.csv')
        for split, table in [('test',frame),('calibration',cal)]:
            indices = ctx['ids'][split]
            ids_equal(table['window_index'],indices,split+' prediction IDs')
            close(table['y_true'],ctx['data'].y_raw[indices],split+' raw MW labels')
            ids_equal(pd.to_datetime(table['time']).to_numpy(dtype='datetime64[ns]'),ctx['data'].timestamps[indices],split+' timestamps')
            close(table.select_dtypes(include='number').to_numpy(),table.select_dtypes(include='number').to_numpy(),split+' finite predictions')
        lower = np.minimum(frame['raw_q025'],frame['raw_q975']).to_numpy()
        upper = np.maximum(frame['raw_q025'],frame['raw_q975']).to_numpy()
        cal_lower = np.minimum(cal['q025'],cal['q975']).to_numpy()
        cal_upper = np.maximum(cal['q025'],cal['q975']).to_numpy()
        y, cy = frame['y_true'].to_numpy(),cal['y_true'].to_numpy()
        scores = np.maximum(0.,np.maximum(cal_lower-cy,cy-cal_upper))
        close(cal['score'],scores,'Calibration residual scores')
        q, rank = calibration_rank(scores,self.cfg['alpha'])
        check(result['calibration']['n'] == len(scores) and result['calibration']['rank'] == rank,'Calibration rank/count differs')
        close(result['calibration']['static_q'],q,'Static expansion')
        close(frame['static_lower'],lower-q,'Static lower endpoint')
        close(frame['static_upper'],upper+q,'Static upper endpoint')
        test_scores = np.maximum(0.,np.maximum(lower-y,y-upper))
        rolling = causal_rolling_quantiles(scores,test_scores,self.cfg['rolling_window'],self.cfg['alpha'])
        close(frame['rolling_q'],rolling,'Strict-past rolling expansion')
        close(frame['rolling_lower'],lower-rolling,'Rolling lower endpoint')
        close(frame['rolling_upper'],upper+rolling,'Rolling upper endpoint')
        metrics_equal(result['point'],point_metrics(y,frame['prediction'].to_numpy()),'Final point metrics')
        for name, lo, hi in [('raw_interval',lower,upper),('static_interval',lower-q,upper+q),
                             ('rolling_interval',lower-rolling,upper+rolling)]:
            metrics_equal(result[name],interval_metrics(y,lo,hi,self.cfg['alpha']),name)
        check(len(result['fits']) == len(self.cfg['taus']),'Missing final quantile fit')
        for item, tau in zip(result['fits'],self.cfg['taus']):
            fit_folder = self.directory/item['path']
            saved = self.fit_metadata(fit_folder)
            check({k:v for k,v in item.items() if k != 'path'} == saved,'Embedded final fit metadata differ')
            check(saved['tau'] == tau and saved['seed'] == seed,'Final fit quantile/seed differs')
            check(saved['candidate_index'] == selection['selected_index'],'Interval settings differ from point selection')
        self.audit['results'] += 1
        self.audit['result_hashes'][str(path.relative_to(self.directory))] = digest(path)
        self.final_records.append(result)
        return result

    def model(self, relative):
        folder = self.directory/relative
        fit = self.fit_metadata(folder)
        ctx = self.context(fit['task'],fit['method'])
        saved = np.load(folder/'model.npz',allow_pickle=False)
        feature = np.load(self.directory/fit['feature_map'],allow_pickle=False)
        centers, normalization = feature['centers'],feature['normalization']
        gamma = float(feature['gamma'])
        close(gamma,fit['candidate']['gamma'],'Feature bandwidth')
        check(len(centers) == min(self.cfg['landmarks'],len(ctx['x'])), 'Effective landmark count differs')
        close(centers,np.load((self.directory/fit['feature_map']).parent.parent/'landmarks.npy'),'Shared seed landmarks')
        W = np.exp(-gamma*cdist(centers,centers,'sqeuclidean'))
        close(normalization,normalization.T,'Symmetric landmark normalization')
        close(normalization@(W+1e-6*np.eye(len(W)))@normalization,np.eye(len(W)),
              'Stabilized inverse-square-root map',atol=2e-7,rtol=2e-7)
        Z = np.exp(-gamma*cdist(ctx['x'],centers,'sqeuclidean'))@normalization
        ns, m = ctx['source_n'],Z.shape[1]
        candidate, method = fit['candidate'],fit['method']
        lam1 = candidate['lambda1'] if method in ('Full','NoDiscrepancy') else 0.
        lam2 = candidate['lambda2'] if method in ('Full','NoVariance') else 0.
        R, c = np.eye(m),np.zeros(m)
        if ns:
            centered = Z[:ns]-Z[:ns].mean(0)
            centered_y = ctx['y'][:ns]-ctx['y'][:ns].mean()
            delta = Z[:ns].mean(0)-Z[ns:].mean(0)
            R += lam1/ns*centered.T@centered+lam2*np.outer(delta,delta)
            c += lam1/ns*centered.T@centered_y
        fraction, total = (candidate['target_fraction'] if ns else 1.),candidate['total_loss']
        C = np.r_[np.full(ns,total*(1-fraction)/ns) if ns else [],np.full(len(Z)-ns,total*fraction/(len(Z)-ns))]
        for name,expected in [('R',R),('c',c),('C',C),('y',ctx['y'])]:
            close(saved[name],expected,'Reconstructed '+name,atol=2e-8,rtol=2e-8)
        w,b,v,tau = saved['w'],float(saved['b']),saved['dual'],fit['tau']
        residual = ctx['y']-Z@w-b
        P = float(.5*w@R@w-c@w+C@np.where(residual >= 0,tau*residual,(tau-1)*residual))
        h = c+Z.T@v
        D = float(ctx['y']@v-.5*h@cho_solve(cho_factor(R,lower=True),h))
        tolerance = 100*np.finfo(float).eps*max(1.,float(C.sum()))
        check(abs(v.sum()) <= tolerance,'Reconstructed dual is not zero-sum')
        check(np.all(v >= C*(tau-1)-tolerance) and np.all(v <= C*tau+tolerance),'Reconstructed dual violates its box')
        close(P,fit['diagnostics']['primal'],'Reconstructed primal objective',atol=2e-7,rtol=2e-8)
        close(D,fit['diagnostics']['dual_bound'],'Reconstructed dual objective',atol=2e-7,rtol=2e-8)
        check(D <= P+1e-8*max(1.,abs(P)),'Reconstructed primal/dual inequality fails')
        # Reconstruct candidate validation or final calibrated/test raw outputs.
        tables = []
        if (folder/'validation.csv').exists():
            tables.append(('validation',pd.read_csv(folder/'validation.csv'),'prediction'))
        for result in self.final_records:
            if any(item['path'] == relative for item in result['fits']):
                base = self.directory/result['task']/result['method']/f"final_seed{result['seed']}"
                column = 'prediction' if tau == .5 else ('raw_q025' if tau == .025 else 'raw_q975')
                tables.append(('test',pd.read_csv(base/'predictions.csv'),column))
                if tau != .5:
                    tables.append(('calibration',pd.read_csv(base/'calibration.csv'),'q025' if tau == .025 else 'q975'))
        for split, table, column in tables:
            raw_x = ctx['data'].X[ctx['ids'][split]]
            state = ctx['state'];mean,scale = np.asarray(state['mean']),np.asarray(state['scale'])
            x = (np.where(np.isfinite(raw_x),raw_x,mean)-mean)/scale
            prediction = (np.exp(-gamma*cdist(x,centers,'sqeuclidean'))@normalization)@w+b
            prediction = (prediction*state['label_scale']+state['label_mean'])*CAPACITY[ctx['site']]
            close(table[column],prediction,'Reconstructed MW prediction',atol=5e-7,rtol=5e-8)
        self.audit['kernel_models_reconstructed'].append({'fit':relative,'primal':P,'dual_bound':D,
                                                         'dual_sum_residual':float(abs(v.sum()))})

    def run(self):
        selections = {}
        for task in self.cfg['tasks']:
            name = task['name']
            for method in self.cfg['methods']:
                candidates = [self.attempt(f'{name}/{method}/candidate{i}',self.candidate,name,method,i)
                              for i in range(self.cfg['candidate_count_per_method'])]
                selections[(name,method)] = self.attempt(f'{name}/{method}/selection',self.selection,name,method,candidates)
                for seed in ([0] if method == 'HistGBR' else self.cfg['seeds']):
                    if self.lock_at_start is None:
                        self.pending.append(f'{name}/{method}/final_seed{seed}/result.json')
                    else:
                        self.attempt(f'{name}/{method}/seed{seed}',self.result,name,method,seed,selections[(name,method)])
            print(json.dumps({'verified_task':name,'candidate_validations':self.audit['candidate_validations'],
                              'results':self.audit['results'],'errors':len(self.errors)}),flush=True)
        if self.lock_at_start is not None:
            lock = self.lock_at_start
            expected = {f'{task}/{method}/selection.json' for task,method in selections}
            check(lock['protocol_sha256'] == digest(self.directory/'protocol.json'),'Locked protocol hash differs')
            check(set(lock['selection_files']) == expected,'Lock omits or duplicates selections')
            for path, expected_hash in lock['selection_files'].items():
                check(digest(self.directory/path) == expected_hash,'Locked selection file changed')
            check(len(lock['selections']) == len(selections),'Embedded lock length differs')
            embedded = {(row['task'],row['method']):row for row in lock['selections']}
            check(embedded == selections,'Embedded lock contents differ from independently checked selections')
        else:
            check(not self.final_records,'Test results exist without a selection lock')
            self.pending.append('selection_lock.json')
        kernel_fits = sorted(path for path,item in self.validated_fits.items() if item['method'] != 'HistGBR')
        # Prefer completed final fits, spread across task/method/seed/quantile;
        # no score influences the deterministic sampling of verification work.
        final_fits = sorted({item['path'] for record in self.final_records if record['method'] != 'HistGBR'
                             for item in record['fits']})
        pool = final_fits or kernel_fits
        if self.sample_models < 0:
            chosen = kernel_fits
        elif len(pool) <= self.sample_models:
            chosen = pool
        elif self.sample_models:
            chosen = [pool[i] for i in np.linspace(0,len(pool)-1,self.sample_models,dtype=int)]
        else:
            chosen = []
        for index, relative in enumerate(chosen):
            self.attempt(relative+'/model',self.model,relative)
            print(json.dumps({'reconstructed_model':index+1,'of':len(chosen),'errors':len(self.errors)}),flush=True)
        expected_selections = len(self.cfg['tasks'])*len(self.cfg['methods'])
        expected_candidates = expected_selections*self.cfg['candidate_count_per_method']
        expected_results = len(self.cfg['tasks'])*sum(1 if m == 'HistGBR' else len(self.cfg['seeds']) for m in self.cfg['methods'])
        expected_fits = expected_candidates + len(self.cfg['tasks'])*sum(
            len(self.cfg['taus'])-1 if m == 'HistGBR' else len(self.cfg['seeds'])*len(self.cfg['taus'])-1
            for m in self.cfg['methods'])
        complete = self.manifest['status'] == 'complete'
        if complete or not self.allow_partial:
            check(not self.pending,'Required completed-study artifacts are missing')
            check(self.audit['candidate_validations'] == expected_candidates,'Wrong total candidate count')
            check(self.audit['selections'] == expected_selections,'Wrong total selection count')
            check(self.audit['results'] == expected_results,'Wrong total result count')
            check(len(list(self.directory.glob('*/*/**/fit.json'))) == expected_fits,'Wrong unique fitted-model count')
            aggregate = json.loads((self.directory/'results.json').read_text())
            expected_records = {(r['task'],r['method'],r['seed']):r for r in self.final_records}
            check(len(aggregate) == len(expected_records),'Aggregate results missing/duplicated')
            check({(r['task'],r['method'],r['seed']):r for r in aggregate} == expected_records,'Aggregate results differ')
        return {'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(self.directory),
                'status':'failed' if self.errors else ('verified_complete' if complete else 'verified_partial'),
                'manifest_status':self.manifest['status'],'expected':{'selections':expected_selections,
                    'candidate_fits':expected_candidates,'final_results':expected_results,'unique_fits':expected_fits},
                **self.audit,'validated_unique_fits':len(self.validated_fits),'pending':self.pending,'errors':self.errors,
                'scope':'All saved metrics/selection/calibration checks; only listed kernel fits have independently reconstructed objectives and predictions. No test-based model selection or training performed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--sample-models',type=int,default=15,help='Deterministic kernel-fit sample; -1 checks every fit; 0 skips reconstruction')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    verifier = None
    try:
        with threadpool_limits(1):
            verifier = Verifier(args.directory,args.allow_partial,args.sample_models)
            report = verifier.run()
    except Exception as error:
        preceding = verifier.errors if verifier is not None else []
        report = {'status':'failed','directory':str(args.directory),
                  'errors':preceding+[{'error':f'{type(error).__name__}: {error}'}]}
    output = args.output or args.directory/'independent_verification.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('status','errors')}),flush=True)
    sys.exit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
