#!/usr/bin/env python3
"""Produce complete-study tables from all saved follow-up results, without fitting.

Refuses partial seed sets and incomplete studies. Recomputes every displayed
metric from saved prediction rows; no result selection or missing-value filling.
Kernel means summarize three algorithmic seeds, trees one. These are not
independent-data replicates or confidence intervals. The reused test period
remains a follow-up evaluation, not an independent confirmatory experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.verify_selection_study import point_metrics, interval_metrics, metrics_equal, check


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def aggregate(records):
    keys=set(records[0])
    check(all(set(row)==keys for row in records),'Metric schemas differ across seeds')
    answer={}
    for key in sorted(keys):
        values=[row[key] for row in records]
        if any(value is None for value in values):
            check(all(value is None for value in values),f'{key}: only some seed metrics are undefined')
            answer[key]=None
            continue
        x=np.asarray(values,float)
        check(np.isfinite(x).all(),f'{key}: nonfinite metric')
        answer[key]={'mean':float(x.mean()),'min':float(x.min()),'max':float(x.max()),
                     'sd':float(x.std(ddof=1)) if len(x)>1 else None,'runs':len(x)}
    return answer


def num(value, digits=3):
    return '---' if value is None else f'{value:.{digits}f}'


def escape(text):
    return str(text).replace('_',r'\_').replace('%',r'\%')


def task_label(task):
    return 'Within 6' if task=='within_site6' else 'Cross '+task.removeprefix('cross_site')


def mean_quantile_score(y, low, high):
    """Mean of the two pinball losses, in the target's MW units."""
    values=[]
    for tau,prediction in ((.025,low),(.975,high)):
        residual=y-prediction
        values.append(np.maximum(tau*residual,(tau-1.)*residual))
    return {'mean_pinball':float(np.mean(values))}


def latex_table(caption,label,header,rows,*,small=True):
    lines=[r'\begin{table}[htbp]',r'\centering',r'\color{blue}']
    if small:lines.append(r'\small')
    lines += [r'\caption{\textcolor{blue}{'+caption+'}}',r'\label{'+label+'}',
              r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+'l'*len(header)+'}',
              r'\toprule',' & '.join(header)+r' \\',r'\midrule']
    lines += [' & '.join(str(value) for value in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}','']
    return '\n'.join(lines)


def build(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    cfg=json.loads((directory/'protocol.json').read_text())
    check(manifest['status']=='complete','Study is incomplete; no tables generated')
    check(manifest['protocol']==cfg,'Manifest/protocol mismatch')
    for path,value in manifest['source_hashes'].items():
        check(digest(ROOT/path)==value,'Frozen source changed: '+path)
    for path,value in manifest['data_hashes'].items():
        check(digest(ROOT/path)==value,'Frozen input changed: '+path)
    lock=json.loads((directory/'selection_lock.json').read_text())
    check(lock['protocol_sha256']==digest(directory/'protocol.json'),'Selection lock protocol differs')
    methods=cfg['methods'];tasks=cfg['tasks'];tau_list=cfg['taus']
    expected_selections={(task['name'],method) for task in tasks for method in methods}
    selected={(row['task'],row['method']):row for row in lock['selections']}
    check(len(lock['selections'])==len(expected_selections) and set(selected)==expected_selections,'Incomplete/duplicate selection lock')
    for path,expected_hash in lock['selection_files'].items():
        check(digest(directory/path)==expected_hash,'Locked selection changed: '+path)
    results=json.loads((directory/'results.json').read_text())
    lookup={(r['task'],r['method'],r['seed']):r for r in results}
    expected_results={(task['name'],method,seed) for task in tasks for method in methods
                      for seed in ([0] if method=='HistGBR' else cfg['seeds'])}
    check(len(results)==len(expected_results) and set(lookup)==expected_results,'Missing or duplicate final seed results')
    hashes={};groups=[];selected_fit_paths=set()
    for task in tasks:
        for method in methods:
            seeds=[0] if method=='HistGBR' else cfg['seeds']
            entries=[]
            for seed in seeds:
                key=(task['name'],method,seed);record=lookup[key]
                base=directory/task['name']/method/f'final_seed{seed}'
                check(json.loads((base/'result.json').read_text())==record,'Aggregate result differs from branch')
                table=pd.read_csv(base/'predictions.csv')
                y=table['y_true'].to_numpy(float)
                check(np.isfinite(table.select_dtypes(include='number').to_numpy()).all(),'Nonfinite prediction table')
                calculated={'point':point_metrics(y,table['prediction'].to_numpy(float))}
                calculated_score=mean_quantile_score(
                    y,table['raw_q025'].to_numpy(float),table['raw_q975'].to_numpy(float))
                lo=np.minimum(table['raw_q025'],table['raw_q975']).to_numpy(float)
                hi=np.maximum(table['raw_q025'],table['raw_q975']).to_numpy(float)
                calculated['raw_interval']=interval_metrics(y,lo,hi,cfg['alpha'])
                for interval in ('static','rolling'):
                    calculated[interval+'_interval']=interval_metrics(
                        y,table[interval+'_lower'].to_numpy(float),table[interval+'_upper'].to_numpy(float),cfg['alpha'])
                for family,values in calculated.items():
                    metrics_equal(record[family],values,f'{key}/{family}')
                check(record['selected_index']==selected[(task['name'],method)]['selected_index'],'Unexpected selected candidate')
                check(len(record['fits'])==len(tau_list),'Missing final quantile fit')
                selected_fit_paths.update(fit['path'] for fit in record['fits'])
                for name in ('result.json','predictions.csv','calibration.csv'):
                    path=base/name;hashes[str(path.relative_to(directory))]=digest(path)
                entries.append({'seed':seed,**calculated,'quantile_score':calculated_score})
            first=lookup[(task['name'],method,seeds[0])]
            check(all(lookup[(task['name'],method,s)]['counts']==first['counts'] for s in seeds),'Seed test/fitting populations differ')
            groups.append({'task':task['name'],'site':task['site'],'mode':task['mode'],'method':method,
                           'seeds':seeds,'runs':len(seeds),'selected_index':first['selected_index'],
                           'counts':first['counts'],'statistics':{
                               family:aggregate([entry[family] for entry in entries])
                               for family in ('point','quantile_score','raw_interval','static_interval','rolling_interval')},
                           'per_seed':entries})
    by_group={(row['task'],row['method']):row for row in groups}
    config_rows=[]
    for task in tasks:
        for method in methods:
            selection=selected[(task['name'],method)]
            path=directory/task['name']/method/'selection.json'
            check(json.loads(path.read_text())==selection,'Selection lock differs from separate file')
            candidates=selection['candidates']
            check(len(candidates)==cfg['candidate_count_per_method'],'Candidate omitted from selection')
            check([r['candidate_index'] for r in candidates]==list(range(cfg['candidate_count_per_method'])),'Candidate order/IDs differ')
            best=min(candidates,key=lambda row:(row['validation']['rmse'],row['candidate_index']))
            check(selection['selected_index']==best['candidate_index'],'Selection violates declared validation argmin')
            for index in range(cfg['candidate_count_per_method']):
                candidate_path=directory/task['name']/method/'candidates'/f'c{index:02d}'
                check(all((candidate_path/name).exists() for name in ('fit.json','validation.csv','validation_metrics.json')),'Missing candidate artifact')
            grid=cfg['tree_candidates' if method=='HistGBR' else 'kernel_candidates']
            chosen=dict(grid[selection['selected_index']])
            if method!='HistGBR':
                chosen['lambda1']=chosen['lambda1'] if method in ('Full','NoDiscrepancy') else 0.
                chosen['lambda2']=chosen['lambda2'] if method in ('Full','NoVariance') else 0.
                if method=='TargetSVQR':chosen['target_fraction']=1.
            config_rows.append({'task':task['name'],'site':task['site'],'method':method,
                                'candidate_index':selection['selected_index'],'effective_parameters':chosen,
                                'validation':selection['selected_validation']})
    fits={str(path.parent.relative_to(directory)):json.loads(path.read_text())
          for path in sorted(directory.glob('*/*/**/fit.json'))}
    expected_fit_count=len(expected_selections)*cfg['candidate_count_per_method']+len(tasks)*sum(
        len(tau_list)-1 if method=='HistGBR' else len(cfg['seeds'])*len(tau_list)-1 for method in methods)
    check(len(fits)==expected_fit_count,'Unique fitted-model count differs from protocol')
    check(selected_fit_paths <= fits.keys(),'Final result references a missing fitted model')
    kernels={p:r for p,r in fits.items() if r['method']!='HistGBR'}
    trees={p:r for p,r in fits.items() if r['method']=='HistGBR'}
    final_kernels={p:kernels[p] for p in selected_fit_paths if p in kernels}
    def convergence(records):
        if not records:return {'fits':0}
        rows=list(records.values());gaps=np.array([r['diagnostics']['relative_gap'] for r in rows])
        return {'fits':len(rows),'converged':sum(bool(r['diagnostics']['converged']) for r in rows),
                'not_converged':sum(not bool(r['diagnostics']['converged']) for r in rows),
                'max_relative_gap':float(gaps.max()),'median_relative_gap':float(np.median(gaps)),
                'max_absolute_gap':max(r['diagnostics']['gap'] for r in rows),
                'max_iterations':max(r['diagnostics']['iterations'] for r in rows),
                'nonconverged_fits':[{'path':p,'relative_gap':r['diagnostics']['relative_gap'],
                                    'iterations':r['diagnostics']['iterations']} for p,r in records.items()
                                   if not r['diagnostics']['converged']]}
    created=datetime.fromisoformat(manifest['created_utc'])
    updated=datetime.fromisoformat(manifest['updated_utc'])
    envelope=(updated-created).total_seconds()
    kernel_seconds=sum(r['diagnostics']['elapsed_seconds'] for r in kernels.values())
    tree_seconds=sum(r['diagnostics']['elapsed_seconds'] for r in trees.values())
    counts={'unique_fits':len(fits),'unique_kernel_fits':len(kernels),'unique_tree_fits':len(trees),
            'candidate_fits':len(expected_selections)*cfg['candidate_count_per_method'],
            'final_results':len(results),'final_selected_unique_fits':len(selected_fit_paths),
            'all_kernel_convergence':convergence(kernels),'final_selected_kernel_convergence':convergence(final_kernels),
            'final_selected_point_convergence':convergence({p:r for p,r in final_kernels.items() if r['tau']==.5}),
            'final_selected_interval_convergence':convergence({p:r for p,r in final_kernels.items() if r['tau']!=.5}),
            'fit_counts_by_method':dict(Counter(r['method'] for r in fits.values())),
            'times':{
                'run_created_to_updated_seconds':envelope,
                'sum_unique_kernel_admm_seconds':kernel_seconds,
                'sum_unique_kernel_feature_access_fit_save_seconds':sum(r['diagnostics']['fit_seconds_including_feature_access'] for r in kernels.values()),
                'sum_unique_tree_fit_save_seconds':tree_seconds,
                'sum_selection_loop_seconds':sum(r['selection_seconds'] for r in selected.values()),
                'timing_scope':'The created-to-updated interval includes setup, waits/resumes, selection and evaluation. Unique kernel ADMM timing excludes feature access and model serialization; tree timing includes fitting and serialization. Feature caches are shared across kernel methods: first construction is charged to the first accessor, so feature-access timing cannot rank standalone method runtime. Selection-loop timing overlaps fitting and includes validation prediction/I/O. These sums must not be added together or presented as matched pure-solver timing.',
                'serial_measured_fit_lower_component_seconds':kernel_seconds+tree_seconds,
                'walltime_bounds_interpretation':'For this sequential runner, the sum of unique ADMM and tree fit/save durations is a measured component of end-to-end time; the created-to-updated envelope is the broad recorded elapsed interval. Neither is deployment latency or a hardware-independent complexity result.'},
            'convergence_interpretation':'Only the given fixed-feature training objective is certified by the numerical gap. Trees do not have these convex-objective certificates. Every unfinished ADMM solve remains reported and was eligible under the predeclared validation-selection rule.'}
    comparisons=[]
    for site in range(1,7):
        task=f'cross_site{site}'
        if (task,'Full') not in by_group:continue
        full=by_group[(task,'Full')]['statistics']
        comparisons.append({'site':site,'full_rmse':full['point']['rmse']['mean'],
                            'target_svqr_rmse':by_group[(task,'TargetSVQR')]['statistics']['point']['rmse']['mean'],
                            'weighted_svqr_rmse':by_group[(task,'WeightedSVQR')]['statistics']['point']['rmse']['mean'],
                            'histgbr_rmse':by_group[(task,'HistGBR')]['statistics']['point']['rmse']['mean'],
                            'full_static_picp':full['static_interval']['picp']['mean'],
                            'full_rolling_picp':full['rolling_interval']['picp']['mean']})
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(directory),
             'manifest_sha256':digest(directory/'manifest.json'),'protocol_sha256':digest(directory/'protocol.json'),
             'selection_lock_sha256':digest(directory/'selection_lock.json'),'protocol':cfg,'groups':groups,
             'selected_configs':config_rows,'solver_counts':counts,'cross_farm_comparisons':comparisons,
             'artifact_hashes':hashes,'interpretation':[
                 'Means are over every prescribed algorithmic seed; they are not ensemble predictions or confidence intervals.',
                 'Point settings were selected by target Q1 RMSE; the same settings were reused for quantile models and are not claimed interval-optimal.',
                 'Static calibration uses Q2 only. Rolling calibration additionally uses outcomes of strictly earlier evaluated test forecasts, under the declared online observation assumption.',
                 'All previously inspected Q3-Q4 outcomes form a reused-test follow-up, not an independent confirmatory evaluation.',
                 'Kernel and tree search budgets match candidate counts, not model capacities, optimization accuracy, or wall-clock cost.',
                 'The 128 cross-farm fitting-label budget excludes additional Q1 selection and Q2 calibration labels; rolling calibration also consumes previously revealed test labels.']}
    def mean(task,method,family,metric):
        result=by_group[(task,method)]['statistics'][family][metric]
        return None if result is None else result['mean']
    main=[]
    rows=[]
    for method in methods:
        rows.append([escape(method),num(mean('within_site6',method,'point','rmse')),
                     num(mean('within_site6',method,'point','mae')),
                     num(mean('within_site6',method,'quantile_score','mean_pinball'))]+
                    [num(mean('within_site6',method,family,key)) for family in ('static_interval','rolling_interval')
                     for key in ('picp','mpiw','winkler')])
    main.append(latex_table('Validation-selected within-Farm~6 comparison. Entries are seed means (three kernel seeds; one HistGBR seed). Pinball is the mean proper quantile score over the raw 0.025 and 0.975 forecasts. S and R denote static Q2 and causal rolling calibration. RMSE, MAE, pinball, interval widths and Winkler scores are in MW. No test result selects a model or a calibration window.',
                           'tab:selected_within',['Method','RMSE','MAE','Pinball','S PICP','S MPIW','S Winkler','R PICP','R MPIW','R Winkler'],rows))
    rows=[[site]+[num(mean(f'cross_site{site}',method,'point','rmse')) for method in methods] for site in range(1,7)]
    main.append(latex_table('Cross-farm point RMSE (MW), retaining every method and target site. Kernel entries average all three prescribed seeds; HistGBR uses seed 0. TargetSVQR uses the same acquired target fitting labels without source labels.',
                           'tab:selected_cross_rmse',['Site']+[escape(m) for m in methods],rows))
    rows=[[site]+[num(mean(f'cross_site{site}','Full',family,key)) for family in ('static_interval','rolling_interval')
                  for key in ('picp','mpiw','winkler')] for site in range(1,7)]
    main.append(latex_table('Full-model cross-farm interval results, averaged over three seeds. Width and Winkler are in MW. Rolling calibration uses previously revealed target outcomes and has a different information budget from static calibration; neither is guaranteed to attain nominal 0.95 under temporal dependence.',
                           'tab:selected_cross_full_intervals',['Site','Static PICP','Static MPIW','Static Winkler','Rolling PICP','Rolling MPIW','Rolling Winkler'],rows))
    appendix=[]
    for start in (1,4):
        rows=[]
        for site in range(start,start+3):
            for method in methods:
                rows.append([site,escape(method),num(mean(f'cross_site{site}',method,'point','mae')),
                             num(mean(f'cross_site{site}',method,'point','r2')),
                             num(mean(f'cross_site{site}',method,'quantile_score','mean_pinball'))]+
                            [num(mean(f'cross_site{site}',method,family,key)) for family in ('static_interval','rolling_interval')
                             for key in ('picp','mpiw','winkler')])
        appendix.append(latex_table(f'Complete cross-farm metrics for target sites {start}--{start+2}. Entries are means over all prescribed seeds (three for kernels, one for HistGBR). Pinball is the mean proper quantile score over the raw 0.025 and 0.975 forecasts. S/R denote static/rolling calibration; MAE, pinball, MPIW and Winkler are in MW. Negative $R^2$ and undercoverage are retained.',
                                   f'tab:selected_cross_all_{start}',
                                   ['Site','Method','MAE','$R^2$','Pinball','S PICP','S MPIW','S Winkler','R PICP','R MPIW','R Winkler'],rows))
    configurations=[]
    kernels_config=[r for r in config_rows if r['method']!='HistGBR']
    for start in range(0,len(kernels_config),15):
        rows=[]
        for row in kernels_config[start:start+15]:
            p=row['effective_parameters']
            rows.append([task_label(row['task']),escape(row['method']),f"c{row['candidate_index']:02d}",
                         f"{p['gamma']:.4g}",f"{p['total_loss']:.4g}",f"{p['target_fraction']:.3f}",
                         f"{p['lambda1']:.4g}",f"{p['lambda2']:.4g}",num(row['validation']['rmse'])])
        configurations.append(latex_table('Locked kernel configurations, selected only by Q1 point RMSE. Candidate indices are zero-based. $A=n_SC_S+n_TC_T$ and $p=n_TC_T/A$ are the total coefficient mass and target fraction. Inactive penalties are shown as zero; TargetSVQR has $p=1$. All use 128 requested landmarks. Validation RMSE is in MW.',
                                         f'tab:selected_kernel_configs_{start//15+1}',
                                         ['Task','Method','ID',r'$\gamma$','$A$','$p$',r'$\lambda_1$',r'$\lambda_2$','Val. RMSE'],rows))
    rows=[]
    for row in config_rows:
        if row['method']!='HistGBR':continue
        p=row['effective_parameters']
        rows.append([task_label(row['task']),f"c{row['candidate_index']:02d}",num(p['learning_rate'],2),
                     p['max_leaf_nodes'],p['min_samples_leaf'],num(p['l2_regularization'],1),p['max_iter'],
                     num(p['target_fraction']),num(row['validation']['rmse'])])
    configurations.append(latex_table('Locked HistGBR settings selected by Q1 point RMSE, with early stopping disabled. Leaves and minimum leaf size control the tree structure; the same selected settings are used for the two quantile fits. Candidate indices are zero-based and validation RMSE is in MW.',
                                     'tab:selected_tree_configs',['Task','ID','Learning rate','Leaves','Min. leaf','$L_2$','Iterations','$p$','Val. RMSE'],rows))
    return summary, counts, '\n'.join(main), '\n'.join(appendix), '\n'.join(configurations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,help='Defaults to the study directory; never writes manuscript sources')
    args=parser.parse_args()
    # Construct and validate every table in memory before writing any output.
    summary,counts,main_tables,appendix,configs=build(args.directory)
    out=args.output or args.directory
    out.mkdir(parents=True,exist_ok=True)
    save(out/'summary.json',summary);save(out/'solver_counts.json',counts)
    (out/'main_tables.tex').write_text(main_tables)
    (out/'appendix_metrics.tex').write_text(appendix)
    (out/'selected_configs.tex').write_text(configs)
    print(json.dumps({'output':str(out),'groups':len(summary['groups']),
                      'final_results':counts['final_results'],'unique_fits':counts['unique_fits'],
                      'nonconverged_final_kernel_fits':counts['final_selected_kernel_convergence']['not_converged']}),flush=True)


if __name__=='__main__':
    main()
