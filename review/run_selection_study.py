#!/usr/bin/env python3
"""Validation-selected follow-up; freeze all selections before evaluating tests.

The old temporal study and its outputs remain immutable. This experiment uses
the same already-inspected test period, so it is a follow-up, not an independent
confirmation. Every candidate and every prescribed comparison is retained.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, time
from datetime import datetime, timezone
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.selection_data import load_all_selection_data, digest
from review.selection_sampling import stratified_target_indices
from review.convex_refinement import refine_pinball

CAPACITY = {1:99., 2:200., 3:99., 4:66., 5:36., 6:96.}
METHODS = ['Full', 'NoVariance', 'NoDiscrepancy', 'WeightedSVQR', 'TargetSVQR', 'HistGBR']

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def point_metrics(y, p):
    den = np.sum((y-y.mean())**2)
    return {'n':len(y), 'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'mae':float(np.mean(abs(y-p))), 'r2':float(1-np.sum((y-p)**2)/den) if den else None}

def interval_metrics(y, lo, hi, alpha=.05):
    if np.any(lo>hi): raise ValueError('Unordered interval')
    return {'n':len(y), 'picp':float(np.mean((y>=lo)&(y<=hi))),
            'mpiw':float(np.mean(hi-lo)),
            'winkler':float(np.mean(hi-lo+2/alpha*(np.maximum(lo-y,0)+np.maximum(y-hi,0))))}

def conformal_quantile(scores, alpha=.05):
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if not 1<=rank<=len(scores): raise ValueError('Insufficient calibration scores')
    return float(np.partition(scores,rank-1)[rank-1]),rank

def rolling_expansions(calibration_scores, test_scores, window=256, alpha=.05):
    # At test t only Q2 and outcomes of strictly earlier evaluated forecasts
    # are available. Append score t only AFTER calculating expansion t.
    history=list(np.asarray(calibration_scores,dtype=float)[-window:])
    expansion=[]
    for observed_score in test_scores:
        q,_=conformal_quantile(np.asarray(history),alpha)
        expansion.append(q)
        history.append(float(observed_score))
        history=history[-window:]
    return np.asarray(expansion)

def preprocess(x,y):
    valid=np.isfinite(x);count=valid.sum(0)
    mean=np.divide(np.where(valid,x,0).sum(0,dtype=np.float64),count,
                   out=np.zeros(x.shape[1]),where=count>0)
    filled=np.where(valid,x,mean);scale=filled.std(0);scale[scale==0]=1
    ym=float(y.mean());ys=float(y.std()) or 1.
    state={'mean':mean.tolist(),'scale':scale.tolist(),'label_mean':ym,'label_scale':ys}
    def transform(xx):return (np.where(np.isfinite(xx),xx,mean)-mean)/scale
    return transform(x),(y-ym)/ys,transform,state

def protocol():
    design=qmc.LatinHypercube(d=5,seed=20261022).random(15)
    candidates=[{'gamma':.003,'total_loss':1000.,'target_fraction':.5,'lambda1':1.,'lambda2':1.}]
    for row in design:
        candidates.append({'gamma':float(10**(-4+3*row[0])),
                           'total_loss':float(10**(2+3*row[1])),
                           'target_fraction':float(.05+.90*row[2]),
                           'lambda1':float(10**(-2+5*row[3])),
                           'lambda2':float(10**(-2+5*row[4]))})
    rng=np.random.default_rng(20261022)
    grid=[{'learning_rate':lr,'max_leaf_nodes':leaves,'min_samples_leaf':ms,
           'l2_regularization':l2,'max_iter':iterations,'target_fraction':fraction}
          for lr in [.03,.1] for leaves in [7,15,31] for ms in [10,20,40]
          for l2 in [0.,1.,10.] for iterations in [100,200] for fraction in [.1,.5,.9]]
    trees=[grid[i] for i in rng.choice(len(grid),16,replace=False)]
    return {'version':'selection_v1','design_seed':20261022,'feature_track':'meteorology_only_312',
            'test_status':'2020 Q3-Q4 was inspected in chronological_v1; reused-test follow-up',
            'fit_period':'2019','selection_period':'2020 Q1','calibration_period':'2020 Q2',
            'test_period':'2020 Q3-Q4','methods':METHODS,'kernel_candidates':candidates,
            'tree_candidates':trees,'candidate_count_per_method':16,
            'selection_rule':'minimum target Q1 RMSE at tau=0.5; ties use candidate index; no convergence filtering',
            'interval_selection':'reuse point-selected settings for tau=0.025,0.975; not separately interval-optimal',
            'refit_rule':'2019 fitting rows only; validation labels are not included in final fitting',
            'source_cap_per_other_site':1500,'within_site_source_cap':7500,'cross_target_fit_budget':128,
            'target_sampling':'month x disjoint event stratum; one per nonempty cell, proportional remaining quota; time-even within cell',
            'landmarks':128,'landmark_pool_source_cap':2000,'landmark_pool_target_cap':2000,
            'landmark_rule':'KMeans n_init=10 on fitting-only source+target pools; no source pool for TargetSVQR',
            'seeds':[0,1,2],'selection_seed':0,'taus':[.5,.025,.975],
            'solver':{'max_iter':3000,'gap_rtol':.001,'gap_atol':1e-7,'check_every':25},
            'alpha':.05,'rolling_window':256,'threads':1,
            'online_protocol':'after each extreme-target forecast, reveal its outcome before the next evaluated target forecast; no model refitting',
            'label_accounting':'128 fitting + all Q1 target validation + all Q2 target calibration; rolling also uses previously observed test outcomes',
            'tasks':[{'name':'within_site6','site':6,'mode':'within'}]+
                    [{'name':f'cross_site{s}','site':s,'mode':'cross'} for s in range(1,7)]}

class Study:
    def __init__(self,out,cfg):
        self.out=out;self.cfg=cfg;self.data=load_all_selection_data(include_past_power=False)
        self.context_cache={};self.feature_cache={}
        save(out/'data_provenance.json',{str(s):{'provenance':d.provenance,'preprocessing':d.preprocessing,
                                                'statistics':d.statistics} for s,d in self.data.items()})

    def context(self,task,method):
        key=(task['name'],method=='TargetSVQR')
        if key in self.context_cache:return self.context_cache[key]
        d=self.data[task['site']];site=task['site'];folder=self.out/task['name']/('target_data' if key[1] else 'pooled_data')
        folder.mkdir(parents=True,exist_ok=True)
        target_ids=np.flatnonzero(d.mask('train','target'))
        if task['mode']=='cross':target_ids,sampling=stratified_target_indices(d,self.cfg['cross_target_fit_budget'])
        else:sampling={'rule':'all eligible within-farm training target windows','n':len(target_ids)}
        source_parts=[]
        if not key[1]:
            source_sites=[site] if task['mode']=='within' else [s for s in self.data if s!=site]
            cap=self.cfg['within_site_source_cap'] if task['mode']=='within' else self.cfg['source_cap_per_other_site']
            for s in source_sites:
                ids=np.flatnonzero(self.data[s].mask('train','source'))
                if len(ids)>cap:ids=ids[np.linspace(0,len(ids)-1,cap,dtype=int)]
                source_parts.append((s,ids))
        source_n=sum(len(ids) for _,ids in source_parts)
        xs=[self.data[s].X[ids] for s,ids in source_parts]
        ys=[self.data[s].y_raw[ids]/CAPACITY[s] for s,ids in source_parts]
        x=np.concatenate(xs+[d.X[target_ids]])
        y=np.concatenate(ys+[d.y_raw[target_ids]/CAPACITY[site]])
        x,y,transform,state=preprocess(x,y)
        ids={split:np.flatnonzero(d.mask(split,'target')) for split in ['validation','calibration','test']}
        save(folder/'preprocessing.json',state)
        save(folder/'sampling.json',sampling)
        np.savez_compressed(folder/'ids.npz',source_site=np.concatenate([np.full(len(ii),s) for s,ii in source_parts]) if source_n else np.array([],int),
                            source_window=np.concatenate([ii for _,ii in source_parts]) if source_n else np.array([],int),
                            target_window=target_ids,**{f'{split}_window':ii for split,ii in ids.items()})
        counts={'n_source':source_n,'n_target_fit':len(target_ids),**{f'n_{s}':len(ii) for s,ii in ids.items()},
                'events':{s:{event:int(d.event_masks[event][ii].sum()) for event in d.event_masks} for s,ii in ids.items()}}
        save(folder/'counts.json',counts)
        ctx={'folder':folder,'x':x,'y':y,'transform':transform,'state':state,'source_n':source_n,
             'ids':ids,'site':site,'data':d,'counts':counts}
        self.context_cache[key]=ctx;return ctx

    def features(self,ctx,seed,gamma):
        key=(str(ctx['folder']),seed,gamma)
        if key in self.feature_cache:return self.feature_cache[key]
        x=ctx['x'];ns=ctx['source_n'];seedfolder=ctx['folder']/f'features_seed{seed}';seedfolder.mkdir(exist_ok=True)
        landmarks_file=seedfolder/'landmarks.npy'
        start=time.perf_counter()
        if landmarks_file.exists():centers=np.load(landmarks_file)
        else:
            rng=np.random.default_rng(seed)
            si=rng.choice(ns,min(ns,self.cfg['landmark_pool_source_cap']),replace=False) if ns else np.array([],int)
            ti=ns+rng.choice(len(x)-ns,min(len(x)-ns,self.cfg['landmark_pool_target_cap']),replace=False)
            pool=x[np.r_[si,ti]]
            centers=pool if len(pool)<=self.cfg['landmarks'] else KMeans(n_clusters=self.cfg['landmarks'],n_init=10,random_state=seed).fit(pool).cluster_centers_
            np.save(landmarks_file,centers)
        k=rbf_kernel(centers,gamma=gamma)
        values,vectors=np.linalg.eigh((k+k.T)*.5+1e-6*np.eye(len(k)))
        if np.any(values<=0):raise ValueError('Nonpositive stabilized landmark Gram eigenvalue')
        normalization=(vectors/np.sqrt(values))@vectors.T
        z=rbf_kernel(x,centers,gamma=gamma)@normalization
        folder=seedfolder/('gamma_'+format(gamma,'.17g'));folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'map.npz',centers=centers,normalization=normalization,gamma=gamma)
        result={'z':z,'centers':centers,'normalization':normalization,'gamma':gamma,
                'path':folder/'map.npz','construction_seconds':time.perf_counter()-start}
        # Keep only a modest number of transformed training matrices in memory.
        if len(self.feature_cache)>=20:self.feature_cache.pop(next(iter(self.feature_cache)))
        self.feature_cache[key]=result;return result

    def candidate(self,task,method,index):
        return self.cfg['tree_candidates' if method=='HistGBR' else 'kernel_candidates'][index]

    def fit(self,task,method,index,seed,tau,folder):
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
        ctx=self.context(task,method);candidate=self.candidate(task,method,index)
        ns=ctx['source_n'];nt=len(ctx['y'])-ns
        fraction=candidate['target_fraction'] if ns else 1.
        start=time.perf_counter()
        if method=='HistGBR':
            opts={k:v for k,v in candidate.items() if k!='target_fraction'}
            model=HistGradientBoostingRegressor(loss='quantile',quantile=tau,early_stopping=False,
                                                random_state=seed,**opts)
            weights=np.r_[np.full(ns,(1-fraction)/ns) if ns else np.array([]),np.full(nt,fraction/nt)]*len(ctx['y'])
            model.fit(ctx['x'],ctx['y'],sample_weight=weights)
            joblib.dump(model,folder/'model.joblib')
            info={'solver':'histogram gradient boosting','elapsed_seconds':time.perf_counter()-start,
                  'iterations':model.n_iter_,'converged':None}
            map_path=None
        else:
            feature=self.features(ctx,seed,candidate['gamma']);z=feature['z'];m=z.shape[1]
            lam1=candidate['lambda1'] if method in ['Full','NoDiscrepancy'] else 0.
            lam2=candidate['lambda2'] if method in ['Full','NoVariance'] else 0.
            r=np.eye(m);c=np.zeros(m)
            if ns:
                centered=z[:ns]-z[:ns].mean(0);yc=ctx['y'][:ns]-ctx['y'][:ns].mean()
                delta=z[:ns].mean(0)-z[ns:].mean(0)
                r+=lam1*(centered.T@centered)/ns+lam2*np.outer(delta,delta)
                c+=lam1*(centered.T@yc)/ns
            total=candidate['total_loss']
            weights=np.r_[np.full(ns,total*(1-fraction)/ns) if ns else np.array([]),np.full(nt,total*fraction/nt)]
            solution=refine_pinball(z,ctx['y'],r,c,weights,tau,**self.cfg['solver'])
            np.savez_compressed(folder/'model.npz',w=solution.w,b=solution.b,dual=solution.dual_variable,R=r,c=c,C=weights,y=ctx['y'])
            info=solution.report();info['solver']='ADMM';info['fit_seconds_including_feature_access']=time.perf_counter()-start
            map_path=str(feature['path'].relative_to(self.out))
        record={'task':task['name'],'site':task['site'],'method':method,'candidate_index':index,'candidate':candidate,
                'seed':seed,'tau':tau,'feature_map':map_path,'data_folder':str(ctx['folder'].relative_to(self.out)),
                'diagnostics':info,'counts':ctx['counts']}
        save(folder/'fit.json',record)
        return record

    def predict(self,ctx,folder,split):
        record=json.loads((folder/'fit.json').read_text());ids=ctx['ids'][split]
        x=ctx['transform'](ctx['data'].X[ids])
        if record['method']=='HistGBR':pred=joblib.load(folder/'model.joblib').predict(x)
        else:
            mp=np.load(self.out/record['feature_map']);model=np.load(folder/'model.npz')
            pred=(rbf_kernel(x,mp['centers'],gamma=float(mp['gamma']))@mp['normalization'])@model['w']+float(model['b'])
        state=ctx['state'];pred=(pred*state['label_scale']+state['label_mean'])*CAPACITY[ctx['site']]
        return pred

    def tune(self):
        selections=[]
        for task in self.cfg['tasks']:
            for method in self.cfg['methods']:
                ctx=self.context(task,method);rows=[];start=time.perf_counter()
                for index in range(self.cfg['candidate_count_per_method']):
                    folder=self.out/task['name']/method/'candidates'/f'c{index:02d}'
                    rec=self.fit(task,method,index,0,.5,folder)
                    prediction=self.predict(ctx,folder,'validation');ids=ctx['ids']['validation']
                    metrics=point_metrics(ctx['data'].y_raw[ids],prediction)
                    pd.DataFrame({'window_index':ids,'y_true':ctx['data'].y_raw[ids],'prediction':prediction}).to_csv(folder/'validation.csv',index=False)
                    save(folder/'validation_metrics.json',metrics)
                    rows.append({'candidate_index':index,'validation':metrics,'diagnostics':rec['diagnostics']})
                best=min(rows,key=lambda row:(row['validation']['rmse'],row['candidate_index']))
                selection={'task':task['name'],'site':task['site'],'method':method,'selected_index':best['candidate_index'],
                           'selected_validation':best['validation'],'candidates':rows,'selection_seconds':time.perf_counter()-start}
                selections.append(selection)
                save(self.out/task['name']/method/'selection.json',selection)
                print(json.dumps({'phase':'selection','task':task['name'],'method':method,'selected':best['candidate_index'],'validation_rmse':best['validation']['rmse']}),flush=True)
            self.context_cache.clear();self.feature_cache.clear()
        # All 42 selections are locked before any test prediction/score is made.
        lock={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(self.out/'protocol.json'),
              'selections':selections,'selection_files':{str(p.relative_to(self.out)):digest(p) for p in sorted(self.out.glob('*/*/selection.json'))}}
        save(self.out/'selection_lock.json',lock)
        return selections

    def evaluate(self,selections):
        lookup={(row['task'],row['method']):row for row in selections};results=[]
        for task in self.cfg['tasks']:
            for method in self.cfg['methods']:
                ctx=self.context(task,method);index=lookup[(task['name'],method)]['selected_index']
                seeds=[0] if method=='HistGBR' else self.cfg['seeds']
                for seed in seeds:
                    base=self.out/task['name']/method/f'final_seed{seed}'
                    if (base/'result.json').exists():results.append(json.loads((base/'result.json').read_text()));continue
                    fits=[];test=[];cal=[]
                    for tau in self.cfg['taus']:
                        if tau==.5 and seed==0:folder=self.out/task['name']/method/'candidates'/f'c{index:02d}'
                        else:folder=base/f'tau{tau}'
                        rec=self.fit(task,method,index,seed,tau,folder)
                        fits.append({'path':str(folder.relative_to(self.out)),**rec})
                        test.append(self.predict(ctx,folder,'test'))
                        cal.append(self.predict(ctx,folder,'calibration'))
                    base.mkdir(parents=True,exist_ok=True)
                    ti=ctx['ids']['test'];ci=ctx['ids']['calibration'];d=ctx['data'];y=d.y_raw[ti];cy=d.y_raw[ci]
                    lo=np.minimum(test[1],test[2]);hi=np.maximum(test[1],test[2])
                    clo=np.minimum(cal[1],cal[2]);chi=np.maximum(cal[1],cal[2])
                    scores=np.maximum.reduce([clo-cy,cy-chi,np.zeros(len(ci))])
                    test_scores=np.maximum.reduce([lo-y,y-hi,np.zeros(len(ti))])
                    q,rank=conformal_quantile(scores,self.cfg['alpha'])
                    rq=rolling_expansions(scores,test_scores,self.cfg['rolling_window'],self.cfg['alpha'])
                    pd.DataFrame({'window_index':ti,'time':d.timestamps[ti].astype(str),'y_true':y,'prediction':test[0],
                                  'raw_q025':test[1],'raw_q975':test[2],'static_lower':lo-q,'static_upper':hi+q,
                                  'rolling_q':rq,'rolling_lower':lo-rq,'rolling_upper':hi+rq}).to_csv(base/'predictions.csv',index=False)
                    pd.DataFrame({'window_index':ci,'time':d.timestamps[ci].astype(str),'y_true':cy,'q025':cal[1],'q975':cal[2],'score':scores}).to_csv(base/'calibration.csv',index=False)
                    result={'task':task['name'],'site':task['site'],'method':method,'seed':seed,'selected_index':index,
                            'counts':ctx['counts'],'point':point_metrics(y,test[0]),'raw_interval':interval_metrics(y,lo,hi),
                            'static_interval':interval_metrics(y,lo-q,hi+q),'rolling_interval':interval_metrics(y,lo-rq,hi+rq),
                            'calibration':{'n':len(ci),'rank':rank,'static_q':q,'rolling_window':self.cfg['rolling_window']},'fits':fits}
                    save(base/'result.json',result);results.append(result)
                    print(json.dumps({'phase':'evaluation','task':task['name'],'method':method,'seed':seed,'rmse':result['point']['rmse'],'static_picp':result['static_interval']['picp'],'rolling_picp':result['rolling_interval']['picp']}),flush=True)
            self.context_cache.clear();self.feature_cache.clear()
        save(self.out/'results.json',results)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'review/selection_runs/selection_v1')
    parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=protocol();paths=[Path(__file__),ROOT/'review/convex_refinement.py',ROOT/'review/selection_data.py',ROOT/'review/selection_sampling.py',ROOT/'review/temporal_data.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    data_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'data/wind_farm_data/data_original/wind_farms').glob('*.xlsx'))}
    reference_hash=digest(ROOT/'review/replied_comment_runs/chronological_v1/manifest.json')
    if args.output.exists():
        manifest=json.loads((args.output/'manifest.json').read_text())
        if not args.resume:raise FileExistsError('Use explicit resume with identical frozen source hashes.')
        if manifest['source_hashes']!=hashes or manifest['protocol']!=cfg:raise ValueError('Frozen experiment source/protocol changed')
        if manifest['data_hashes']!=data_hashes or manifest['reference_manifest_sha256']!=reference_hash:raise ValueError('Frozen source data or reference manifest changed')
    else:
        args.output.mkdir(parents=True)
        manifest={'status':'frozen','created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':hashes,'protocol':cfg,
                  'python':platform.python_version(),'numpy':np.__version__,'data_hashes':data_hashes,'reference_manifest_sha256':reference_hash}
        save(args.output/'protocol.json',cfg);save(args.output/'manifest.json',manifest)
    if args.freeze_only:return
    manifest['status']='running';save(args.output/'manifest.json',manifest)
    try:
        with threadpool_limits(limits=cfg['threads']):
            study=Study(args.output,cfg)
            if (args.output/'selection_lock.json').exists():
                lock=json.loads((args.output/'selection_lock.json').read_text())
                if lock['protocol_sha256']!=digest(args.output/'protocol.json'):raise ValueError('Selection lock protocol mismatch')
                expected={f"{task['name']}/{method}/selection.json" for task in cfg['tasks'] for method in cfg['methods']}
                if set(lock['selection_files'])!=expected:raise ValueError('Selection lock list incomplete')
                if any(digest(args.output/p)!=h for p,h in lock['selection_files'].items()):raise ValueError('Selection lock changed')
                selections=lock['selections']
                if len(selections)!=len(expected):raise ValueError('Duplicate or missing locked selections')
                if {f"{row['task']}/{row['method']}/selection.json" for row in selections}!=expected:raise ValueError('Embedded selection list mismatch')
                if any(row!=json.loads((args.output/row['task']/row['method']/'selection.json').read_text()) for row in selections):raise ValueError('Embedded lock differs from selection records')
            else:selections=study.tune()
            study.evaluate(selections)
        manifest['status']='complete'
    except Exception as error:
        manifest['status']='failed';manifest['error']=repr(error);raise
    finally:
        manifest['updated_utc']=datetime.now(timezone.utc).isoformat();save(args.output/'manifest.json',manifest)

if __name__=='__main__':main()
