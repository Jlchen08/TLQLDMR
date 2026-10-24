#!/usr/bin/env python3
"""Fixed chronological empirical checks; no tuning or test-based filtering.

Every completed branch stores its model, actual fitting IDs, preprocessing,
test/calibration predictions, and metrics. Historical results are never edited.
Use --resume only with identical protocol/source/data hashes.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, math, platform, re, time
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.temporal_data import TemporalProtocol, load_temporal_farm
from Train_TL_QLDMR import Fast_TL_QLDMR

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def point_metrics(y,p):
    mse=float(np.mean((y-p)**2)); den=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'rmse':math.sqrt(mse),'mae':float(np.mean(abs(y-p))),
            'r2':float(1-np.sum((y-p)**2)/den) if den>0 else None}

def interval_metrics(y,lo,hi,alpha):
    w=hi-lo; picp=float(np.mean((y>=lo)&(y<=hi))); span=float(np.ptp(y))
    return {'n':len(y),'picp':picp,'mpiw':float(w.mean()),'pinaw':float(w.mean()/span) if span>0 else None,
            'winkler':float(np.mean(w+2/alpha*(np.maximum(lo-y,0)+np.maximum(y-hi,0))))}

def event_definition(data,q):
    wind=float(np.percentile(data.training_weather['wind_hub'],q))
    low,high=np.percentile(data.training_weather['temperature'],[100-q,q])
    masks={'wind':data.target_weather['wind_hub']>wind,
           'cold':data.target_weather['temperature']<low,'hot':data.target_weather['temperature']>high,
           'cutout':data.target_weather['wind_hub']>25}
    domain=np.logical_or.reduce(list(masks.values()))
    return domain,masks,{'wind':wind,'temperature_low':float(low),'temperature_high':float(high)}

def temporal_uniform(ids,cap):
    return ids if len(ids)<=cap else ids[np.linspace(0,len(ids)-1,cap,dtype=int)]

def loss_weights(ns,nt,total,scheme,task):
    if not ns:return 0.0,total/nt
    if scheme=='domain_balanced':return total/(2*ns),total/(2*nt)
    if scheme=='sample_equal':return total/(ns+nt),total/(ns+nt)
    if scheme=='historical_ratio':r=3000.0 if task=='point' else 8000.0
    elif scheme=='target_10x':r=10*ns/nt
    else:raise ValueError(scheme)
    return total/(ns+r*nt),r*total/(ns+r*nt)

def preprocess(xs,ys,xt,yt):
    x=np.concatenate([xs,xt]);y=np.r_[ys,yt]
    valid=np.isfinite(x);count=valid.sum(0)
    mean=np.divide(np.where(valid,x,0).sum(0,dtype=np.float64),count,out=np.zeros(x.shape[1]),where=count>0)
    filled=np.where(valid,x,mean);scale=filled.std(0);scale[scale==0]=1
    ym=float(y.mean());yscale=float(y.std()) or 1.0
    state={'mean':mean.tolist(),'scale':scale.tolist(),'label_mean':ym,'label_scale':yscale,
           'fit_source_count':len(xs),'fit_target_count':len(xt),'scope':'actual model-fit windows and labels only; lag-wise feature statistics'}
    def transform(a):return ((np.where(np.isfinite(a),a,mean)-mean)/scale).astype(np.float32)
    return transform(xs),((ys-ym)/yscale).astype(np.float32),transform(xt),((yt-ym)/yscale).astype(np.float32),transform,state

class Study:
    def __init__(self,out,cfg):
        self.out=out;self.cfg=cfg;self.data={};self.capacity={};self.jobs=[]
        self.cache=out/'data';self.cache.mkdir(exist_ok=True)
        for site in range(1,7):
            file=next((ROOT/'data/wind_farm_data/data_original/wind_farms').glob(f'Wind farm site {site} (*.xlsx'))
            cap=float(re.search(r'(\d+)MW',file.name).group(1));self.capacity[site]=cap
            with contextlib.redirect_stdout(io.StringIO()):d=load_temporal_farm(file,TemporalProtocol(**cfg['temporal']))
            self.data[site]=d
            save(self.cache/f'farm{site}_audit.json',{'provenance':d.provenance,'statistics':d.statistics,'preprocessing':d.preprocessing,'capacity':cap,'feature_names':d.feature_names})
            np.savez_compressed(self.cache/f'farm{site}_ids.npz',times=d.timestamps,split=d.split,raw_target_row=d.target_row_indices,input_rows=d.input_row_indices)
            print(json.dumps({'loaded_site':site,'windows':len(d.y_raw),'counts':{s:{dom:int(d.mask(s,dom).sum()) for dom in ['source','target']} for s in ['train','validation','calibration','test']}}),flush=True)

    def run(self,name,site,source_parts,target_ids,test_ids,cal_ids,*,task='point',variant='Full',scheme='domain_balanced',seed=0,m=None,extra=None):
        out=self.out/name
        if (out/'result.json').exists():
            j=json.loads((out/'result.json').read_text());self.jobs.append(j);return j
        if out.exists():
            raise RuntimeError(f'Incomplete branch must be inspected, not silently overwritten: {out}')
        out.mkdir();d=self.data[site];cap=self.capacity[site]
        reference_n=sum(len(ids) for _,ids in source_parts)+len(target_ids)
        if variant=='TargetOnly':source_parts=[]
        xs=np.concatenate([self.data[s].X[ids] for s,ids in source_parts]) if source_parts else np.empty((0,d.X.shape[1]),np.float32)
        ys=np.concatenate([self.data[s].y_raw[ids]/self.capacity[s] for s,ids in source_parts]) if source_parts else np.empty(0)
        xt=d.X[target_ids];yt=d.y_raw[target_ids]/cap
        zs,ysc,zt,ytc,transform,prep=preprocess(xs,ys,xt,yt)
        save(out/'preprocessing.json',prep)
        source_site=np.concatenate([np.full(len(ids),s) for s,ids in source_parts]) if source_parts else np.array([],int)
        source_rows=np.concatenate([self.data[s].target_row_indices[ids] for s,ids in source_parts]) if source_parts else np.array([],int)
        np.savez_compressed(out/'fit_ids.npz',source_site=source_site,source_raw_row=source_rows,target_raw_row=d.target_row_indices[target_ids],target_window_index=target_ids,
                            test_window_index=test_ids,calibration_window_index=cal_ids)
        cfg=self.cfg['model'];cs,ct=loss_weights(len(zs),len(zt),cfg['total_loss_weight'],scheme,task)
        reference_updates=cfg['epochs']*math.ceil(reference_n/cfg['batch_size'])
        batches_per_epoch=math.ceil((len(zs)+len(zt))/cfg['batch_size'])
        epochs=math.ceil(reference_updates/batches_per_epoch) if variant=='TargetOnly' else cfg['epochs']
        taus=[self.cfg['point_tau']] if task=='point' else self.cfg['interval_taus']
        tx=transform(d.X[test_ids]);cx=transform(d.X[cal_ids]) if task=='interval' else None
        predictions=[];cal_predictions=[];fits=[]
        with (out/'training.log').open('w') as log:
            for tau in taus:
                opts={'lambda1':0.0 if variant=='NoVariance' else cfg['lambda1'],
                      'lambda2':0.0 if variant in ('NoMMD','TargetOnly') else cfg['lambda2'],
                      'C_S':cs,'C_T':ct,'tau':tau,'kernel_gamma':cfg['gamma'],
                      'n_components':m or cfg['landmarks'],'epochs':epochs,'learning_rate':cfg['learning_rate'],
                      'batch_size':cfg['batch_size'],'variance_mode':'centered_residual','asym_scale':0.0}
                np.random.seed(seed);solver=Fast_TL_QLDMR(**opts)
                start=time.perf_counter()
                with contextlib.redirect_stdout(log):solver.fit(zs,ysc,zt,ytc)
                fit_seconds=time.perf_counter()-start
                def predict(x):return ((solver.feature_map.transform(x)@solver.w+solver.b)*prep['label_scale']+prep['label_mean'])*cap
                predictions.append(predict(tx))
                if task=='interval':cal_predictions.append(predict(cx))
                np.savez_compressed(out/f'model_tau_{tau}.npz',w=solver.w,b=solver.b,landmarks=solver.feature_map.components_,normalization=solver.feature_map.normalization_)
                # Empirical fitting objective, not a certificate of optimization accuracy.
                z=np.r_[solver.feature_map.transform(zs),solver.feature_map.transform(zt)] if len(zs) else solver.feature_map.transform(zt)
                yy=np.r_[ysc,ytc];cc=np.r_[np.full(len(zs),cs),np.full(len(zt),ct)]
                residual=yy-(z@solver.w+solver.b)
                obj=float(.5*solver.w@solver.regularizer_matrix_@solver.w-solver.regularizer_linear_@solver.w+np.sum(cc*np.maximum(tau*residual,(tau-1)*residual)))
                fits.append({'options':opts,'fit_seconds':fit_seconds,'effective_landmarks':len(solver.w),'adam_updates':epochs*batches_per_epoch,'reference_full_updates':reference_updates,'objective_without_label_only_constant':obj})
        y=d.y_raw[test_ids];table={'window_index':test_ids,'raw_target_row':d.target_row_indices[test_ids],'time':d.timestamps[test_ids].astype(str),'y_true':y}
        calibration=None
        if task=='point':
            table['prediction']=predictions[0];metrics=point_metrics(y,predictions[0])
        else:
            lo=np.minimum(*predictions);hi=np.maximum(*predictions);clo=np.minimum(*cal_predictions);chi=np.maximum(*cal_predictions)
            cy=d.y_raw[cal_ids];score=np.maximum.reduce([clo-cy,cy-chi,np.zeros(len(cy))])
            rank=math.ceil((len(cy)+1)*(1-self.cfg['alpha']))
            if rank>len(cy):raise ValueError('Insufficient calibration labels for finite conformal rank.')
            q=float(np.sort(score)[rank-1]);table.update(raw_low=predictions[0],raw_high=predictions[1],ordered_lower=lo,ordered_upper=hi,lower=lo-q,upper=hi+q)
            metrics=interval_metrics(y,lo-q,hi+q,self.cfg['alpha']);calibration={'n':len(cy),'rank':rank,'q_hat':q,'alpha':self.cfg['alpha']}
            pd.DataFrame({'window_index':cal_ids,'raw_target_row':d.target_row_indices[cal_ids],'time':d.timestamps[cal_ids].astype(str),'y_true':cy,'raw_low':cal_predictions[0],'raw_high':cal_predictions[1],'score':score}).to_csv(out/'calibration.csv',index=False)
        pd.DataFrame(table).to_csv(out/'predictions.csv',index=False)
        result={'name':name,'site':site,'task':task,'variant':variant,'scheme':scheme,'seed':seed,'source_n':len(zs),'target_n':len(zt),'fits':fits,'metrics':metrics,'calibration':calibration,'extra':extra or {}}
        save(out/'result.json',result);self.jobs.append(result)
        pd.DataFrame([{'name':j['name'],'site':j['site'],'task':j['task'],'variant':j['variant'],'scheme':j['scheme'],'seed':j['seed'],'source_n':j['source_n'],'target_n':j['target_n'],**j['metrics'],'fit_seconds':sum(f['fit_seconds'] for f in j['fits'])} for j in self.jobs]).to_csv(self.out/'summary.csv',index=False)
        print(json.dumps({'done':name,**metrics}),flush=True);return result

    def execute(self):
        site=6;d=self.data[site];src=np.flatnonzero(d.mask('train','source'));target=np.flatnonzero(d.mask('train','target'))
        test=np.flatnonzero(d.mask('test','target'));cal=np.flatnonzero(d.mask('calibration','target'))
        # Complete component runs; all seeds retained.
        for seed in self.cfg['seeds']:
            for variant in self.cfg['components']:
                for task in ['point','interval']:
                    self.run(f'components_{variant}_{task}_s{seed}',site,[(site,src)],target,test,cal,task=task,variant=variant,seed=seed)
        # Nested target-label budgets; no target labels were used for parameter selection.
        for seed in self.cfg['seeds']:
            order=np.random.default_rng(seed).permutation(target)
            for budget in self.cfg['label_budgets']:
                if budget=='all':continue # Exact reference is the full component point run.
                ids=order[:int(budget)]
                self.run(f'scarcity_n{budget}_s{seed}',site,[(site,src)],ids,test,cal,seed=seed,extra={'requested_budget':budget,'domain_total_weight_fixed':True})
        for scheme in self.cfg['weight_schemes']:
            if scheme=='domain_balanced':continue
            for task in ['point','interval']:
                self.run(f'weights_{scheme}_{task}',site,[(site,src)],target,test,cal,task=task,scheme=scheme,extra={'total_loss_weight_fixed':True})
        threshold_rows=[]
        common_core,_,_=event_definition(d,99);core_test=(d.split=='test')&common_core
        for q in self.cfg['threshold_quantiles']:
            domain,masks,thresholds=event_definition(d,q)
            ns=np.flatnonzero((d.split=='train')&~domain);nt=np.flatnonzero((d.split=='train')&domain)
            tt=np.flatnonzero((d.split=='test')&domain);cc=np.flatnonzero((d.split=='calibration')&domain)
            assert np.all(np.isin(np.flatnonzero(core_test),tt))
            for cutoff in self.cfg['cutout_sensitivity_ms']:
                alternative=masks['wind']|masks['cold']|masks['hot']|(d.target_weather['wind_hub']>cutoff)
                same=bool(np.array_equal(alternative,domain))
                alt_source=np.flatnonzero((d.split=='train')&~alternative);alt_target=np.flatnonzero((d.split=='train')&alternative)
                alt_test=np.flatnonzero((d.split=='test')&alternative);alt_cal=np.flatnonzero((d.split=='calibration')&alternative)
                threshold_rows.append({'q':q,'cutout_ms':cutoff,'thresholds':thresholds,'identical_domain_mask_to_25':bool(np.array_equal(alternative,domain)),
                                       'n_train_source':len(alt_source),'n_train_target':len(alt_target),'n_cal':len(alt_cal),'n_test':len(alt_test)})
                if not same:
                    for task in ['point','interval']:
                        self.run(f'threshold_q{q}_cutout{cutoff}_{task}',site,[(site,alt_source)],alt_target,alt_test,alt_cal,task=task,extra={'q':q,'cutout_ms':cutoff,'thresholds':thresholds})
            for task in ['point','interval']:
                name=f'threshold_q{q}_{task}'
                if q==98:
                    self.alias(name,f'components_Full_{task}_s0')
                else:self.run(name,site,[(site,ns)],nt,tt,cc,task=task,extra={'q':q,'thresholds':thresholds,'common_core_q':99})
        save(self.out/'threshold_definitions.json',threshold_rows)
        # Accuracy/runtime curves use the same test set and full training set.
        for seed in self.cfg['seeds']:
            for m in self.cfg['landmark_grid']:
                if m==self.cfg['model']['landmarks']:
                    self.alias(f'landmarks_m{m}_s{seed}',f'components_Full_point_s{seed}')
                else:self.run(f'landmarks_m{m}_s{seed}',site,[(site,src)],target,test,cal,seed=seed,m=m,extra={'landmarks_requested':m})
        for fraction in self.cfg['source_fractions']:
            if fraction==1:
                self.alias('scale_fraction1','components_Full_point_s0');continue
            ns=temporal_uniform(src,max(1,int(len(src)*fraction)))
            self.run(f'scale_fraction{fraction}',site,[(site,ns)],target,test,cal,extra={'source_fraction':fraction,'selection':'deterministic evenly spaced by training time'})
        # Source farms exclude the target farm, with an explicitly bounded source budget.
        for site in self.cfg['cross_farm_sites']:
            d=self.data[site];nt=np.flatnonzero(d.mask('train','target'))
            nt=np.random.default_rng(0).permutation(nt)[:self.cfg['cross_farm_target_labels']]
            tt=np.flatnonzero(d.mask('test','target'));cc=np.flatnonzero(d.mask('calibration','target'))
            sources=[(s,temporal_uniform(np.flatnonzero(self.data[s].mask('train','source')),self.cfg['cross_farm_source_cap_per_site'])) for s in self.data if s!=site]
            for variant in ['Full','TargetOnly']:
                for task in ['point','interval']:
                    self.run(f'crossfarm_site{site}_{variant}_{task}',site,sources,nt,tt,cc,task=task,variant=variant,extra={'source_sites':[s for s,_ in sources],'target_labels':len(nt),'supervised_target_adaptation':True})

    def alias(self,name,reference):
        out=self.out/name;out.mkdir(exist_ok=True);save(out/'alias.json',{'reference':reference,'reason':'identical predeclared configuration; no duplicate fitting or selective result choice'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=ROOT/'review/replied_study_protocol.json');p.add_argument('--output',type=Path,default=ROOT/'review/replied_comment_runs/chronological_v1');p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=json.loads(a.protocol.read_text());sources=[Path(__file__),ROOT/'review/temporal_data.py',ROOT/'Train_TL_QLDMR.py',ROOT/'Kernel_Function.py']
    hashes={str(x.relative_to(ROOT)):digest(x) for x in sources}
    datahash={str(x.relative_to(ROOT)):digest(x) for x in (ROOT/'data/wind_farm_data/data_original/wind_farms').glob('*.xlsx')}
    if a.output.exists():
        if not a.resume:raise FileExistsError('Existing output refused; inspect before explicit identical-protocol resume.')
        manifest=json.loads((a.output/'manifest.json').read_text())
        assert manifest['source_hashes']==hashes and manifest['data_hashes']==datahash and manifest['protocol']==cfg
    else:
        a.output.mkdir(parents=True);manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol':cfg,'source_hashes':hashes,'data_hashes':datahash,'python':platform.python_version(),'numpy':np.__version__}
        save(a.output/'manifest.json',manifest)
    try:
        with threadpool_limits(limits=cfg['threads']):
            study=Study(a.output,cfg);study.execute()
        manifest['status']='complete';manifest['completed_fits']=len(study.jobs)
    except Exception as e:
        manifest['status']='failed';manifest['error']=repr(e);raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save(a.output/'manifest.json',manifest)

if __name__=='__main__':main()
