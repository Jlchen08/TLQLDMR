"""Reproduce one preselected saved-model training-only refinement diagnostic.

No calibration/test labels or predictions enter model assessment. Historical
artifacts are read only; output goes to review/solver_refinement_diagnostics.
"""
import contextlib, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.temporal_data import TemporalProtocol, load_temporal_farm
from review.convex_refinement import refine_pinball, primal_objective
from Kernel_Function import rbf_kernel

run = ROOT/'review/replied_comment_runs/chronological_v1'
branch = run/'components_Full_point_s0'
output = ROOT/'review/solver_refinement_diagnostics'
output.mkdir(exist_ok=True)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads((run/'manifest.json').read_text())
audit=json.loads((run/'data/farm6_audit.json').read_text())
result=json.loads((branch/'result.json').read_text())
prep=json.loads((branch/'preprocessing.json').read_text())
opts=result['fits'][0]['options']
assert digest(audit['provenance']['data_file']) == audit['provenance']['data_sha256']
with threadpool_limits(1):
    with contextlib.redirect_stdout(io.StringIO()):
        data=load_temporal_farm(audit['provenance']['data_file'],TemporalProtocol(**manifest['protocol']['temporal']))
    ids=np.load(branch/'fit_ids.npz',allow_pickle=False)
    assert np.all(ids['source_site']==6)
    row_to_window={int(row):idx for idx,row in enumerate(data.target_row_indices)}
    source_ids=np.array([row_to_window[int(row)] for row in ids['source_raw_row']])
    target_ids=ids['target_window_index']
    assert np.array_equal(data.target_row_indices[target_ids], ids['target_raw_row'])
    selected=np.r_[source_ids,target_ids]
    assert np.all(data.split[selected]=='train')
    mean,scale=np.array(prep['mean']),np.array(prep['scale'])
    raw_x=data.X[selected]
    X=((np.where(np.isfinite(raw_x),raw_x,mean)-mean)/scale).astype(np.float32)
    y=((data.y_raw[selected]/audit['capacity']-prep['label_mean'])/prep['label_scale']).astype(np.float32)
    model=np.load(branch/'model_tau_0.5.npz',allow_pickle=False)
    mapping_start=time.perf_counter()
    Z=rbf_kernel(X,model['landmarks'],gamma=opts['kernel_gamma'])@model['normalization']
    ns=len(source_ids)
    source_Z=Z[:ns]-Z[:ns].mean(0)
    source_y=y[:ns]-y[:ns].mean()
    delta=Z[:ns].mean(0)-Z[ns:].mean(0)
    R=np.eye(Z.shape[1])+opts['lambda1']*(source_Z.T@source_Z)/ns+opts['lambda2']*np.outer(delta,delta)
    c=(opts['lambda1']/ns)*(source_Z.T@source_y)
    C=np.r_[np.full(ns,opts['C_S']),np.full(len(target_ids),opts['C_T'])]
    elapsed_mapping=time.perf_counter()-mapping_start
    original=primal_objective(Z,y,R,c,C,.5,model['w'],float(model['b']))
    recorded=result['fits'][0]['objective_without_label_only_constant']
    assert np.isclose(original,recorded,rtol=1e-9,atol=1e-8),(original,recorded)
    def progress(row):
        if row['iteration']%250==0:
            print(json.dumps(row),flush=True)
    fitted=refine_pinball(Z,y,R,c,C,.5,w0=model['w'],b0=float(model['b']),
                         gap_rtol=1e-3,gap_atol=1e-7,max_iter=3000,callback=progress)
    report={
        'branch':str(branch),'scope':'One preselected saved Adam fit, training objective only; no test/calibration predictions evaluated.',
        'source_n':ns,'target_n':len(target_ids),'features':Z.shape[1],'tau':.5,
        'recorded_adam_objective':recorded,'recomputed_adam_objective':original,
        'feature_and_regularizer_seconds':elapsed_mapping,
        'fixed_feature_objective_decrease':original-fitted.primal,
        'minimum_adam_suboptimality_certified_by_improved_feasible_primal':original-fitted.primal,
        'maximum_adam_suboptimality_from_dual_bound':original-fitted.dual_bound,
        'refinement':fitted.report(),
        'hashes':{str(p.relative_to(ROOT)):digest(p) for p in [branch/'model_tau_0.5.npz',branch/'result.json',branch/'fit_ids.npz',branch/'preprocessing.json',ROOT/'review/convex_refinement.py']},
        'problem_array_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in {'Z':Z,'y':y,'R':R,'c':c,'C':C}.items()},
        'interpretation':'Training suboptimality is bounded for these fixed features and this objective. Generalization and exact-kernel approximation are not assessed.',
    }
    (output/'adam_seed0_point_training.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'adam_seed0_point_refined.npz',w=fitted.w,b=fitted.b,dual=fitted.dual_variable)
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ['refinement','hashes','problem_array_sha256']}),flush=True)
    print('REFINEMENT',json.dumps({k:v for k,v in fitted.report().items() if k!='history'}),flush=True)
