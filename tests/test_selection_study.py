"""Synthetic isolation checks; no real held-out labels or model selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import review.run_selection_study as study


def test_preprocessing_uses_fitting_values_and_preserves_missing_policy():
    x = np.array([[1., np.nan, np.nan], [3., 2., np.nan], [5., 6., np.nan]])
    y = np.array([.1, .3, .8])
    xx, yy, transform, state = study.preprocess(x, y)
    np.testing.assert_allclose(state['mean'], [3., 4., 0.])
    np.testing.assert_allclose(xx.mean(0), 0., atol=1e-15)
    np.testing.assert_allclose(yy, (y-y.mean())/y.std())
    previous = json.dumps(state, sort_keys=True)
    transformed = transform(np.array([[1e12, np.nan, np.nan]]))
    assert transformed[0, 1] == 0 and transformed[0, 2] == 0
    assert json.dumps(state, sort_keys=True) == previous


def fake_data(site):
    split = np.array(['train']*6 + ['validation']*2 + ['calibration']*2 + ['test']*2)
    extreme = np.array([False, False, False, True, True, True] + [True]*6)
    x = np.arange(36., dtype=float).reshape(12, 3) + site
    y = np.arange(1., 13.) * study.CAPACITY[site] / 10
    y[6:] = 1e8  # held-out outcomes must never influence fitting statistics
    times = np.datetime64('2019-01-01') + np.arange(12)*np.timedelta64(15, 'm')
    def mask(period, domain):
        return (split == period) & (extreme if domain == 'target' else ~extreme)
    return SimpleNamespace(X=x, y_raw=y, split=split, timestamps=times, is_extreme=extreme,
                           mask=mask, event_masks={'wind':extreme, 'cold':np.zeros(12,bool), 'hot':np.zeros(12,bool)})


def test_context_keeps_heldout_data_out_and_normalizes_each_source_capacity(tmp_path, monkeypatch):
    data = {1:fake_data(1), 2:fake_data(2)}
    instance = study.Study.__new__(study.Study)
    instance.out, instance.cfg, instance.data = tmp_path, study.protocol(), data
    instance.context_cache, instance.feature_cache = {}, {}
    instance.cfg['source_cap_per_other_site'] = 2
    instance.cfg['cross_target_fit_budget'] = 2
    monkeypatch.setattr(study, 'stratified_target_indices',
                        lambda d, budget: (np.array([3, 5]), {'n':2}))
    ctx = instance.context({'name':'cross_site1','site':1,'mode':'cross'}, 'Full')
    # Time-even source IDs 0,2 from site2; target IDs 3,5 from site1.
    expected_x = np.r_[data[2].X[[0,2]],data[1].X[[3,5]]]
    expected_y = np.array([.1,.3,.4,.6])
    np.testing.assert_allclose(ctx['state']['mean'], expected_x.mean(0))
    np.testing.assert_allclose(ctx['state']['label_mean'], expected_y.mean())
    np.testing.assert_allclose(ctx['y'], (expected_y-expected_y.mean())/expected_y.std())
    ids = np.load(ctx['folder']/'ids.npz')
    np.testing.assert_array_equal(ids['source_site'], [2,2])
    np.testing.assert_array_equal(ids['target_window'], [3,5])
    target = instance.context({'name':'cross_site1','site':1,'mode':'cross'}, 'TargetSVQR')
    assert target['source_n'] == 0
    np.testing.assert_allclose(target['state']['label_mean'], .5)
    np.testing.assert_array_equal(np.load(target['folder']/'ids.npz')['target_window'], [3,5])


@pytest.mark.parametrize('method', ['Full','NoVariance','NoDiscrepancy','WeightedSVQR','TargetSVQR'])
def test_method_regularizers_linear_term_and_total_loss_weights(tmp_path, monkeypatch, method):
    rng = np.random.default_rng(72)
    z, y = rng.normal(size=(7,3)), rng.normal(size=7)
    ns = 0 if method == 'TargetSVQR' else 4
    candidate = {'gamma':.01,'total_loss':19.,'target_fraction':.7,'lambda1':2.3,'lambda2':.8}
    instance = study.Study.__new__(study.Study)
    instance.out, instance.cfg = tmp_path, study.protocol()
    data_folder = tmp_path/'data'
    ctx = {'folder':data_folder,'source_n':ns,'y':y,'counts':{},'x':z}
    instance.context = lambda task, model:ctx
    instance.candidate = lambda task, model, index:candidate
    instance.features = lambda context, seed, gamma:{'z':z,'path':data_folder/'map.npz'}
    observed = {}
    def fake_solver(Z,Y,R,c,C,tau,**kwargs):
        observed.update(R=R.copy(),c=c.copy(),C=C.copy(),y=Y.copy(),tau=tau)
        return SimpleNamespace(w=np.zeros(3),b=0.,dual_variable=np.zeros(7),report=lambda:{'converged':False})
    monkeypatch.setattr(study,'refine_pinball',fake_solver)
    folder = tmp_path/method
    instance.fit({'name':'fake','site':1},method,0,0,.025,folder)
    lam1 = 2.3 if method in ('Full','NoDiscrepancy') else 0.
    lam2 = .8 if method in ('Full','NoVariance') else 0.
    expected_R, expected_c = np.eye(3), np.zeros(3)
    if ns:
        H = np.eye(ns)-np.ones((ns,ns))/ns
        delta = z[:ns].mean(0)-z[ns:].mean(0)
        expected_R += lam1/ns*z[:ns].T@H@z[:ns] + lam2*np.outer(delta,delta)
        expected_c = lam1/ns*z[:ns].T@H@y[:ns]
        np.testing.assert_allclose(observed['C'][:ns].sum(), 19*.3)
        np.testing.assert_allclose(observed['C'][ns:].sum(), 19*.7)
    np.testing.assert_allclose(observed['R'],expected_R,atol=1e-13)
    np.testing.assert_allclose(observed['c'],expected_c,atol=1e-13)
    np.testing.assert_allclose(observed['C'].sum(),19.)
    assert observed['tau'] == .025
    stored = np.load(folder/'model.npz')
    np.testing.assert_array_equal(stored['C'],observed['C'])


def test_candidate_selection_uses_only_validation_keeps_all_and_locks_afterwards(tmp_path):
    instance = study.Study.__new__(study.Study)
    cfg = study.protocol()
    cfg['tasks'] = [{'name':'fake1','site':1},{'name':'fake2','site':2}]
    cfg['methods'] = ['Full','WeightedSVQR']
    cfg['candidate_count_per_method'] = 3
    instance.out, instance.cfg = tmp_path, cfg
    instance.context_cache,instance.feature_cache = {},{}
    study.save(tmp_path/'protocol.json',cfg)
    ctx = {'ids':{'validation':np.array([0,1])},'data':SimpleNamespace(y_raw=np.array([1.,2.]))}
    instance.context = lambda task,method:ctx
    visited = []
    def fake_fit(task,method,index,seed,tau,folder):
        folder.mkdir(parents=True)
        # The best candidate is deliberately marked unconverged: the protocol
        # retains such candidates and does not filter on convergence/results.
        return {'diagnostics':{'converged':index==0}}
    def fake_predict(context,folder,split):
        assert not (tmp_path/'selection_lock.json').exists()
        assert split == 'validation'
        visited.append(str(folder))
        index = int(folder.name[1:])
        return np.array([1.,2.]) + [2.,.5,.5][index]
    instance.fit,instance.predict = fake_fit,fake_predict
    selected = instance.tune()
    assert len(visited) == 12
    assert all(row['selected_index']==1 for row in selected)  # deterministic tie
    assert all(len(row['candidates'])==3 for row in selected)
    lock = json.loads((tmp_path/'selection_lock.json').read_text())
    assert lock['selections'] == selected and len(lock['selection_files']) == 4
    assert len(list(tmp_path.glob('*/*/candidates/*/validation.csv'))) == 12


def test_rolling_calibration_uses_only_strictly_earlier_outcomes():
    calibration = np.arange(30.,dtype=float)
    observed = np.array([1000.,1.,2.,3.,4.,5.,6.,7.])
    q = study.rolling_expansions(calibration,observed,window=25)
    assert q[0] == 29. and q[1] == 1000.
    changed = observed.copy();changed[5] = 1e9
    altered = study.rolling_expansions(calibration,changed,window=25)
    np.testing.assert_array_equal(q[:6],altered[:6])
    assert altered[6] == 1e9
    assert study.conformal_quantile(np.arange(19.))[1] == 19
    with pytest.raises(ValueError,match='Insufficient'):
        study.conformal_quantile(np.arange(18.))


def test_prediction_units_and_interval_metrics(tmp_path):
    instance=study.Study.__new__(study.Study);instance.out=tmp_path
    folder=tmp_path/'fit';folder.mkdir()
    np.savez(tmp_path/'map.npz',centers=np.zeros((1,2)),normalization=np.ones((1,1)),gamma=.5)
    np.savez(folder/'model.npz',w=np.array([2.]),b=.3)
    study.save(folder/'fit.json',{'method':'Full','feature_map':'map.npz'})
    x=np.array([[0.,0.],[1.,1.]])
    ctx={'ids':{'test':np.array([0,1])},'transform':lambda x:x,
         'data':SimpleNamespace(X=x),'state':{'label_scale':.2,'label_mean':.4},'site':6}
    prediction=instance.predict(ctx,folder,'test')
    np.testing.assert_allclose(prediction,((2*np.array([1.,np.exp(-1)])+.3)*.2+.4)*96)
    metrics=study.interval_metrics(np.array([-1.,1.,4.]),np.array([0.,0.,0.]),np.array([2.,2.,2.]),alpha=.1)
    assert metrics['picp'] == 1/3 and metrics['mpiw'] == 2.
    assert metrics['winkler'] == 22.


def test_summary_refuses_partial_study_before_writing_tables(tmp_path):
    from review.summarize_selection_study import build
    study.save(tmp_path/'manifest.json',{'status':'running'})
    study.save(tmp_path/'protocol.json',{})
    with pytest.raises(AssertionError,match='incomplete'):
        build(tmp_path)
    assert not list(tmp_path.glob('*.tex'))
    assert not (tmp_path/'summary.json').exists()


def test_independent_verifier_calibration_matches_finite_rank_and_past_only():
    from review.verify_selection_study import calibration_rank, causal_rolling_quantiles
    calibration=np.arange(25.,dtype=float)
    observed=np.array([900.,2.,3.,4.])
    assert calibration_rank(calibration,.05)==(24.,25)
    expansions=causal_rolling_quantiles(calibration,observed,25,.05)
    np.testing.assert_array_equal(expansions,[24.,900.,900.,900.])
