"""Stage 4B provenance, likelihood, held-out, Windows-worker and postprocessor tests."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from avant_model.inversion.coupled_dream import sha256_file
from avant_model.inversion.coupled_likelihood import gaussian_log_likelihood
from avant_model.inversion.coupled_observations import StrainObservations, station_masks
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_dream import (
    WorkerJointLikelihood, load_stage4a, validate_joint_mle)
from avant_model.inversion.joint_pressure_strain import (
    JointPressureStrainLikelihood, PressureObservations)
from avant_model.model.pressure_gauges import PressureGaugeOperator


class FakeFV:
    x_m = np.array([-205.,0.,205.])
    y_m = np.array([-380.,0.,380.])

    def __init__(self):
        self.calls = 0

    def solve(self,K,Ss,times):
        self.calls += 1
        X,Y=np.meshgrid(self.x_m,self.y_m)
        grid=(1.0+.15*X/205.+.12*Y/380.)
        field=np.stack([1.e5*(1+.08*t)*(K/1e-6+.6*Ss/1e-6*t/(1+t))*grid for t in times])
        return SimpleNamespace(times_h=np.asarray(times,float),
                               pressure_change_pa=field, x_m=self.x_m,y_m=self.y_m)


class FakeModel:
    def __init__(self,channels):
        self.channels=tuple(channels)
        self.hydraulic=FakeFV()
        self.P=np.eye(9)
        rng=np.random.default_rng(33)
        self.H=np.abs(rng.normal(size=(len(channels),9)))*1.5e-6

    def predict(self,K,Ss,times):
        fv=self.hydraulic.solve(K,Ss,times)
        strain=fv.pressure_change_pa.reshape(len(times),-1) @ self.P.T @ self.H.T
        return SimpleNamespace(strain_ns=strain)


@pytest.fixture
def staged(tmp_path):
    root=tmp_path
    source=root/'stage1';stage4a=root/'stage4a';mapdir=root/'stage2'
    for p in (source,stage4a,mapdir):p.mkdir()
    hc=root/'hydraulic.json'; mc=root/'mechanical.json';gauges=root/'gauges.csv'
    station=root/'stations.csv'
    station.write_text('station,x,y,z\nS01,0,0,30\n',encoding='utf-8')
    hc.write_text('{"injection_hours": 19.0}',encoding='utf-8')
    mc.write_text(json.dumps({'station_file':str(station)}),encoding='utf-8')
    gauges.write_text('gauge,x_local_m,y_local_m,role\nP01,25,0,train\nP02,-60,80,train\nP03,120,-150,holdout\n',encoding='utf-8')
    channels=tuple(f'{comp}_S{i:02d}' for comp in ('eXX','eYY','eXY','eZZ') for i in range(1,9))
    mask,held=station_masks(channels,('S07','S08'))
    times=np.arange(17,dtype=float)*.5
    model=FakeModel(channels)
    eps=model.predict(1e-6,1e-6,times).strain_ns
    observed_eps=eps+np.random.default_rng(42).normal(0,2,eps.shape)
    np.savez_compressed(source/'synthetic_observations.npz',times_h=times,channels=np.asarray(channels),
                        observed_ns=observed_eps,synthetic_truth_ns=eps,train_mask=mask,holdout_mask=held)
    stage1_report={'scope':'Same-model synthetic mock',
                   'hydraulic_config_sha256':sha256_file(hc),
                   'mechanical_config_sha256':sha256_file(mc),
                   'noise_std_ns':2.,'fine_n':18,
                   'truth_log10_K_log10_Ss':[-6.,-6.]}
    (source/'summary.json').write_text(json.dumps(stage1_report),encoding='utf-8')
    xp=np.array([25.,-60.,120.]);yp=np.array([0.,80.,-150.])
    op=PressureGaugeOperator(model.hydraulic.x_m,model.hydraulic.y_m,xp,yp)
    pressure_true=op.sample(model.hydraulic.solve(1e-6,1e-6,times).pressure_change_pa)
    pressure_obs=pressure_true+np.random.default_rng(314159).normal(0,20000,pressure_true.shape)
    pt=np.array([True,True,False]);ph=~pt
    np.savez_compressed(stage4a/'synthetic_pressure_observations.npz',
                        times_h=times,gauge_names=np.array(['P01','P02','P03']),
                        gauge_x_m=xp,gauge_y_m=yp,observed_pa=pressure_obs,
                        truth_pa=pressure_true,train_mask=pt,holdout_mask=ph,sigma_pa=20000.)
    strain=StrainObservations(times,channels,observed_eps,2.,mask,held)
    pressure=PressureObservations(times,('P01','P02','P03'),xp,yp,pressure_obs,20000.,pt,ph)
    joint=JointPressureStrainLikelihood(model,strain,pressure,HydraulicBounds())
    ls,lp=joint.log_likelihood_blocks([-6.,-6.])
    summary={'scope':'Stage 4A same-model synthetic joint FV mock',
             'source_dir':str(source),'stage2_map_dir':str(mapdir),
             'strain_observations_sha256':sha256_file(source/'synthetic_observations.npz'),
             'hydraulic_config_sha256':sha256_file(hc),'mechanical_config_sha256':sha256_file(mc),
             'gauge_csv_sha256':sha256_file(gauges),'pressure_sigma_pa':20000.,'strain_sigma_ns':2.,
             'pressure_gauges':[
                 {'gauge':'P01','role':'train','x_local_m':25.,'y_local_m':0.},
                 {'gauge':'P02','role':'train','x_local_m':-60.,'y_local_m':80.},
                 {'gauge':'P03','role':'holdout','x_local_m':120.,'y_local_m':-150.}],
             'n_training_strain':408,'n_heldout_strain':136,
             'n_training_pressure':34,'n_heldout_pressure':17,
             'fit_results':[{'method':'joint_MLE','log10_K':-6.,'log10_Ss':-6.,
                             'log_likelihood_joint_training':ls+lp,
                             'log_likelihood_strain_training':ls,
                             'log_likelihood_pressure_training':lp}]}
    (stage4a/'joint_summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (mapdir/'summary.json').write_text('{"scope":"mock Stage2"}',encoding='utf-8')
    return SimpleNamespace(root=root,source=source,stage4a=stage4a,mapdir=mapdir,
                           hc=hc,mc=mc,gauges=gauges,station=station,
                           strain=strain,pressure=pressure,model=model,summary=summary)


def test_saved_dataset_preserves_holdouts_and_order(staged):
    report,stage1,strain,pressure=load_stage4a(staged.stage4a,staged.hc,staged.mc,staged.gauges)
    assert int(strain.training_mask.sum())==24
    assert int(pressure.training_mask.sum())==2
    assert pressure.gauge_names==('P01','P02','P03')
    assert np.array_equal(strain.observed_ns,staged.strain.observed_ns)
    assert np.array_equal(pressure.observed_pa,staged.pressure.observed_pa)


def test_joint_mle_reproduces_full_normalized_likelihood(staged):
    ll=JointPressureStrainLikelihood(staged.model,staged.strain,staged.pressure)
    theta,value=validate_joint_mle(ll,staged.summary,HydraulicBounds())
    np.testing.assert_array_equal(theta,[-6.,-6.])
    assert value==pytest.approx(ll([-6.,-6.]))


def test_mle_mismatch_detected(staged):
    ll=JointPressureStrainLikelihood(staged.model,staged.strain,staged.pressure)
    report=json.loads(json.dumps(staged.summary))
    report['fit_results'][0]['log_likelihood_joint_training']+=10
    with pytest.raises(ValueError,match='does not reproduce'):
        validate_joint_mle(ll,report,HydraulicBounds())


def test_strain_dataset_tampering_rejected(staged):
    with open(staged.source/'synthetic_observations.npz','ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='saved Stage 1|Stage 1'):
        load_stage4a(staged.stage4a,staged.hc,staged.mc,staged.gauges)


def test_gauge_csv_tampering_rejected(staged):
    staged.gauges.write_text(staged.gauges.read_text().replace('25,0,train','28,0,train'))
    with pytest.raises(ValueError,match='Gauge coordinate CSV'):
        load_stage4a(staged.stage4a,staged.hc,staged.mc,staged.gauges)


def test_holdout_mask_tampering_rejected(staged):
    path=staged.stage4a/'synthetic_pressure_observations.npz'
    with np.load(path,allow_pickle=False) as d:
        data={k:d[k] for k in d.files}
    data['train_mask']=np.array([True,False,True])
    data['holdout_mask']=~data['train_mask']
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='mask differs'):
        load_stage4a(staged.stage4a,staged.hc,staged.mc,staged.gauges)


def test_pressure_time_mismatch_rejected(staged):
    path=staged.stage4a/'synthetic_pressure_observations.npz'
    with np.load(path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
    data['times_h']=data['times_h']+0.25
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='timestamps differ'):
        load_stage4a(staged.stage4a,staged.hc,staged.mc,staged.gauges)


def test_worker_spawn_pickleable(staged):
    worker=WorkerJointLikelihood(str(staged.stage4a),str(staged.hc),str(staged.mc),
        str(staged.gauges),(-7.,-4.),(-7.,-4.))
    assert isinstance(pickle.loads(pickle.dumps(worker)),WorkerJointLikelihood)


def test_no_holdout_used_during_likelihood(staged):
    ll=JointPressureStrainLikelihood(staged.model,staged.strain,staged.pressure)
    altp=staged.pressure.observed_pa.copy()
    altp[:,2]+=1e8
    altered=PressureObservations(staged.pressure.times_h,staged.pressure.gauge_names,
         staged.pressure.gauge_x_m,staged.pressure.gauge_y_m,altp,20000.,
         staged.pressure.training_mask,staged.pressure.holdout_mask)
    altll=JointPressureStrainLikelihood(staged.model,staged.strain,altered)
    assert ll([-6.05,-5.95])==pytest.approx(altll([-6.05,-5.95]))


def test_joint_uses_one_fv_solve_per_candidate(staged):
    ll=JointPressureStrainLikelihood(staged.model,staged.strain,staged.pressure)
    pre=staged.model.hydraulic.calls
    value=ll([-6.,-6.])
    assert np.isfinite(value)
    assert staged.model.hydraulic.calls==pre+1


def test_predictive_bands_and_coverage(staged):
    from scripts.postprocessing.postprocess_joint_pressure_strain_dream import predictive_bands,heldout_assessment
    data=np.stack([staged.strain.observed_ns,staged.strain.observed_ns+1.],axis=0)
    a=predictive_bands(data,2.,np.random.default_rng(8),10)
    b=predictive_bands(data,2.,np.random.default_rng(8),10)
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    rmse,coverage=heldout_assessment(a,staged.strain.observed_ns,staged.strain.holdout_mask)
    assert 0<=coverage<=1 and rmse==pytest.approx(.5)


def test_summary_posterior_has_physical_units_and_log_width():
    from scripts.postprocessing.postprocess_joint_pressure_strain_dream import posterior_statistics
    rng=np.random.default_rng(5)
    samples=rng.normal([-6,-6],[.01,.02],size=(500,2))
    posterior=posterior_statistics(samples)
    assert 9e-7<posterior['log10_K_m_s']['physical_median']<1.1e-6
    assert posterior['log10_Ss_per_m']['width_95pct_log10']>.04


def test_stage4b_sampler_and_postprocess_full_interface_smoke(staged,monkeypatch):
    """Fake PyDREAM engine; real saved-data paths, provenance and charts tested."""
    import scripts.inversion.run_joint_pressure_strain_dream as runmod
    import scripts.postprocessing.postprocess_joint_pressure_strain_dream as postmod

    def fake_build(hc,mc,stage1,strain,pressure,bounds):
        model=FakeModel(strain.channels)
        return JointPressureStrainLikelihood(model,strain,pressure,bounds),{'injection_hours':19.}

    monkeypatch.setattr(runmod,'build_joint_likelihood',fake_build)
    monkeypatch.setattr(postmod,'build_joint_likelihood',fake_build)
    fake_pydream=ModuleType('pydream')
    fake_core=ModuleType('pydream.core')
    fake_params=ModuleType('pydream.parameters')
    fake_params.SampledParam=lambda *args,**kwargs:SimpleNamespace(args=args,kwargs=kwargs)
    def fake_dream(**kwargs):
        assert kwargs['nCR']==2 and kwargs['niterations']==120 and kwargs['nchains']==4
        assert kwargs['likelihood'].__class__.__name__=='WorkerJointLikelihood'
        rng=np.random.default_rng(10)
        chains=[rng.normal([-6,-6],[.01,.02],size=(120,2)) for _ in range(4)]
        logp=[np.full(120,-1230.) for _ in range(4)]
        return chains,logp
    fake_core.run_dream=fake_dream
    monkeypatch.setitem(sys.modules,'pydream',fake_pydream)
    monkeypatch.setitem(sys.modules,'pydream.core',fake_core)
    monkeypatch.setitem(sys.modules,'pydream.parameters',fake_params)
    output=staged.root/'stage4b'
    runmod.main(['--stage4a-dir',str(staged.stage4a),'--hydraulic-config',str(staged.hc),
        '--mechanical-config',str(staged.mc),'--gauges-csv',str(staged.gauges),
        '--output-dir',str(output),'--niterations','120','--burn-in','60','--nchains','4'])
    assert (output/'chains_log10.npy').exists()
    report=postmod.main(['--run-dir',str(output),'--posterior-draws','25','--noise-replicates','8'])
    assert (output/'joint_posterior_summary.json').exists()
    assert (output/'joint_heldout_pressure_predictive.png').exists()
    assert (output/'joint_heldout_strain_predictive.png').exists()
    assert (output/'joint_posterior_scatter.png').exists()
    assert report['retained_samples']==4*60
    assert report['heldout_pressure_rmse_pa']>=0
    assert report['heldout_strain_rmse_ns']>=0


def test_stage4b_postprocessing_rejects_changed_pressure_observations(staged,monkeypatch):
    import scripts.postprocessing.postprocess_joint_pressure_strain_dream as postmod
    run=staged.root/'stage4b_fake';run.mkdir()
    manifest={
        'stage4a_dir':str(staged.stage4a),'stage4a_report_sha256':sha256_file(staged.stage4a/'joint_summary.json'),
        'stage4a_pressure_observations_sha256':sha256_file(staged.stage4a/'synthetic_pressure_observations.npz'),
        'stage1_dir':str(staged.source),'stage1_summary_sha256':sha256_file(staged.source/'summary.json'),
        'stage1_strain_observations_sha256':sha256_file(staged.source/'synthetic_observations.npz'),
        'stage2_map_dir':str(staged.mapdir),'stage2_map_summary_sha256':sha256_file(staged.mapdir/'summary.json'),
        'hydraulic_config':str(staged.hc),'hydraulic_config_sha256':sha256_file(staged.hc),
        'mechanical_config':str(staged.mc),'mechanical_config_sha256':sha256_file(staged.mc),
        'gauges_csv':str(staged.gauges),'gauges_csv_sha256':sha256_file(staged.gauges),
        'station_file':str(staged.station),'station_file_sha256':sha256_file(staged.station),
        'sampling_completed':True}
    path=staged.stage4a/'synthetic_pressure_observations.npz'
    with open(path,'ab') as f:f.write(b'extra')
    with pytest.raises(ValueError,match='provenance changed'):
        postmod.verify_manifest(manifest)


def _build_stage3_comparison(staged, *, corrupt_prior=False, passed=True):
    stage3=staged.root/'stage3';stage3.mkdir()
    manifest={
        'stage1_observations_sha256':sha256_file(staged.source/'synthetic_observations.npz'),
        'hydraulic_config_sha256':sha256_file(staged.hc),
        'mechanical_config_sha256':sha256_file(staged.mc),
        'station_file_sha256':sha256_file(staged.station),
        'bounds_log10':[[-7,-4],[-7,-3] if corrupt_prior else [-7,-4]],
        'burn_in_per_chain':50}
    (stage3/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (stage3/'posterior_summary.json').write_text(json.dumps({'diagnostic_pass':passed}),encoding='utf-8')
    rng=np.random.default_rng(123)
    chains=rng.normal([-6.,-6.],[.02,.06],size=(4,250,2))
    np.save(stage3/'chains_log10.npy',chains)
    return stage3


def _minimal_joint_manifest(staged):
    return {
        'stage1_strain_observations_sha256':sha256_file(staged.source/'synthetic_observations.npz'),
        'hydraulic_config_sha256':sha256_file(staged.hc),
        'mechanical_config_sha256':sha256_file(staged.mc),
        'station_file_sha256':sha256_file(staged.station),
        'bounds_log10':[[-7.,-4.],[-7.,-4.]],
    }


def test_strain_only_comparison_quantifies_interval_width(staged):
    from scripts.postprocessing.postprocess_joint_pressure_strain_dream import (
        load_strain_only_comparison, posterior_statistics)
    stage3=_build_stage3_comparison(staged)
    posterior=posterior_statistics(np.random.default_rng(21).normal(
        [-6,-6],[.007,.018],size=(500,2)))
    report,old_samples=load_strain_only_comparison(stage3,_minimal_joint_manifest(staged),posterior)
    assert old_samples.shape==(4*200,2)
    for name in ('log10_K_m_s','log10_Ss_per_m'):
        assert 0<report['interval_comparison'][name]['joint_over_strain_only_95pct_width_ratio']<1.5


def test_strain_only_comparison_rejects_different_priors(staged):
    from scripts.postprocessing.postprocess_joint_pressure_strain_dream import (
        load_strain_only_comparison, posterior_statistics)
    stage3=_build_stage3_comparison(staged,corrupt_prior=True)
    posterior=posterior_statistics(np.random.default_rng(2).normal([-6,-6],[.02,.03],size=(500,2)))
    with pytest.raises(ValueError,match='prior bounds'):
        load_strain_only_comparison(stage3,_minimal_joint_manifest(staged),posterior)


def test_strain_only_comparison_rejects_nonconverged_run(staged):
    from scripts.postprocessing.postprocess_joint_pressure_strain_dream import (
        load_strain_only_comparison, posterior_statistics)
    stage3=_build_stage3_comparison(staged,passed=False)
    posterior=posterior_statistics(np.random.default_rng(2).normal([-6,-6],[.02,.03],size=(500,2)))
    with pytest.raises(ValueError,match='passed diagnostics'):
        load_strain_only_comparison(stage3,_minimal_joint_manifest(staged),posterior)
