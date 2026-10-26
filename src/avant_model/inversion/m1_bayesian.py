"""Auditable M1 Bayesian likelihood and geology-informed priors.

Read exactly the saved M1 MAP synthetic observations. Do NOT generate noise,
fix parameters at their generating values, or use held-out observations in the
likelihood. Uniform/softly informative priors are illustrative, NOT measured
geological facts. Python modules with the legacy strain kernel are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
from scipy.stats import truncnorm

from avant_model.inversion.m1_map_recovery import (
    PARAMETER_NAMES, SEARCH_BOUNDS, ObservationLayout,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prior_config(path):
    """Validate all nine priors; never silently fix an unknown to truth."""
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema_version') != 'm1-bayes-priors-1':
        raise ValueError('Unrecognized M1 Bayesian prior schema')
    specs = doc.get('parameters', {})
    if set(specs) != set(PARAMETER_NAMES):
        raise ValueError('Prior file must specify each of nine M1 parameters once')
    for name in PARAMETER_NAMES:
        item = specs[name]
        if item.get('distribution') not in ('uniform', 'truncated_normal'):
            raise ValueError(f'Unsupported prior distribution for {name}')
        bounds = np.asarray(item.get('bounds', ()), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f'Invalid prior support for {name}')
        allowed = SEARCH_BOUNDS[name]
        if bounds[0] < allowed[0] - 1e-12 or bounds[1] > allowed[1] + 1e-12:
            raise ValueError(f'Prior {name} exceeds verified synthetic search limits')
        if item['distribution'] == 'truncated_normal':
            mean, sd = float(item['mean']), float(item['sd'])
            if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0):
                raise ValueError(f'Invalid normal prior parameters for {name}')
            if not bounds[0] < mean < bounds[1]:
                raise ValueError(f'Prior mean for {name} must be within bounds')
    if doc.get('prior_provenance') != 'illustrative_not_field_measured':
        raise ValueError('Prior provenance must explicitly mark assumptions as illustrative')
    return doc


def _pdf_for(item):
    lo, hi = map(float, item['bounds'])
    if item['distribution'] == 'uniform':
        return None
    mu, sd = float(item['mean']), float(item['sd'])
    return truncnorm((lo-mu)/sd, (hi-mu)/sd, loc=mu, scale=sd)


def log_prior(theta, priors):
    """Normalized bounded log density, including truncation normalization."""
    x = np.asarray(theta, dtype=float)
    if x.shape != (9,) or not np.isfinite(x).all():
        return -np.inf
    total = 0.0
    for j, name in enumerate(PARAMETER_NAMES):
        item = priors['parameters'][name]
        lo, hi = item['bounds']
        if x[j] < lo or x[j] > hi:
            return -np.inf
        if item['distribution'] == 'uniform':
            total -= np.log(hi-lo)
        else:
            total += float(_pdf_for(item).logpdf(x[j]))
    return float(total)


def sample_prior(priors, rng):
    """Draw independent full nine-dimensional starts; no oracle fixing."""
    x = []
    for name in PARAMETER_NAMES:
        item = priors['parameters'][name]
        if item['distribution'] == 'uniform':
            lo, hi = item['bounds']
            x.append(rng.uniform(lo, hi))
        else:
            x.append(float(_pdf_for(item).ppf(rng.uniform(1e-9, 1-1e-9))))
    return np.array(x, dtype=float)


def make_pydream_priors(priors):
    """Import PyDREAM lazily; SciPy priors and reported prior density agree."""
    from pydream.parameters import SampledParam
    from scipy.stats import uniform
    result = []
    for name in PARAMETER_NAMES:
        item = priors['parameters'][name]
        lo, hi = item['bounds']
        if item['distribution'] == 'uniform':
            result.append(SampledParam(uniform, loc=lo, scale=hi-lo))
        else:
            mu, sd = item['mean'], item['sd']
            result.append(SampledParam(truncnorm, a=(lo-mu)/sd, b=(hi-mu)/sd,
                                       loc=mu, scale=sd))
    return result


@dataclass(frozen=True)
class SavedM1:
    map_dir: str
    map_report: dict
    times_h: np.ndarray
    observed_strain_ns: np.ndarray
    observed_pressure_pa: np.ndarray
    gauges_prime_m: np.ndarray
    channel_names: tuple[str, ...]
    layout: ObservationLayout
    truth: np.ndarray
    observations_sha256: str
    map_report_sha256: str


def load_saved_m1(map_dir, config_path):
    """Enforce saved-data provenance, fixed train/held-out masks and geometry."""
    folder = Path(map_dir).resolve()
    report_path = folder/'m1_map_recovery_report.json'
    npz_path = folder/'m1_synthetic_observations.npz'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not str(report.get('scope', '')).startswith('Same-model synthetic bounded Gaussian MAP'):
        raise ValueError('Expected immutable M1 synthetic MAP report')
    config_path = Path(config_path).resolve()
    if sha256(config_path) != report['provenance_sha256']['config']:
        raise ValueError('M1 configuration was changed after synthetic data generation')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    for key, hash_name in (('stage6a_contract','contract'),('hydraulic_config','hydraulic_config'),
                           ('stations_csv','stations_csv')):
        if sha256(cfg[key]) != report['provenance_sha256'][hash_name]:
            raise ValueError(f'M1 input changed: {key}')
    with np.load(npz_path, allow_pickle=False) as z:
        t = np.asarray(z['times_h'], float)
        channel = tuple(map(str, z['channels']))
        so = np.asarray(z['strain_observed_ns'], float)
        po = np.asarray(z['pressure_observed_pa'], float)
        gauges = np.asarray(z['gauges_global_prime_m'], float)
        mtrain = np.asarray(z['train_channel_mask'], bool)
        mheld = np.asarray(z['heldout_channel_mask'], bool)
        truth = np.asarray(z['parameter_truth'], float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid saved timestamps')
    if not np.array_equal(t, np.asarray(report['time_h'], dtype=float)):
        raise ValueError('Saved timestamps do not match MAP report')
    if so.shape != (len(t),len(channel)) or po.shape != (len(t),len(gauges)):
        raise ValueError('Saved observation array shapes inconsistent')
    if gauges.shape != (3,2) or not np.allclose(gauges, report['training_gauges_global_prime_m'] +
                                                report['held_out_gauge_global_prime_m'], atol=0, rtol=0):
        raise ValueError('Saved gauge coordinates differ from MAP report')
    if truth.shape != (9,) or not np.allclose(truth, [report['generating_parameters'][k]
                                                        for k in PARAMETER_NAMES], atol=1e-12):
        raise ValueError('Saved synthetic reference vector differs from MAP report')
    if not np.isfinite(so).all() or not np.isfinite(po).all():
        raise ValueError('Nonfinite saved observations')
    mask_expected = np.array([c.rsplit('_',1)[-1] in ('S01','S02','S03','S04','S05','S06')
                              for c in channel], dtype=bool)
    if len(channel)!=32 or not np.array_equal(mtrain,mask_expected) or not np.array_equal(mheld,~mask_expected):
        raise ValueError('M1 station partitions or channel ordering changed')
    noise = report['noise_model']
    if noise['kind'] != 'independent Gaussian':
        raise ValueError('This sampler only supports the recorded independent-error synthetic benchmark')
    layout = ObservationLayout(mtrain,mheld,(0,1),(2,),
                               float(noise['strain_sigma_ns']),float(noise['pressure_sigma_pa']))
    if (layout.extract(so,po).shape[0] != report['n_training_strain']+report['n_training_pressure'] or
        layout.extract(so,po,subset='holdout').shape[0] != report['n_heldout_strain']+report['n_heldout_pressure']):
        raise ValueError('Training/holdout observation counts changed')
    return SavedM1(str(folder),report,t,so,po,gauges,channel,layout,truth,
                   sha256(npz_path),sha256(report_path))


def build_forward(config_path, saved: SavedM1):
    """Build one unchanged analytical kernel and one two-zone FV model per worker."""
    from avant_model.inversion.stage6_parameters import Stage6Contract
    from avant_model.model.stage6_coordinates import SourceRectangle
    from avant_model.model.stage6_m1_forward import Stage6M1Forward
    import pandas as pd
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    contract = Stage6Contract.from_json(Path(cfg['stage6a_contract']))
    hc = json.loads(Path(cfg['hydraulic_config']).read_text(encoding='utf-8'))
    hc['nx'] = int(saved.map_report['grid_n'])
    hc['ny'] = int(saved.map_report['grid_n'])
    stations = pd.read_csv(cfg['stations_csv']).to_dict(orient='list')
    support = SourceRectangle(**cfg['mechanical_support'])
    model = Stage6M1Forward(hc,contract.domain,contract.mechanics,stations,support,
                            splits=tuple(cfg['mechanical_tile_splits']),
                            quadrature_order=int(cfg['quadrature_order']),
                            gauges_prime_m=saved.gauges_prime_m)
    if tuple(model.channels) != saved.channel_names:
        raise ValueError('Production kernel channel order changed since saved MAP observations')
    return model


class M1GammaCachedForward:
    """Same exact gamma-only caching as saved M1 MAP; no cross-script imports.

    Gamma changes only the linear-elastic strain amplitude, not the FV pressure.
    Other parameter changes trigger a fresh full M1 forward solve.
    """
    def __init__(self, model, times_h):
        self.model = model
        self.times_h = np.asarray(times_h, dtype=float)
        self.key = None
        self.result = None
        self.full_forward_solves = 0

    def __call__(self, theta):
        x = np.asarray(theta, dtype=float)
        if x.shape != (9,) or not np.isfinite(x).all():
            raise ValueError('Expected finite M1 nine-parameter vector')
        physical_key = tuple(np.r_[x[:3], x[4:]])
        if self.key != physical_key:
            from avant_model.model.stage6_m1_forward import M1State
            from avant_model.model.stage6_coordinates import SourceRectangle
            base = M1State(10**x[0], 10**x[1], 10**x[2], 1., SourceRectangle(*x[4:]))
            self.result = self.model.predict(base, self.times_h)
            self.key = physical_key
            self.full_forward_solves += 1
        return (np.asarray(self.result.strain_ns)*10**x[3],
                np.asarray(self.result.pressure_gauges_pa))


class M1BayesianLikelihood:
    """Training-only Gaussian likelihood; exact gamma scaling caches FV solves."""
    def __init__(self, saved: SavedM1, predict):
        self.saved = saved
        self.predict = predict
        self.data = saved.layout.extract(saved.observed_strain_ns,saved.observed_pressure_pa,
                                         normalized=True)

    def residual(self, theta):
        x = np.asarray(theta,float)
        if x.shape != (9,) or not np.isfinite(x).all():
            raise ValueError('M1 vector must have nine finite parameters')
        from avant_model.inversion.m1_map_recovery import check_bounds
        b = check_bounds()
        if np.any(x < b[:,0]) or np.any(x > b[:,1]):
            raise ValueError('M1 vector outside verified bounds')
        strain, pressure = self.predict(x)
        return self.saved.layout.extract(strain,pressure,normalized=True)-self.data

    def __call__(self, theta):
        try:
            r = self.residual(theta)
            return float(-0.5 * np.dot(r,r))
        except (ValueError, FloatingPointError, OverflowError):
            return -np.inf


@lru_cache(maxsize=2)
def _worker_problem(map_dir, config_path):
    saved = load_saved_m1(map_dir,config_path)
    model = build_forward(config_path,saved)
    predictor = M1GammaCachedForward(model,saved.times_h)
    return M1BayesianLikelihood(saved,predictor)


@dataclass(frozen=True)
class WorkerM1Likelihood:
    map_dir: str
    config_path: str

    def __call__(self, theta):
        return _worker_problem(self.map_dir,self.config_path)(theta)


def assert_worker_pickles(worker):
    pickle.loads(pickle.dumps(worker))
    return True


def validate_map_likelihood(likelihood, report, tolerance=1e-3):
    """Assert that fixed synthetic likelihood reproduces existing full MAP chi2."""
    record = report.get('fits',{}).get('full',{}).get('best')
    if record is None:
        raise ValueError('Saved M1 report must include a successful full nine-parameter MAP')
    theta = np.asarray(record['fitted_parameters'],float)
    ll = float(likelihood(theta))
    expected = -float(record['chi2'])/2
    if not np.isfinite(ll) or abs(ll-expected) > tolerance:
        raise ValueError(f'M1 MAP likelihood mismatch: computed {ll}, saved {expected}')
    return theta,ll


def generate_chain_starts(center, priors, *, nchains=4, seed=20261026):
    """Two MAP-near starts plus prior-dispersed starts; never use truth for starts."""
    x = np.asarray(center,float)
    if x.shape != (9,) or not np.isfinite(x).all() or nchains < 4:
        raise ValueError('Need finite full MAP and >=4 chains')
    rng = np.random.default_rng(seed)
    starts = []
    for i in range(nchains):
        if i < (nchains+1)//2:
            for attempt in range(100):
                trial = x + rng.normal(0,0.035,size=9) * np.array(
                    [priors['parameters'][name]['bounds'][1]-priors['parameters'][name]['bounds'][0]
                     for name in PARAMETER_NAMES])
                if np.isfinite(log_prior(trial,priors)):
                    break
            else:
                raise RuntimeError('Could not initialize MAP-near chain inside prior')
        else:
            trial = sample_prior(priors,rng)
        starts.append(trial)
    return np.array(starts)


def normalize_dream(chains, logps, niterations, nchains):
    if len(chains) != nchains or len(logps) != nchains:
        raise ValueError('Unexpected chain count')
    x = np.stack([np.asarray(c,float) for c in chains])
    p = np.stack([np.asarray(c,float).reshape(-1) for c in logps])
    if x.shape != (nchains,niterations,9) or p.shape != (nchains,niterations):
        raise ValueError(f'Unexpected DREAM output shapes {x.shape} and {p.shape}')
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite DREAM outputs')
    return x,p
