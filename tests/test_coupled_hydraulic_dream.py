"""Stage 3 pure numerical tests; no expensive FV run or PyDREAM dependency."""
import json
import numpy as np
import pytest

from avant_model.inversion.coupled_dream import (
    converged, generate_starts, load_stage1, normalize_chains, rank_diagnostics,
    sha256_file,
)
from avant_model.inversion.coupled_parameters import HydraulicBounds


def test_dispersed_starts_reproducible_in_prior():
    bounds = HydraulicBounds((-7, -4), (-7, -4))
    a = generate_starts([-5.997, -5.974], bounds, 4, seed=22)
    b = generate_starts([-5.997, -5.974], bounds, 4, seed=22)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 2)
    assert np.unique(a, axis=0).shape[0] == 4
    assert all(bounds.contains(x) for x in a)


def test_bad_starts_rejected():
    bounds = HydraulicBounds()
    with pytest.raises(ValueError):
        generate_starts([-8, -6], bounds, 4)
    with pytest.raises(ValueError):
        generate_starts([-6, -6], bounds, 2)
    with pytest.raises(ValueError):
        generate_starts([-6, -6], bounds, 4, width=[0, 0.01])


def test_normalize_chains_and_logps():
    x = [np.full((31, 2), float(i)) for i in range(4)]
    lp = [np.full((31, 1), -float(i)) for i in range(4)]
    a, b = normalize_chains(x, lp, 31, 4)
    assert a.shape == (4, 31, 2)
    assert b.shape == (4, 31)
    with pytest.raises(ValueError):
        normalize_chains(x[:3], lp, 31, 4)


def test_iid_chains_have_good_diagnostics():
    rng = np.random.default_rng(101)
    x = rng.normal(size=(4, 1200, 2))
    d = rank_diagnostics(x, 200)
    for val in d.values():
        assert 0.97 < val['rank_folded_split_rhat'] < 1.04
        assert val['bulk_ess'] > 350
        assert val['tail_ess'] > 300
    assert converged(d, threshold=1.05, min_ess=300, retained_per_chain=1000)


def test_shifted_chains_are_detected():
    rng = np.random.default_rng(102)
    x = rng.normal(size=(4, 500, 2))
    x[0, :, 0] += 5.0
    d = rank_diagnostics(x, 100)
    assert d['log10_K_m_s']['rank_folded_split_rhat'] > 1.1
    assert not converged(d, threshold=1.01, min_ess=20, retained_per_chain=400)


def test_autocorrelation_reduces_ess():
    rng = np.random.default_rng(103)
    x = rng.normal(size=(4, 1400, 2))
    for k in range(1, 1400):
        x[:, k, :] += 0.95 * x[:, k-1, :]
    d = rank_diagnostics(x, 200)
    assert d['log10_K_m_s']['bulk_ess'] < 2000


def test_diagnostics_requires_enough_draws():
    with pytest.raises(ValueError):
        rank_diagnostics(np.ones((4, 9, 2)), 3)


def test_stage1_file_contract_and_config_hashes(tmp_path):
    source = tmp_path / 'stage1'
    source.mkdir()
    hyd, mec = tmp_path / 'hyd.json', tmp_path / 'mech.json'
    hyd.write_text('{"dt_hours": 0.5}', encoding='utf-8')
    mec.write_text('{"geometry": {}}', encoding='utf-8')
    summary = {
        'scope': 'Same-model synthetic fixed-geometry K/Ss likelihood grid; not field validation',
        'noise_std_ns': 2., 'hydraulic_config_sha256': sha256_file(hyd),
        'mechanical_config_sha256': sha256_file(mec),
    }
    (source / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    channels = [f'eXX_S{i:02d}' for i in range(1, 9)] + [f'eYY_S{i:02d}' for i in range(1, 9)] + [f'eXY_S{i:02d}' for i in range(1, 9)] + [f'eZZ_S{i:02d}' for i in range(1, 9)]
    train = np.array([not c.endswith(('S07', 'S08')) for c in channels])
    np.savez(source / 'synthetic_observations.npz', times_h=np.arange(17, dtype=float),
             channels=np.asarray(channels), observed_ns=np.zeros((17, 32)),
             train_mask=train, holdout_mask=~train)
    _, obs = load_stage1(source, hyd, mec)
    assert np.sum(obs.training_mask) == 24
    assert np.sum(obs.holdout_mask) == 8
    mec.write_text('{"geometry":{"changed":true}}', encoding='utf-8')
    with pytest.raises(ValueError, match='Configuration differs'):
        load_stage1(source, hyd, mec)


def test_lightweight_worker_callable_is_picklable():
    import pickle
    from avant_model.inversion.coupled_dream import WorkerHydraulicLikelihood
    worker = WorkerHydraulicLikelihood('saved', 'hyd.json', 'mechanical.json', (-7., -4.), (-7., -4.))
    blob = pickle.dumps(worker)
    restored = pickle.loads(blob)
    assert restored.bounds_ss == (-7., -4.)
    assert len(blob) < 1000  # critical Windows parallelism performance guard
