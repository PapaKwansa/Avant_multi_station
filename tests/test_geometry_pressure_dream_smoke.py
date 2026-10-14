"""Lightweight interface checks; actual kernel smoke runs via script in repo."""
import importlib.util
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('geometry_dream_smoke', ROOT / 'scripts/inversion/run_geometry_pressure_dream_smoke.py')
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_b_greater_than_a():
    geom, sigma = mod.geometry_from_vector([215, 185, 15, 53, 141, .3], {'c': 2.5, 'h': 522.5})
    assert geom.b == 400 and geom.b > geom.a
    assert np.isclose(sigma, 10**.3)


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        mod.geometry_from_vector([215, -1, 15, 53, 141, .3], {'c': 2.5, 'h': 522.5})


def test_synthetic_config_shapes():
    import json
    cfg = json.loads((ROOT / 'configs/pressure_superposition/larry_geometry_dream_smoke_001.json').read_text())
    assert np.asarray(cfg['bounds']).shape == (6, 2)
    assert np.prod(cfg['pressure_splits']) == len(cfg['synthetic_pressure_pa'])


def test_dream_likelihood_is_picklable_and_equivalent():
    cfg = {"pressure_splits": [1, 1, 1], "prior_mean_pa": 0.0, "prior_std_pa": 1e6}
    problem = {"fixed": {"c": 2.5, "h": 522.5}}
    callback = mod.make_dream_likelihood(problem, cfg)
    restored = pickle.loads(pickle.dumps(callback))
    theta = [-1, 185, 15, 53, 141, 0.3]
    assert restored(theta) == mod.log_likelihood(theta, problem, cfg)
