
"""Regression tests for the AVANT Theis analytical reference."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import exp1

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "verify_injection_against_theis.py"

spec = importlib.util.spec_from_file_location("theis_verification", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pressure(times):
    return module.theis_pressure(
        r=100.0,
        t_h=np.asarray(times, dtype=float),
        K=1e-6,
        Ss=1e-6,
        b=5.0,
        q=30.0 * module.GPM_TO_M3S,
        rho=1000.0,
        g=9.81,
        shutin_h=19.0,
    )


def test_initial_pressure_zero():
    assert pressure([0.0])[0] == 0.0


def test_injection_pressure_increases():
    p = pressure([1.0, 5.0, 10.0, 19.0])
    assert np.all(np.diff(p) > 0.0)


def test_recovery_pressure_decreases():
    p = pressure([19.0, 19.5, 24.0, 48.0, 96.0])
    assert np.all(np.diff(p) < 0.0)
    assert np.all(p > 0.0)


def test_exact_shutin_superposition():
    K = 1e-6
    Ss = 1e-6
    b = 5.0
    r = 100.0
    q = 30.0 * module.GPM_TO_M3S
    rho = 1000.0
    g = 9.81

    t = 24.0 * 3600.0
    ts = 19.0 * 3600.0
    D = K / Ss

    expected = rho * g * q / (4.0 * np.pi * K * b) * (
        exp1(r**2 / (4.0 * D * t))
        - exp1(r**2 / (4.0 * D * (t - ts)))
    )

    np.testing.assert_allclose(
        pressure([24.0])[0],
        expected,
        rtol=1e-12,
        atol=1e-10,
    )


def test_continuity_at_shutin():
    p = pressure([19.0, 19.0 + 1e-8])
    np.testing.assert_allclose(p[0], p[1], rtol=1e-8)


def test_no_nonfinite_pressures():
    times = np.linspace(0.0, 96.0, 1000)
    assert np.all(np.isfinite(pressure(times)))
