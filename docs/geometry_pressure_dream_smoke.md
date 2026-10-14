# Stage 3: synthetic geometry-pressure DREAM smoke

This additive runner uses the **validated analytical-depth station snapshot** from Larry's benchmark. It keeps `c`, `h`, and elastic properties fixed and samples `[a, b-a, theta_deg, x0_prime, y0_prime, log10(sigma_ns)]`, so `b>a` by construction. Four pressure coefficients are **marginalized**, not sampled. The Gaussian pressure prior and Gaussian strain noise are explicitly configured.

The synthetic data are generated with a known geometry and known four-zone pressures. The marginal likelihood is evaluated using the existing `evaluate_geometry_pressure` adapter. The pressure prior mean is not the true synthetic pressure vector; this is intentional.

Run from repository root:

```bat
python -m pytest -q tests/test_geometry_pressure_dream_smoke.py
python -m pytest -q
python scripts/inversion/run_geometry_pressure_dream_smoke.py --config configs/pressure_superposition/larry_geometry_dream_smoke_001.json --output-dir results/pressure_superposition/geometry_smoke_eval_001 --evaluate-only
python scripts/inversion/run_geometry_pressure_dream_smoke.py --config configs/pressure_superposition/larry_geometry_dream_smoke_001.json --output-dir results/pressure_superposition/geometry_smoke_dream_001
```

The DREAM invocation requires PyDREAM and SciPy in the `vrugt` environment. It is intentionally only 20 iterations across 6 chains; these chains **must not** be used for credible intervals, convergence claims, or scientific geometry recovery. Check PyDREAM version compatibility and run the evaluation-only mode first.

The runner currently fits one **synthetic snapshot**; it does not use COMSOL observed strain, model-discrepancy covariance, or the shared-reference time covariance. It is not a full inversion and does not modify the production DREAM code. Its generated observations and chain files should be archived separately from the Git source.
