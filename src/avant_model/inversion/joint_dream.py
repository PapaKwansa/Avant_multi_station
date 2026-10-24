"""Stage 4B: audited, saved-data joint FV pressure + cuboid strain DREAM adapter.

The Stage 4A observations are immutable inputs: no noise is drawn here. Each
candidate uses ONE homogeneous FV solve to predict pressure at gauges and
strain using the existing fixed H = G @ A analytical pressure superposition.
"""
from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from avant_model.inversion.coupled_dream import build_likelihood, load_stage1, sha256_file
from avant_model.inversion.coupled_parameters import HydraulicBounds
from avant_model.inversion.joint_pressure_strain import (
    JointPressureStrainLikelihood, PressureObservations,
)

import json


def _require_hash(path, expected, description):
    path = Path(path).resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f'{description} is missing or has changed since Stage 4A: {path}')
    return path


def load_stage4a(stage4a_dir, hydraulic_config, mechanical_config, gauges_csv):
    """Validate provenance, times, coordinates and holdout masks before DREAM.

    Returns (stage4a_summary, stage1_summary, strain_obs, pressure_obs).
    Validation does NOT invoke the hydraulic or mechanical forward model.
    """
    stage4a_dir = Path(stage4a_dir).resolve()
    report_path = stage4a_dir / 'joint_summary.json'
    saved_path = stage4a_dir / 'synthetic_pressure_observations.npz'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not str(report.get('scope', '')).startswith('Stage 4A same-model synthetic joint'):
        raise ValueError('Stage 4B requires a Stage 4A synthetic joint output directory')
    _require_hash(hydraulic_config, report['hydraulic_config_sha256'], 'Hydraulic config')
    _require_hash(mechanical_config, report['mechanical_config_sha256'], 'Mechanical config')
    _require_hash(gauges_csv, report['gauge_csv_sha256'], 'Gauge coordinate CSV')
    source = Path(report['source_dir']).resolve()
    _require_hash(source / 'synthetic_observations.npz',
                  report['strain_observations_sha256'], 'Saved Stage 1 strain observations')
    stage1_summary, strain_obs = load_stage1(source, hydraulic_config, mechanical_config)
    with np.load(saved_path, allow_pickle=False) as data:
        p_obs = PressureObservations(
            times_h=data['times_h'], gauge_names=tuple(map(str, data['gauge_names'])),
            gauge_x_m=data['gauge_x_m'], gauge_y_m=data['gauge_y_m'],
            observed_pa=data['observed_pa'], sigma_pa=float(data['sigma_pa']),
            training_mask=data['train_mask'], holdout_mask=data['holdout_mask'])
    if not np.array_equal(strain_obs.times_h, p_obs.times_h):
        raise ValueError('Stage 1 strain and Stage 4A pressure timestamps differ')
    if not np.isclose(p_obs.sigma_pa, float(report['pressure_sigma_pa']), rtol=0, atol=1e-9):
        raise ValueError('Pressure noise scale differs from Stage 4A record')
    if not np.isclose(strain_obs.sigma_ns, float(report['strain_sigma_ns']), rtol=0, atol=1e-12):
        raise ValueError('Strain noise scale differs from Stage 4A record')
    with open(gauges_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['gauge', 'x_local_m', 'y_local_m', 'role']:
            raise ValueError('Gauge CSV has unexpected columns')
        csv_gauges = list(reader)
    if len(csv_gauges) != len(p_obs.gauge_names) or len(report['pressure_gauges']) != len(p_obs.gauge_names):
        raise ValueError('Stage 4A gauge counts differ')
    for j, (row, recorded) in enumerate(zip(csv_gauges, report['pressure_gauges'])):
        name, role = str(row['gauge']), str(row['role']).strip().lower()
        if name != p_obs.gauge_names[j] or name != recorded['gauge']:
            raise ValueError('Gauge ordering differs from Stage 4A observations')
        if role not in ('train', 'holdout') or role != str(recorded['role']).lower():
            raise ValueError('Gauge calibration/holdout role differs')
        if bool(p_obs.training_mask[j]) != (role == 'train'):
            raise ValueError('Pressure mask differs from recorded gauge role')
        for coord, observed, recorded_coord in (
            (row['x_local_m'], p_obs.gauge_x_m[j], recorded['x_local_m']),
            (row['y_local_m'], p_obs.gauge_y_m[j], recorded['y_local_m']),
        ):
            if not np.isclose(float(coord), observed, rtol=0, atol=1e-9) or not np.isclose(
                    float(recorded_coord), observed, rtol=0, atol=1e-9):
                raise ValueError('Gauge coordinates differ between CSV and saved observations')
    counts = {'n_training_strain': int(strain_obs.times_h.size * strain_obs.training_mask.sum()),
              'n_heldout_strain': int(strain_obs.times_h.size * strain_obs.holdout_mask.sum()),
              'n_training_pressure': int(p_obs.times_h.size * p_obs.training_mask.sum()),
              'n_heldout_pressure': int(p_obs.times_h.size * p_obs.holdout_mask.sum())}
    if any(counts[k] != int(report[k]) for k in counts):
        raise ValueError('Stage 4A training/holdout counts differ')
    return report, stage1_summary, strain_obs, p_obs


def joint_mle_record(report):
    records = [r for r in report['fit_results'] if r.get('method') == 'joint_MLE']
    if len(records) != 1:
        raise ValueError('Stage 4A must contain exactly one joint_MLE result')
    return records[0]


def build_joint_likelihood(hydraulic_config, mechanical_config, stage1_summary,
                           strain_obs, pressure_obs, bounds):
    """Build analytical operator once, then share one FV solve for both observables."""
    strain_likelihood, hydraulic_cfg = build_likelihood(
        hydraulic_config, mechanical_config, stage1_summary, strain_obs, bounds)
    return JointPressureStrainLikelihood(strain_likelihood.model, strain_obs,
                                         pressure_obs, bounds), hydraulic_cfg


def validate_joint_mle(likelihood, report, bounds, tolerance=1e-5):
    record = joint_mle_record(report)
    theta = np.array((record['log10_K'], record['log10_Ss']), float)
    if not bounds.contains(theta):
        raise ValueError('Stage 4A joint optimum is outside the prior bounds')
    actual = float(likelihood(theta))
    expected = float(record['log_likelihood_joint_training'])
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError('Stage 4B joint likelihood does not reproduce Stage 4A MLE')
    # This also detects accidental changes to the strain-only comparison.
    s, p = likelihood.log_likelihood_blocks(theta)
    if (abs(s - float(record['log_likelihood_strain_training'])) > tolerance or
            abs(p - float(record['log_likelihood_pressure_training'])) > tolerance):
        raise ValueError('Joint likelihood blocks disagree with Stage 4A')
    return theta, actual


@lru_cache(maxsize=8)
def _cached_joint(stage4a_dir, hydraulic_config, mechanical_config,
                  gauges_csv, bounds_k, bounds_ss):
    report, stage1, strain, pressure = load_stage4a(
        stage4a_dir, hydraulic_config, mechanical_config, gauges_csv)
    ll, _ = build_joint_likelihood(
        hydraulic_config, mechanical_config, stage1, strain, pressure,
        HydraulicBounds(bounds_k, bounds_ss))
    validate_joint_mle(ll, report, HydraulicBounds(bounds_k, bounds_ss))
    return ll


@dataclass(frozen=True)
class WorkerJointLikelihood:
    """Lightweight, pickleable PyDREAM callable; per-worker mechanical cache."""
    stage4a_dir: str
    hydraulic_config: str
    mechanical_config: str
    gauges_csv: str
    bounds_k: tuple
    bounds_ss: tuple

    def __call__(self, theta):
        return _cached_joint(self.stage4a_dir, self.hydraulic_config,
                             self.mechanical_config, self.gauges_csv,
                             self.bounds_k, self.bounds_ss)(theta)


def worker_is_pickleable(worker):
    pickle.dumps(worker)
    return True
