#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian likelihood for the AVANT multi-station analytical strain model.

This module contains the reusable statistical model used by the PyDREAM
Bayesian inversion.

Physical parameters of interest
-------------------------------
    a
        Inclusion semi-dimension along the local x-axis [m].

    b
        Inclusion semi-dimension along the local y-axis [m].

    h
        Inclusion depth parameter [m].

    theta_deg
        Inclusion rotation angle [degrees], following the coordinate
        convention used by the analytical forward model.

    x0_prime
        Inclusion-center x-coordinate in the primed/global station
        coordinate system [m].

    y0_prime
        Inclusion-center y-coordinate in the primed/global station
        coordinate system [m].

Statistical nuisance parameter
------------------------------
    log10_sigma_strain
        Base-10 logarithm of the effective residual strain-error scale
        in nanostrain.

The complete MCMC state is therefore

    [
        a,
        b,
        h,
        theta_deg,
        x0_prime,
        y0_prime,
        log10_sigma_strain,
    ]

The first six quantities are the physical parameters of interest.
The final quantity is a nuisance/error-model parameter.

The effective residual scale sigma_strain can represent aggregate
measurement uncertainty, unresolved variability, and model-data
discrepancy between the simplified analytical solution and the
reference/observed strain dataset.

An optional volumetric likelihood contribution is also supported.
It is disabled unless all required volume-observation arguments are
provided.

No priors and no MCMC controls are defined in this module. Those belong
to the production inversion runner/configuration.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from avant_model.model.forward_model_multi_station import (
    forward_model_multi_station,
)


# ============================================================================
# CANONICAL PARAMETER DEFINITIONS
# ============================================================================

PHYSICAL_PARAMETER_NAMES = (
    "a",
    "b",
    "h",
    "theta_deg",
    "x0_prime",
    "y0_prime",
)

NUISANCE_PARAMETER_NAMES = (
    "log10_sigma_strain",
)

SAMPLED_PARAMETER_NAMES = (
    *PHYSICAL_PARAMETER_NAMES,
    *NUISANCE_PARAMETER_NAMES,
)


# ============================================================================
# DATA-VECTOR HELPERS
# ============================================================================

def flatten_dataframe_columns(
    dataframe,
    columns: Sequence[str],
) -> np.ndarray:
    """
    Flatten selected DataFrame columns in component-major order.

    Parameters
    ----------
    dataframe
        Pandas DataFrame containing model predictions or observations.

    columns
        Ordered sequence of column names.

    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point vector.

    Notes
    -----
    The ordering of ``columns`` is scientifically important. The same
    ordering must be used for the observed and predicted strain vectors.

    For the current AVANT workflow the canonical ordering is:

        eXX for all stations,
        eYY for all stations,
        eXY for all stations,
        eZZ for all stations.

    Each selected time series is concatenated in the supplied column order.
    """

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            "DataFrame is missing required strain columns: "
            + ", ".join(missing)
        )

    return np.concatenate(
        [
            dataframe[column]
            .to_numpy(dtype=float)
            .ravel()
            for column in columns
        ]
    )


# Backward-compatible private alias.
_flatten_df_to_vector = flatten_dataframe_columns


# ============================================================================
# ELASTIC / VOLUME HELPERS
# ============================================================================

def bulk_modulus(
    E: float,
    nu: float,
) -> float:
    """
    Return isotropic bulk modulus.

    K = E / [3 (1 - 2 nu)]

    Parameters
    ----------
    E
        Young's modulus [Pa].

    nu
        Poisson's ratio [-].
    """

    E = float(E)
    nu = float(nu)

    if not np.isfinite(E) or E <= 0.0:
        raise ValueError(
            "Young's modulus E must be finite and positive."
        )

    if not np.isfinite(nu):
        raise ValueError(
            "Poisson's ratio nu must be finite."
        )

    denominator = 3.0 * (
        1.0 - 2.0 * nu
    )

    if denominator <= 0.0:
        raise ValueError(
            "Poisson's ratio must satisfy nu < 0.5 "
            "for a finite positive bulk modulus."
        )

    return E / denominator


def cuboid_volume(
    a: float,
    b: float,
    c: float,
) -> float:
    """
    Return the undeformed volume of the analytical cuboid.

    The analytical geometry uses semi-dimensions a, b, and c, so the
    full dimensions are

        2a x 2b x 2c

    and therefore

        V0 = 8 a b c.
    """

    a = float(a)
    b = float(b)
    c = float(c)

    if (
        not np.isfinite(a)
        or not np.isfinite(b)
        or not np.isfinite(c)
        or a <= 0.0
        or b <= 0.0
        or c <= 0.0
    ):
        raise ValueError(
            "Cuboid semi-dimensions a, b, and c must "
            "be finite and positive."
        )

    return (
        8.0
        * a
        * b
        * c
    )


def volume_forward(
    a: float,
    b: float,
    c: float,
    E: float,
    nu: float,
    delta_P,
) -> np.ndarray:
    """
    Approximate pressure-dependent inclusion volume.

    Parameters
    ----------
    a, b, c
        Cuboid semi-dimensions [m].

    E
        Young's modulus [Pa].

    nu
        Poisson's ratio [-].

    delta_P
        Scalar or array-like pressure change [Pa].

    Returns
    -------
    numpy.ndarray
        Predicted volume corresponding to ``delta_P``.

    Notes
    -----
    This preserves the simple bulk-compressibility approximation used in
    the earlier Bayesian workflow:

        V = V0 (1 + delta_P / K)

    where

        V0 = 8 a b c

    for the analytical cuboid and

        K = E / [3(1 - 2 nu)].

    This term is optional and is not part of the default strain-only
    likelihood.
    """

    V0 = cuboid_volume(
        a,
        b,
        c,
    )

    K = bulk_modulus(
        E,
        nu,
    )

    delta_P = np.asarray(
        delta_P,
        dtype=float,
    )

    return V0 * (
        1.0
        + delta_P / K
    )


# ============================================================================
# PARAMETER UNPACKING
# ============================================================================

def unpack_sampled_parameters(
    sampled_parameters: Iterable[float],
) -> dict:
    """
    Convert the canonical seven-dimensional MCMC state to named values.

    Returns
    -------
    dict
        Keys:

            a
            b
            h
            theta_deg
            x0_prime
            y0_prime
            log10_sigma_strain
            sigma_strain
    """

    values = np.asarray(
        sampled_parameters,
        dtype=float,
    ).ravel()

    expected = len(
        SAMPLED_PARAMETER_NAMES
    )

    if values.size != expected:
        raise ValueError(
            "Expected "
            f"{expected} sampled parameters "
            f"{SAMPLED_PARAMETER_NAMES}, "
            f"but received {values.size}."
        )

    (
        a,
        b,
        h,
        theta_deg,
        x0_prime,
        y0_prime,
        log10_sigma_strain,
    ) = values

    sigma_strain = (
        10.0
        ** log10_sigma_strain
    )

    return {
        "a": float(a),
        "b": float(b),
        "h": float(h),
        "theta_deg": float(theta_deg),
        "x0_prime": float(x0_prime),
        "y0_prime": float(y0_prime),
        "log10_sigma_strain": float(
            log10_sigma_strain
        ),
        "sigma_strain": float(
            sigma_strain
        ),
    }


def parameters_are_valid(
    parameters: dict,
) -> bool:
    """
    Basic physical/statistical validity check for sampled parameters.

    Prior bounds remain the responsibility of the MCMC prior definitions.
    """

    positive_names = (
        "a",
        "b",
        "h",
        "sigma_strain",
    )

    for name in positive_names:
        value = parameters[name]

        if (
            not np.isfinite(value)
            or value <= 0.0
        ):
            return False

    for name in (
        "theta_deg",
        "x0_prime",
        "y0_prime",
        "log10_sigma_strain",
    ):
        if not np.isfinite(
            parameters[name]
        ):
            return False

    return True


# ============================================================================
# GAUSSIAN LOG-LIKELIHOOD HELPERS
# ============================================================================

def gaussian_log_likelihood(
    residual,
    sigma: float,
) -> float:
    """
    IID Gaussian log-likelihood for a residual vector.

    Parameters
    ----------
    residual
        Observed minus predicted values.

    sigma
        Positive Gaussian standard deviation in the same units as residual.
    """

    residual = np.asarray(
        residual,
        dtype=float,
    ).ravel()

    sigma = float(sigma)

    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        return -np.inf

    if not np.all(
        np.isfinite(residual)
    ):
        return -np.inf

    n = residual.size

    if n == 0:
        raise ValueError(
            "Residual vector is empty."
        )

    standardized = (
        residual / sigma
    )

    return float(
        -0.5
        * np.sum(
            standardized ** 2
        )
        - n * np.log(sigma)
        - 0.5
        * n
        * np.log(
            2.0 * np.pi
        )
    )


# ============================================================================
# FORWARD PREDICTION
# ============================================================================

def predict_strain_vector(
    sampled_parameters,
    *,
    time,
    x_prime,
    y_prime,
    z,
    c: float,
    E: float,
    nu: float,
    pmax: float,
    tpeak: float,
    d: float,
    alpha: float,
    component_cols: Sequence[str],
    station_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Evaluate the analytical multi-station model for one MCMC state.

    Parameters sampled by the MCMC are

        a,
        b,
        h,
        theta_deg,
        x0_prime,
        y0_prime,
        log10_sigma_strain.

    The quantities

        c,
        E,
        nu,
        pmax,
        tpeak,
        d,
        alpha

    are fixed inputs to this production Bayesian model.
    """

    p = unpack_sampled_parameters(
        sampled_parameters
    )

    if not parameters_are_valid(p):
        raise ValueError(
            "Nonphysical or non-finite sampled parameter state."
        )

    df_pred = forward_model_multi_station(
        pmax=float(pmax),
        tpeak=float(tpeak),
        d=float(d),
        time=np.asarray(
            time,
            dtype=float,
        ),
        x_prime=np.asarray(
            x_prime,
            dtype=float,
        ),
        y_prime=np.asarray(
            y_prime,
            dtype=float,
        ),
        x0_prime=p["x0_prime"],
        y0_prime=p["y0_prime"],
        z=np.asarray(
            z,
            dtype=float,
        ),
        a=p["a"],
        b=p["b"],
        c=float(c),
        nu=float(nu),
        h=p["h"],
        E=float(E),
        theta_deg=p["theta_deg"],
        alpha=float(alpha),
        station_names=station_names,
    )

    return flatten_dataframe_columns(
        df_pred,
        component_cols,
    )


# ============================================================================
# MAIN BAYESIAN LIKELIHOOD
# ============================================================================

def likelihood(
    sampled_parameters,
    observed_vector,
    *,
    time,
    x_prime,
    y_prime,
    z,
    c: float,
    E: float,
    nu: float,
    pmax: float,
    tpeak: float,
    d: float,
    alpha: float,
    component_cols: Sequence[str],
    station_names: Optional[Sequence[str]] = None,
    volume_obs=None,
    delta_P=None,
    sigma_volume: Optional[float] = None,
    volume_weight: float = 1.0,
) -> float:
    """
    Return the total log-likelihood for one MCMC state.

    The default likelihood is strain-only.

    If ``volume_obs``, ``delta_P``, and ``sigma_volume`` are all supplied,
    an optional volumetric likelihood contribution is added.

    Parameters
    ----------
    sampled_parameters
        Canonical MCMC state:

            [
                a,
                b,
                h,
                theta_deg,
                x0_prime,
                y0_prime,
                log10_sigma_strain,
            ]

    observed_vector
        Flattened observed strain vector in nanostrain.

    time
        Observation times [s].

    x_prime, y_prime
        Station coordinates [m].

    z
        Station vertical coordinates [m].

    c
        Fixed inclusion vertical semi-dimension [m].

    E
        Fixed Young's modulus [Pa].

    nu
        Fixed Poisson's ratio.

    pmax, tpeak, d
        Fixed transient-pressure-model parameters.

    alpha
        Fixed Biot coefficient.

    component_cols
        Ordered model/observation strain columns.

    station_names
        Optional station-name sequence.

    volume_obs
        Optional observed volume value(s).

    delta_P
        Optional pressure change(s) corresponding to volume observations.

    sigma_volume
        Optional volume-observation error standard deviation.

    volume_weight
        Optional relative multiplier on the volume log-likelihood.

    Returns
    -------
    float
        Total log-likelihood.
    """

    try:
        p = unpack_sampled_parameters(
            sampled_parameters
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return -np.inf

    if not parameters_are_valid(p):
        return -np.inf

    observed_vector = np.asarray(
        observed_vector,
        dtype=float,
    ).ravel()

    if (
        observed_vector.size == 0
        or not np.all(
            np.isfinite(
                observed_vector
            )
        )
    ):
        raise ValueError(
            "Observed strain vector must be finite and non-empty."
        )

    # ------------------------------------------------------------------
    # Analytical strain prediction
    # ------------------------------------------------------------------

    try:
        predicted_vector = (
            predict_strain_vector(
                sampled_parameters,
                time=time,
                x_prime=x_prime,
                y_prime=y_prime,
                z=z,
                c=c,
                E=E,
                nu=nu,
                pmax=pmax,
                tpeak=tpeak,
                d=d,
                alpha=alpha,
                component_cols=component_cols,
                station_names=station_names,
            )
        )
    except Exception:
        # During MCMC, invalid model states should simply receive zero
        # posterior support rather than terminating the full sampler.
        return -np.inf

    if (
        predicted_vector.shape
        != observed_vector.shape
    ):
        raise ValueError(
            "Observed/predicted strain-vector size mismatch: "
            f"observed={observed_vector.shape}, "
            f"predicted={predicted_vector.shape}."
        )

    residual_strain = (
        observed_vector
        - predicted_vector
    )

    ll_strain = gaussian_log_likelihood(
        residual_strain,
        p["sigma_strain"],
    )

    if not np.isfinite(
        ll_strain
    ):
        return -np.inf

    total_log_likelihood = (
        ll_strain
    )

    # ------------------------------------------------------------------
    # Optional volume constraint
    # ------------------------------------------------------------------

    use_volume = all(
        value is not None
        for value in (
            volume_obs,
            delta_P,
            sigma_volume,
        )
    )

    if use_volume:
        if (
            not np.isfinite(
                float(volume_weight)
            )
            or float(volume_weight) < 0.0
        ):
            raise ValueError(
                "volume_weight must be finite and non-negative."
            )

        volume_obs_array = np.asarray(
            volume_obs,
            dtype=float,
        ).ravel()

        delta_P_array = np.asarray(
            delta_P,
            dtype=float,
        ).ravel()

        if volume_obs_array.size == 0:
            raise ValueError(
                "volume_obs is empty."
            )

        if delta_P_array.size not in (
            1,
            volume_obs_array.size,
        ):
            raise ValueError(
                "delta_P must be scalar or have the same number "
                "of entries as volume_obs."
            )

        volume_pred = np.asarray(
            volume_forward(
                a=p["a"],
                b=p["b"],
                c=float(c),
                E=float(E),
                nu=float(nu),
                delta_P=delta_P_array,
            ),
            dtype=float,
        ).ravel()

        if (
            volume_pred.size == 1
            and volume_obs_array.size > 1
        ):
            volume_pred = np.full(
                volume_obs_array.shape,
                volume_pred.item(),
                dtype=float,
            )

        residual_volume = (
            volume_obs_array
            - volume_pred
        )

        ll_volume = gaussian_log_likelihood(
            residual_volume,
            float(
                sigma_volume
            ),
        )

        if not np.isfinite(
            ll_volume
        ):
            return -np.inf

        total_log_likelihood += (
            float(volume_weight)
            * ll_volume
        )

    return float(
        total_log_likelihood
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "PHYSICAL_PARAMETER_NAMES",
    "NUISANCE_PARAMETER_NAMES",
    "SAMPLED_PARAMETER_NAMES",
    "flatten_dataframe_columns",
    "bulk_modulus",
    "cuboid_volume",
    "volume_forward",
    "unpack_sampled_parameters",
    "parameters_are_valid",
    "gaussian_log_likelihood",
    "predict_strain_vector",
    "likelihood",
]