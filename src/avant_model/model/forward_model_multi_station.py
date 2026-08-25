"""
Multi-station analytical forward model for the Avant inclusion model.

This module orchestrates the core model components:

    multi_station_coord_transform
        -> transforms station coordinates into the inclusion frame

    multi_station_strain
        -> evaluates the analytical strain field

    multi_station_rotation
        -> rotates the strain tensor into the reported/global frame

The forward model itself does not read datasets from disk. All inputs are
provided explicitly by the caller, typically through:

    avant_model.data.multi_stations_input.read_input()

This separation allows the same forward model to be used by:

    - baseline model evaluation
    - deterministic fitting
    - sensitivity analysis
    - Bayesian inversion
    - plotting
    - validation tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import multi_station_coord_transform
from . import multi_station_rotation
from . import multi_station_strain


# =====================================================================
# PRESSURE HISTORY
# =====================================================================

def pressure_time_series(
    pmax: float,
    tpeak: float,
    d: float,
    time,
) -> np.ndarray:
    """
    Evaluate the piecewise pressure history.

    Pressure rises linearly from zero to pmax at tpeak and then
    decays exponentially.

    Parameters
    ----------
    pmax
        Peak pressure [Pa].

    tpeak
        Time of peak pressure [s].

    d
        Dimensionless post-peak decay parameter.

    time
        Time array [s].

    Returns
    -------
    numpy.ndarray
        Pressure history [Pa].
    """

    time = np.asarray(
        time,
        dtype=float,
    )

    if tpeak <= 0.0:
        raise ValueError(
            "tpeak must be greater than zero."
        )

    pressure = np.zeros_like(
        time,
        dtype=float,
    )

    rising = time <= tpeak
    falling = time > tpeak

    pressure[rising] = (
        pmax / tpeak
    ) * time[rising]

    pressure[falling] = (
        pmax
        * np.exp(
            -d
            * (
                time[falling] / tpeak
                - 1.0
            )
        )
    )

    return pressure


# =====================================================================
# GEOMETRY SCALING
# =====================================================================

def geometry_from_scale(
    s: float,
    a0: float,
    b0: float,
    c0: float,
) -> tuple[float, float, float]:
    """
    Convert dimensionless geometry scale into semi-axes.

    Parameters
    ----------
    s
        Dimensionless scale factor.

    a0, b0, c0
        Reference semi-axes [m].

    Returns
    -------
    tuple
        Scaled semi-axes (a, b, c) [m].
    """

    a = s * a0
    b = s * b0
    c = s * c0

    return a, b, c


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def _validate_station_inputs(
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
    station_names,
) -> list[str]:
    """
    Validate station and time inputs.

    Returns
    -------
    list[str]
        Cleaned station names.
    """

    if x_prime.ndim != 1:
        raise ValueError(
            "x_prime must be a one-dimensional array."
        )

    if y_prime.ndim != 1:
        raise ValueError(
            "y_prime must be a one-dimensional array."
        )

    if z.ndim != 1:
        raise ValueError(
            "z must be a one-dimensional array."
        )

    if time.ndim != 1:
        raise ValueError(
            "time must be a one-dimensional array."
        )

    if not (
        x_prime.size
        == y_prime.size
        == z.size
    ):
        raise ValueError(
            "x_prime, y_prime, and z must have "
            "the same number of stations."
        )

    if station_names is None:
        cleaned_names = [
            f"S{index + 1:02d}"
            for index in range(x_prime.size)
        ]
    else:
        cleaned_names = [
            str(name).strip()
            for name in station_names
        ]

        if len(cleaned_names) != x_prime.size:
            raise ValueError(
                "station_names length must match "
                "the number of stations."
            )

    if len(set(cleaned_names)) != len(cleaned_names):
        raise ValueError(
            "station_names must be unique."
        )

    if not np.isfinite(x_prime).all():
        raise ValueError(
            "x_prime contains NaN or infinite values."
        )

    if not np.isfinite(y_prime).all():
        raise ValueError(
            "y_prime contains NaN or infinite values."
        )

    if not np.isfinite(z).all():
        raise ValueError(
            "z contains NaN or infinite values."
        )

    if not np.isfinite(time).all():
        raise ValueError(
            "time contains NaN or infinite values."
        )

    if time.size == 0:
        raise ValueError(
            "time must contain at least one sample."
        )

    return cleaned_names


def _validate_geometry(
    a: float,
    b: float,
    c: float,
) -> None:
    """Validate ellipsoidal semi-axis values."""

    if a <= 0.0:
        raise ValueError(
            f"a must be positive; got {a}."
        )

    if b <= 0.0:
        raise ValueError(
            f"b must be positive; got {b}."
        )

    if c <= 0.0:
        raise ValueError(
            f"c must be positive; got {c}."
        )


# =====================================================================
# FORWARD MODEL
# =====================================================================

def forward_model_multi_station(
    pmax,
    tpeak,
    d,
    time,
    x_prime,
    y_prime,
    x0_prime,
    y0_prime,
    z,
    a=None,
    b=None,
    c=None,
    s=None,
    a0=None,
    b0=None,
    c0=None,
    nu=0.25,
    h=518.28,
    E=2.0e9,
    theta_deg=0.0,
    alpha=0.8,
    station_names=None,
    debug=False,
):
    """
    Evaluate the analytical multi-station strain response.

    Parameters
    ----------
    pmax, tpeak, d
        Pressure-history parameters.

    time
        Model time array [s].

    x_prime, y_prime
        Station coordinates in the primed/global input frame [m].

    x0_prime, y0_prime
        Inclusion center in the primed/global input frame [m].

    z
        Station vertical coordinates [m].

    a, b, c
        Inclusion semi-axes [m].

    s, a0, b0, c0
        Optional geometry scaling representation. If a, b, c are not
        supplied, all four of these values must be provided.

    nu
        Poisson's ratio.

    h
        Inclusion depth parameter used by the analytical strain model [m].

    E
        Young's modulus [Pa].

    theta_deg
        Inclusion orientation [degrees].

    alpha
        Biot coefficient / coupling parameter.

    station_names
        Station identifiers.

    debug
        Reserved for diagnostic output.

    Returns
    -------
    pandas.DataFrame
        Predicted strain in nanostrain with columns:

            time_s
            eXX_S01 ...
            eYY_S01 ...
            eXY_S01 ...
            eZZ_S01 ...

    Notes
    -----
    Coordinate transformation is delegated to
    ``multi_station_coord_transform.primed_to_unprimed``.

    Tensor rotation is delegated to
    ``multi_station_rotation.rotate_strain_tensor``.

    The analytical strain field is delegated to
    ``multi_station_strain``.
    """

    # ---------------------------------------------------------------
    # Convert inputs to arrays
    # ---------------------------------------------------------------

    x_prime = np.asarray(
        x_prime,
        dtype=float,
    )

    y_prime = np.asarray(
        y_prime,
        dtype=float,
    )

    z = np.asarray(
        z,
        dtype=float,
    )

    time = np.asarray(
        time,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------------

    station_names = _validate_station_inputs(
        x_prime=x_prime,
        y_prime=y_prime,
        z=z,
        time=time,
        station_names=station_names,
    )

    # ---------------------------------------------------------------
    # Geometry handling
    # ---------------------------------------------------------------

    if (
        a is None
        or b is None
        or c is None
    ):

        if (
            s is None
            or a0 is None
            or b0 is None
            or c0 is None
        ):
            raise ValueError(
                "Either provide a, b, c directly or provide "
                "s together with a0, b0, and c0."
            )

        a, b, c = geometry_from_scale(
            s,
            a0,
            b0,
            c0,
        )

    a = float(a)
    b = float(b)
    c = float(c)

    _validate_geometry(
        a,
        b,
        c,
    )

    # ---------------------------------------------------------------
    # Coordinate transformation
    # ---------------------------------------------------------------
    #
    # This is intentionally delegated to the dedicated module.
    # There is only one authoritative coordinate transformation in
    # the production package.
    #

    if not hasattr(
        multi_station_coord_transform,
        "primed_to_unprimed",
    ):
        raise RuntimeError(
            "multi_station_coord_transform must provide "
            "primed_to_unprimed."
        )

    x_arr, y_arr = (
        multi_station_coord_transform.primed_to_unprimed(
            x_prime,
            y_prime,
            x0_prime,
            y0_prime,
            theta_deg,
        )
    )

    x_arr = np.asarray(
        x_arr,
        dtype=float,
    )

    y_arr = np.asarray(
        y_arr,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Pressure history
    # ---------------------------------------------------------------

    pressure = pressure_time_series(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
    )

    # ---------------------------------------------------------------
    # Validate strain module
    # ---------------------------------------------------------------

    strain_module = (
        multi_station_strain
    )

    required_strain_functions = (
        "linear_trans",
        "charac_strain",
        "strain",
    )

    missing_strain_functions = [
        function_name
        for function_name in required_strain_functions
        if not hasattr(
            strain_module,
            function_name,
        )
    ]

    if missing_strain_functions:
        raise RuntimeError(
            "multi_station_strain is missing required "
            f"functions: {missing_strain_functions}"
        )

    # ---------------------------------------------------------------
    # Preserve established strain-module parameter interface
    # ---------------------------------------------------------------

    try:
        setattr(
            strain_module,
            "nu",
            nu,
        )

        setattr(
            strain_module,
            "E",
            E,
        )

        setattr(
            strain_module,
            "alpha",
            alpha,
        )

    except Exception as exc:
        if debug:
            print(
                "[DEBUG] Could not expose parameters "
                f"through strain module: {exc}"
            )

    # ---------------------------------------------------------------
    # Validate rotation module
    # ---------------------------------------------------------------

    if not hasattr(
        multi_station_rotation,
        "rotate_strain_tensor",
    ):
        raise RuntimeError(
            "multi_station_rotation must provide "
            "rotate_strain_tensor."
        )

    rotate_strain_tensor = (
        multi_station_rotation
        .rotate_strain_tensor
    )

    # ---------------------------------------------------------------
    # Allocate output
    # ---------------------------------------------------------------

    n_stations = x_prime.size
    n_times = time.size

    strain_data = np.zeros(
        (
            n_times,
            n_stations * 4,
        ),
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Main forward calculation
    # ---------------------------------------------------------------

    for time_index, pressure_value in enumerate(
        pressure
    ):

        # Linear pressure-to-strain scaling.
        linear_response = (
            strain_module.linear_trans(
                alpha,
                nu,
                pressure_value,
                E,
            )
        )

        characteristic_strain = (
            strain_module.charac_strain(
                linear_response,
                nu,
            )
        )

        exx_values = []
        eyy_values = []
        exy_values = []
        ezz_values = []

        for station_index, (
            x_value,
            y_value,
        ) in enumerate(
            zip(x_arr, y_arr)
        ):

            # -------------------------------------------------------
            # Analytical strain tensor in inclusion coordinates
            # -------------------------------------------------------

            strain_tensor = (
                strain_module.strain(
                    x=x_value,
                    y=y_value,
                    z=z[station_index],
                    a=a,
                    b=b,
                    c=c,
                    ec=characteristic_strain,
                    h=h,
                    nu=nu,
                )
            )

            exx = strain_tensor[0, 0]
            eyy = strain_tensor[1, 1]
            ezz = strain_tensor[2, 2]

            exy = strain_tensor[0, 1]

            # These components are not currently part of the
            # four-channel COMSOL comparison but are included in the
            # tensor rotation operation.
            exz = strain_tensor[0, 2]
            ezy = strain_tensor[1, 2]

            # -------------------------------------------------------
            # Rotate tensor into reported/global coordinates
            # -------------------------------------------------------

            (
                exx_rotated,
                eyy_rotated,
                exy_rotated,
                _exz_rotated,
                _ezy_rotated,
                ezz_rotated,
            ) = rotate_strain_tensor(
                exx,
                eyy,
                exy,
                exz,
                ezy,
                ezz,
                theta_deg,
            )

            # -------------------------------------------------------
            # Convert to nanostrain
            # -------------------------------------------------------

            exx_values.append(
                exx_rotated * 1.0e9
            )

            eyy_values.append(
                eyy_rotated * 1.0e9
            )

            exy_values.append(
                exy_rotated * 1.0e9
            )

            ezz_values.append(
                ezz_rotated * 1.0e9
            )

        # -----------------------------------------------------------
        # Store component-major output
        # -----------------------------------------------------------

        strain_data[
            time_index,
            :n_stations
        ] = np.asarray(
            exx_values,
            dtype=float,
        )

        strain_data[
            time_index,
            n_stations:2 * n_stations
        ] = np.asarray(
            eyy_values,
            dtype=float,
        )

        strain_data[
            time_index,
            2 * n_stations:3 * n_stations
        ] = np.asarray(
            exy_values,
            dtype=float,
        )

        strain_data[
            time_index,
            3 * n_stations:4 * n_stations
        ] = np.asarray(
            ezz_values,
            dtype=float,
        )

    # ---------------------------------------------------------------
    # Build canonical output columns
    # ---------------------------------------------------------------

    output_columns = []

    for component in (
        "eXX",
        "eYY",
        "eXY",
        "eZZ",
    ):

        for station in station_names:

            output_columns.append(
                f"{component}_{station}"
            )

    if len(output_columns) != (
        n_stations * 4
    ):
        raise RuntimeError(
            "Internal output-column count mismatch."
        )

    # ---------------------------------------------------------------
    # Final dataframe
    # ---------------------------------------------------------------

    output = pd.DataFrame(
        np.column_stack(
            [
                time,
                strain_data,
            ]
        ),
        columns=[
            "time_s",
            *output_columns,
        ],
    )

    # ---------------------------------------------------------------
    # Final numerical validation
    # ---------------------------------------------------------------

    if not np.isfinite(
        output.iloc[:, 1:].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Forward model produced NaN or infinite "
            "strain values."
        )

    if debug:
        print(
            "[DEBUG] Forward model completed."
        )
        print(
            f"[DEBUG] Stations: {n_stations}"
        )
        print(
            f"[DEBUG] Time samples: {n_times}"
        )
        print(
            f"[DEBUG] Geometry: "
            f"a={a}, b={b}, c={c}"
        )

    return output


# =====================================================================
# COMPATIBILITY WRAPPER
# =====================================================================

def strain_dataset(
    pmax,
    tpeak,
    d,
    time,
    x,
    y,
    z,
    nu,
    h,
    E,
    theta_deg,
    a=None,
    b=None,
    c=None,
    s=None,
    a0=None,
    b0=None,
    c0=None,
    **kwargs,
):
    """
    Compatibility wrapper around forward_model_multi_station.

    This preserves the historical function interface used by older
    scripts while routing the calculation through the production
    forward model.
    """

    return forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x,
        y_prime=y,
        x0_prime=kwargs.get(
            "x0_prime",
            0.0,
        ),
        y0_prime=kwargs.get(
            "y0_prime",
            0.0,
        ),
        z=z,
        a=a,
        b=b,
        c=c,
        s=s,
        a0=a0,
        b0=b0,
        c0=c0,
        nu=nu,
        h=h,
        E=E,
        theta_deg=theta_deg,
        alpha=kwargs.get(
            "alpha",
            0.8,
        ),
        station_names=kwargs.get(
            "station_names",
            None,
        ),
        debug=kwargs.get(
            "debug",
            False,
        ),
    )