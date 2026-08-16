import numpy as np
import pandas as pd

from . import multi_station_coord_transform
from . import multi_station_strain
from . import multi_station_rotation

def pressure_time_series(pmax, tpeak, d, time):
    """
    Piecewise pressure history:
      - linear ramp to tpeak
      - exponential decay after tpeak
    """
    t = np.asarray(time, dtype=float)
    p = np.zeros_like(t, dtype=float)

    rising = t <= tpeak
    falling = t > tpeak

    p[rising] = (pmax / tpeak) * t[rising]
    p[falling] = pmax * np.exp(-d * (t[falling] / tpeak - 1.0))

    return p


def primed_to_unprimed_fallback(x_prime, y_prime, x0_prime, y0_prime, theta_deg):
    """
    Fallback coordinate transform if multi_station_coord_transform is unavailable.
    """
    theta = np.radians(theta_deg)
    dx = np.asarray(x_prime, dtype=float) - x0_prime
    dy = np.asarray(y_prime, dtype=float) - y0_prime

    x = dx * np.cos(theta) - dy * np.sin(theta)
    y = dx * np.sin(theta) + dy * np.cos(theta)
    return x, y


def geometry_from_scale(s, a0, b0, c0):
    """
    Convert dimensionless geometry ratios into semi-axes.
    """
    a = s * a0
    b = s * b0
    c = s * c0
    return a, b, c


def forward_model_multi_station(
    pmax, tpeak, d, time,
    x_prime, y_prime,
    x0_prime, y0_prime,
    z,
    a=None, b=None, c=None,
    s=None, a0=None, b0=None, c0=None,
    nu=0.25, h=518.28, E=2.0e9, theta_deg=0.0,
    alpha=0.8, station_names=None,
    debug=False,
):
    """
    Multi-station forward model aligned with COMSOL coordinates.
    """

    x_prime = np.asarray(x_prime, dtype=float)
    y_prime = np.asarray(y_prime, dtype=float)
    z = np.asarray(z, dtype=float)
    time = np.asarray(time, dtype=float)

    Ns = x_prime.size
    Nt = len(time)

    if station_names is None:
        station_names = [f"S{js + 1}" for js in range(Ns)]
    else:
        station_names = [str(sname).strip() for sname in station_names]
        if len(station_names) != Ns:
            raise ValueError("station_names length must match number of stations")

    # If a,b,c are not supplied, compute them from s and ratios
    if a is None or b is None or c is None:
        if s is None or a0 is None or b0 is None or c0 is None:
            raise ValueError(
                "Either provide a,b,c directly or provide s together with a0,b0,c0."
            )
        a, b, c = geometry_from_scale(s, a0, b0, c0)

    # Relative to inclusion center (global COMSOL coordinates)
    x_rel = x_prime - x0_prime
    y_rel = y_prime - y0_prime

    # Rotate relative coordinates by -theta_deg to move into inclusion frame
    # (COMSOL tilt is clockwise; here positive theta_deg is clockwise)
    theta = np.radians(theta_deg)
    x_arr = x_rel * np.cos(theta) - y_rel * np.sin(theta)
    y_arr = x_rel * np.sin(theta) + y_rel * np.cos(theta)

    # Pressure history
    p_series = pressure_time_series(pmax, tpeak, d, time)

    # Prepare strain module
    sf = multi_station_strain
    rot_mod = multi_station_rotation

    if sf is None or not (hasattr(sf, "linear_trans") and hasattr(sf, "charac_strain")):
        raise RuntimeError(
            "multi_station_strain must provide linear_trans and charac_strain"
        )

    # Optional: expose parameters to strain module
    try:
        setattr(sf, "nu", nu)
        setattr(sf, "E", E)
        setattr(sf, "alpha", alpha)
    except Exception:
        pass

    # Check for rotation function (to rotate strain tensor back to global)
    rotate_strain_tensor = None
    if rot_mod is not None and hasattr(rot_mod, "rotate_strain_tensor"):
        rotate_strain_tensor = rot_mod.rotate_strain_tensor

    strain_data = np.zeros((Nt, Ns * 4), dtype=float)

    for it, p_val in enumerate(p_series):
        lt = sf.linear_trans(alpha, nu, p_val, E)
        ec = sf.charac_strain(lt, nu)

        exx_list = []
        eyy_list = []
        exy_list = []
        ezz_list = []

        for js, (xx, yy) in enumerate(zip(x_arr, y_arr)):
            # Strain in inclusion principal frame
            S = sf.strain(
                x=xx, y=yy, z=z[js],
                a=a, b=b, c=c,
                ec=ec,
                h=h,
                nu=nu
            )

            exx_i = S[0, 0]
            eyy_i = S[1, 1]
            ezz_i = S[2, 2]
            exy_i = S[0, 1]
            exz_i = 0.0
            ezy_i = 0.0

            if rotate_strain_tensor is not None:
                # Rotate strain tensor from inclusion frame back to global frame
                exx, eyy, exy, exz, ezy, ezz = rotate_strain_tensor(
                    exx_i, eyy_i, exy_i, exz_i, ezy_i, ezz_i, theta_deg
                )
            else:
                # Fallback: assume inclusion frame ≈ global frame
                exx, eyy, exy, ezz = exx_i, eyy_i, exy_i, ezz_i

            exx_list.append(exx * 1e9)
            eyy_list.append(eyy * 1e9)
            exy_list.append(exy * 1e9)
            ezz_list.append(ezz * 1e9)

        strain_data[it, :] = np.concatenate([
            np.asarray(exx_list, dtype=float),
            np.asarray(eyy_list, dtype=float),
            np.asarray(exy_list, dtype=float),
            np.asarray(ezz_list, dtype=float),
        ])

    # Build column names
    out_cols = []
    for comp in ["eXX", "eYY", "eXY", "eZZ"]:
        for sname in station_names:
            out_cols.append(f"{comp}_{sname}")

    if len(out_cols) != strain_data.shape[1]:
        raise RuntimeError(
            f"Column count mismatch: expected {len(out_cols)}, got {strain_data.shape[1]}"
        )

    df = pd.DataFrame(
        np.column_stack([time, strain_data]),
        columns=["time_s"] + out_cols
    )

    return df


def strain_dataset(
    pmax, tpeak, d, time,
    x, y, z,
    nu, h, E, theta_deg,
    a=None, b=None, c=None,
    s=None, a0=None, b0=None, c0=None,
    **kwargs
):
    """
    Wrapper for forward_model_multi_station.
    """
    return forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x,
        y_prime=y,
        x0_prime=kwargs.get("x0_prime", 0.0),
        y0_prime=kwargs.get("y0_prime", 0.0),
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
        alpha=kwargs.get("alpha", None),
        station_names=kwargs.get("station_names", None),
        debug=kwargs.get("debug", False),
    )
