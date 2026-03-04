# forward_model.py
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Local module imports (optional fallbacks)
try:
    import multi_stations_input
except Exception:
    multi_stations_input = None

try:
    import multi_station_coord_transform
except Exception:
    multi_station_coord_transform = None

try:
    import multi_station_strain
except Exception:
    multi_station_strain = None

try:
    import multi_station_rotation
except Exception:
    multi_station_rotation = None


# Read station metadata
stations_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'AVANT_stations.csv'))

# Clean station names: strip spaces/newlines
stations_df["station"] = stations_df["station"].astype(str).str.strip()


def pressure_time_series(pmax, tpeak, d, time):
    t = np.asarray(time)
    p = np.zeros_like(t, dtype=float)
    rising = t <= tpeak
    falling = t > tpeak
    p[rising] = (pmax / tpeak) * t[rising]
    p[falling] = pmax * np.exp(-d * (t[falling] / tpeak - 1))
    return p


def primed_to_unprimed_fallback(x_prime, y_prime, x0_prime, y0_prime, theta_deg):
    theta = np.radians(theta_deg)
    dx = np.asarray(x_prime) - x0_prime
    dy = np.asarray(y_prime) - y0_prime
    x = dx * np.cos(theta) - dy * np.sin(theta)
    y = dx * np.sin(theta) + dy * np.cos(theta)
    return x, y


def forward_model_multi_station(
    pmax, tpeak, d, time,
    x_prime, y_prime,
    x0_prime, y0_prime,
    z, a, b, c,
    nu, h, E, theta_deg,
    alpha=None, station_names=None
):
    """
    Multi-station forward model.
    Returns a pandas DataFrame with Time, Pressure, and one column per component per station.
    """
    # Convert arrays
    x_prime = np.asarray(x_prime, dtype=float)
    y_prime = np.asarray(y_prime, dtype=float)
    z = np.asarray(z, dtype=float)

    Ns = x_prime.size
    Nt = len(time)

    # Ensure station_names is cleaned
    if station_names is None:
        station_names = [f"FS{js+1:02d}" for js in range(Ns)]
    else:
        station_names = [str(s).strip() for s in station_names]
        if len(station_names) != Ns:
            raise ValueError("station_names length must match number of stations")

    # Transform primed -> unprimed
    if multi_station_coord_transform is not None and hasattr(multi_station_coord_transform, 'primed_to_unprimed'):
        x_arr, y_arr = multi_station_coord_transform.primed_to_unprimed(
            x_prime, y_prime, x0_prime, y0_prime, theta_deg
        )
    else:
        x_arr, y_arr = primed_to_unprimed_fallback(
            x_prime, y_prime, x0_prime, y0_prime, theta_deg
        )

    # Pressure
    p_series = pressure_time_series(pmax, tpeak, d, time)

    # Strain function globals
    sf = multi_station_strain
    rotate = multi_station_rotation
    if sf is not None:
        try:
            setattr(sf, "nu", nu)
            setattr(sf, "E", E)
            setattr(sf, "alpha", alpha)
        except Exception:
            pass

    # Prepare storage
    eps_xx = np.zeros((Nt, Ns))
    eps_yy = np.zeros((Nt, Ns))
    eps_zz = np.zeros((Nt, Ns))
    eps_xy = np.zeros((Nt, Ns))
    eps_xz = np.zeros((Nt, Ns))
    eps_yz = np.zeros((Nt, Ns))

    # Compute strains
    for it, p_val in enumerate(p_series):
        if sf is None or not (hasattr(sf, "linear_trans") and hasattr(sf, "charac_strain")):
            raise RuntimeError("strain_function must provide linear_trans and charac_strain")

        lt = sf.linear_trans(alpha, nu, p_val, E)
        ec = sf.charac_strain(lt, nu)

        for js, (xx, yy) in enumerate(zip(x_arr, y_arr)):
            S = sf.strain(
                x=xx, y=yy, z=z[js],
                a=a, b=b, c=c,
                ec=ec,
                h=h,
                nu=nu
            )
            exx, eyy, ezz = S[0,0], S[1,1], S[2,2]
            exy, exz, eyz = S[0,1], S[0,2], S[1,2]

            # Rotate
            if sf is not None and hasattr(sf, "rotate_strain_tensor"):
                exx_p, eyy_p, exy_p, exz_p, eyz_p, ezz_p = sf.rotate_strain_tensor(
                    exx, eyy, exy, exz, eyz, ezz, theta_deg
                )
            elif rotate is not None and hasattr(rotate, "rotate_strain_tensor"):
                exx_p, eyy_p, exy_p, exz_p, eyz_p, ezz_p = rotate.rotate_strain_tensor(
                    exx, eyy, exy, exz, eyz, ezz, theta_deg
                )
            else:
                exx_p, eyy_p, exy_p, exz_p, eyz_p, ezz_p = exx, eyy, exy, exz, eyz, ezz

            # Store nanostrain
            eps_xx[it, js] = exx_p * 1e9
            eps_yy[it, js] = eyy_p * 1e9
            eps_zz[it, js] = ezz_p * 1e9
            eps_xy[it, js] = exy_p * 1e9
            eps_xz[it, js] = exz_p * 1e9
            eps_yz[it, js] = eyz_p * 1e9

    # Build DataFrame
    df = pd.DataFrame({"Time (days)": np.asarray(time), "Pressure (Pa)": p_series})

    comp_names = {
        "Epsilon_XX_nanostrain": eps_xx,
        "Epsilon_YY_nanostrain": eps_yy,
        "Epsilon_ZZ_nanostrain": eps_zz,
        "Epsilon_XY_nanostrain": eps_xy,
        "Epsilon_XZ_nanostrain": eps_xz,
        "Epsilon_YZ_nanostrain": eps_yz,
    }

    # Add columns with cleaned station names
    for comp_base, arr in comp_names.items():
        for js, st_name in enumerate(station_names):
            colname = f"{comp_base}_{st_name}"
            df[colname] = arr[:, js]

    return df


# Convenience wrapper
def strain_dataset(pmax, tpeak, d, time, x, y, z, a, b, c, nu, h, E, theta_deg, **kwargs):
    return forward_model_multi_station(
        pmax, tpeak, d, time, x, y,
        x0_prime=0, y0_prime=0,
        z=z, a=a, b=b, c=c,
        nu=nu, h=h, E=E, theta_deg=theta_deg,
        alpha=kwargs.get("alpha", None),
        station_names=kwargs.get("station_names", None),
    )


def save_dataset_to_excel(dataset, out_path):
    try:
        dataset.to_excel(out_path, index=False, engine="openpyxl")
        return out_path
    except Exception:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        dataset.to_csv(csv_path, index=False)
        print(f"Saved dataset as CSV to '{csv_path}' as fallback.")
        return csv_path


if __name__ == "__main__":
    if multi_stations_input is None or not hasattr(multi_stations_input, "read_input"):
        raise RuntimeError("`multi_stations_input.read_input()` not found")

    params = multi_stations_input.read_input()

    # Clean station names for consistency
    station_names = stations_df["station"].astype(str).str.strip().values

    df_out = forward_model_multi_station(
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        time=params["time"],
        x_prime=params["x_prime"],
        y_prime=params["y_prime"],
        x0_prime=params["x0_prime"],
        y0_prime=params["y0_prime"],
        z=params["z"],
        a=params["a"],
        b=params["b"],
        c=params["c"],
        nu=params["nu"],
        h=params["h"],
        E=params["E"],
        theta_deg=params["theta_deg"],
        alpha=params.get("alpha", None),
        station_names=station_names,
    )

    out_file = os.path.join(os.path.dirname(__file__), "strain_dataset_output.xlsx")
    save_dataset_to_excel(df_out, out_file)
    print(f"[INFO] Saved dataset to {out_file}")
