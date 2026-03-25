import os
import numpy as np
import pandas as pd

# Optional local module imports
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
STATION_FILE = os.path.join(os.path.dirname(__file__), "AVANT_stations.csv")
stations_df = pd.read_csv(STATION_FILE)
stations_df["station"] = stations_df["station"].astype(str).str.strip()


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
    nu=0.25, h=518.28, E=2.0e9, theta_deg=-345.0,
    alpha=0.8, station_names=None,
    debug=False,

):
    """
    Multi-station forward model.

    Output layout matches the COMSOL dataset structure:
        [eXX for all stations,
         eYY for all stations,
         eXY for all stations,
         eZZ for all stations]

    For 4 stations, the output columns are:
        eXX_S1, eXX_S2, eXX_S3, eXX_S4,
        eYY_S1, eYY_S2, eYY_S3, eYY_S4,
        eXY_S1, eXY_S2, eXY_S3, eXY_S4,
        eZZ_S1, eZZ_S2, eZZ_S3, eZZ_S4
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

    # Primed -> unprimed coordinates
    if (
        multi_station_coord_transform is not None
        and hasattr(multi_station_coord_transform, "primed_to_unprimed")
    ):
        x_arr, y_arr = multi_station_coord_transform.primed_to_unprimed(
            x_prime, y_prime, x0_prime, y0_prime, theta_deg
        )
    else:
        x_arr, y_arr = primed_to_unprimed_fallback(
            x_prime, y_prime, x0_prime, y0_prime, theta_deg
        )

    # Pressure history
    p_series = pressure_time_series(pmax, tpeak, d, time)
    peak_idx = int(np.argmax(p_series))

    # Prepare strain module
    sf = multi_station_strain
    rotate = multi_station_rotation

    if sf is None or not (hasattr(sf, "linear_trans") and hasattr(sf, "charac_strain")):
        raise RuntimeError(
            "multi_station_strain must provide linear_trans and charac_strain"
        )

    # Set module globals if that is how the strain kernel is written
    try:
        setattr(sf, "nu", nu)
        setattr(sf, "E", E)
        setattr(sf, "alpha", alpha)
    except Exception:
        pass

    # Store 4 observed components per station: eXX, eYY, eXY, eZZ
    strain_data = np.zeros((Nt, Ns * 4), dtype=float)

    for it, p_val in enumerate(p_series):
        lt = sf.linear_trans(alpha, nu, p_val, E)
        ec = sf.charac_strain(lt, nu)

        if debug and it == peak_idx:
            print("\n[DEBUG] --- Peak pressure diagnostics ---")
            print(f"[DEBUG] time = {time[it]:.6g}")
            print(f"[DEBUG] pressure p = {p_val:.6g}")
            print(f"[DEBUG] linear_trans lt = {lt:.6g}")
            print(f"[DEBUG] charac_strain ec = {ec:.6g}")

        exx_list = []
        eyy_list = []
        exy_list = []
        ezz_list = []

        for js, (xx, yy) in enumerate(zip(x_arr, y_arr)):
            S = sf.strain(
                x=xx, y=yy, z=z[js],
                a=a, b=b, c=c,
                ec=ec,
                h=h,
                nu=nu
            )

            if debug and it == peak_idx and js == 0:
                print(f"\n[DEBUG] --- Station {station_names[js]} ---")
                print(f"[DEBUG] location (x,y,z) = ({xx:.3f}, {yy:.3f}, {z[js]:.3f})")
                print(f"[DEBUG] geometry (a,b,c) = ({a:.6g}, {b:.6g}, {c:.6g})")
                print(f"[DEBUG] inclusion depth h = {h:.6g}")
                print(f"[DEBUG] raw strain tensor S:\n{S}")
                print(f"[DEBUG] raw max |S| = {np.max(np.abs(S)):.6g}")

            exx, eyy, ezz = S[0, 0], S[1, 1], S[2, 2]
            exy, exz, eyz = S[0, 1], S[0, 2], S[1, 2]

            # Rotate if available
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

            # Mapping correction from diagnostic:
            # keep rotation, flip eXY sign only
            # Do not switch signs of eXX, eYY, eZZ since that would be a more fundamental issue in the strain kernel

            if debug and it == peak_idx and js == 0:
                print(f"[DEBUG] rotated components:")
                print(f"   exx={exx_p:.6g}, eyy={eyy_p:.6g}, exy={exy_p:.6g}, ezz={ezz_p:.6g}")

            # Convert to nanostrain
            exx_list.append(exx_p * 1e9)
            eyy_list.append(eyy_p * 1e9)
            exy_list.append(exy_p * 1e9)
            ezz_list.append(ezz_p * 1e9)

        # Concatenate in COMSOL order:
        # all eXX, then all eYY, then all eXY, then all eZZ
        strain_data[it, :] = np.concatenate([
            np.asarray(exx_list, dtype=float),
            np.asarray(eyy_list, dtype=float),
            np.asarray(exy_list, dtype=float),
            np.asarray(ezz_list, dtype=float),
        ])

    # Build output DataFrame with explicit component/station names
    df = pd.DataFrame({"time_s": time})

    out_cols = []
    for comp in ["eXX", "eYY", "eXY", "eZZ"]:
        for sname in station_names:
            out_cols.append(f"{comp}_{sname}")

    if len(out_cols) != strain_data.shape[1]:
        raise RuntimeError(
            f"Column count mismatch: expected {len(out_cols)}, got {strain_data.shape[1]}"
        )

    for i, col in enumerate(out_cols):
        df[col] = strain_data[:, i]

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


def save_dataset_to_excel(dataset, out_path):
    """
    Save dataset to Excel or fallback CSV.
    """
    try:
        dataset.to_excel(out_path, index=False, engine="openpyxl")
        return out_path
    except Exception:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        dataset.to_csv(csv_path, index=False)
        print(f"[INFO] Saved dataset as CSV to '{csv_path}' as fallback.")
        return csv_path


if __name__ == "__main__":
    if multi_stations_input is None or not hasattr(multi_stations_input, "read_input"):
        raise RuntimeError("`multi_stations_input.read_input()` not found")

    params = multi_stations_input.read_input()
    station_names = stations_df["station"].astype(str).str.strip().values

    s0 = params.get("s", 1.0)

    # Semi-axes of the lens (half-lengths)
    a0 = params.get("a0", 150.0)   # short axis / 2
    b0 = params.get("b0", 2.5)     # thickness / 2
    c0 = params.get("c0", 290.0)   # long axis / 2

    a, b, c = geometry_from_scale(s0, a0, b0, c0)


    E0 = params.get("E", 2.0e9)
    theta0 = params.get("theta_deg", params.get("theta_deg_start", -345.0))

    df_out = strain_dataset(
        pmax=params["pmax"],
        tpeak=params["tpeak"],
        d=params["d"],
        time=params["time"],
        x=params["x_prime"],
        y=params["y_prime"],
        z=params["z"],
        nu=params["nu"],
        h=params["h"],
        E=E0,
        theta_deg=theta0,
        a=a,
        b=b,
        c=c,
        s=s0,
        a0=a0,
        b0=b0,
        c0=c0,
        alpha=params.get("alpha", None),
        station_names=station_names,
        x0_prime=params.get("x0_prime", 0.0),
        y0_prime=params.get("y0_prime", 0.0),
        debug=True,
    )

    out_file = os.path.join(os.path.dirname(__file__), "strain_dataset_output.xlsx")
    save_dataset_to_excel(df_out, out_file)
    print(f"[INFO] Saved dataset to {out_file}")