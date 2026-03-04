# -------------------------------------------------------------
# bayesian_inversion_multi_station.py
# -------------------------------------------------------------
import numpy as np
from forward_model_multi_station import forward_model_multi_station

# -------------------------
# Flatten DataFrame to vector
# -------------------------
def _flatten_df_to_vector(df, cols):
    return np.concatenate([df[c].values for c in cols])


# -------------------------
# Volume forward model
# -------------------------
def bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))


def volume_forward(a, b, c, E, nu, delta_P):
    V0 = a * b * c
    K = bulk_modulus(E, nu)
    return V0 * (1.0 + delta_P / K)


# -------------------------
# Likelihood function
# -------------------------
def likelihood(
    params,
    observed_vector,
    time, x_prime, y_prime, x0_prime, y0_prime,
    z, nu, pmax, tpeak, d, h, alpha,
    component_cols,
    volume_obs=None,
    delta_P=None,
    sigma_volume=None,
    volume_weight=1.0,
    station_names=None,        # <-- ADDED
):
    """
    Computes log-likelihood combining strain and optional volume constraints.
    """

    a, b, c, E, theta_deg, sigma = params

    # Reject non-physical values
    if sigma <= 0 or E <= 0 or a <= 0 or b <= 0 or c <= 0:
        return -np.inf

    # -------------------------
    # Strain prediction
    # -------------------------
    df_pred = forward_model_multi_station(
        pmax=pmax,
        tpeak=tpeak,
        d=d,
        time=time,
        x_prime=x_prime,
        y_prime=y_prime,
        x0_prime=x0_prime,
        y0_prime=y0_prime,
        z=z,
        a=a,
        b=b,
        c=c,
        nu=nu,
        h=h,
        E=E,
        theta_deg=theta_deg,
        alpha=alpha,
        station_names=station_names,   # <-- ADDED
    )

    predicted_vector = _flatten_df_to_vector(df_pred, component_cols)

    resid = observed_vector - predicted_vector
    n = resid.size

    # Gaussian likelihood for strain
    ll_strain = -0.5 * (
        np.sum((resid / sigma) ** 2) + n * np.log(2 * np.pi * sigma ** 2)
    )

    ll = ll_strain

    # -------------------------
    # Volume constraint (optional)
    # -------------------------
    if volume_obs is not None and delta_P is not None and sigma_volume is not None:
        V_pred = volume_forward(a, b, c, E, nu, delta_P)
        resid_V = volume_obs - V_pred

        ll_vol = -0.5 * volume_weight * (
            np.sum((resid_V / sigma_volume) ** 2) +
            len(resid_V) * np.log(2 * np.pi * sigma_volume ** 2)
        )

        ll += ll_vol

    return ll
