import numpy as np

def bulk_modulus(E, nu):
    """Bulk modulus K = E / [3(1 - 2ν)]"""
    return E / (3.0 * (1.0 - 2.0 * nu))


def delta_volume_scale(a, b, c, E, nu):
    """
    Effective volume change scaling:
    ΔV ∝ (abc) / K
    """
    K = bulk_modulus(E, nu)
    return a * b * c / K

params = [deltaV_scale, theta_deg, sigma]

# Example: reconstruct an effective pressure source strength
Veff = deltaV_scale * pressure_time_series

def likelihood(params, observed_vector, time, x_prime, y_prime,
               x0_prime, y0_prime, z, nu, pmax, tpeak, d, h, alpha,
               component_cols):

    deltaV_scale, theta_deg, sigma = params

    if sigma <= 0:
        return -np.inf

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
        deltaV_scale=deltaV_scale,
        nu=nu,
        h=h,
        theta_deg=theta_deg,
        alpha=alpha
    )

    predicted_vector = _flatten_df_to_vector(df_pred, component_cols)
    resid = observed_vector - predicted_vector
    n = observed_vector.size

    return -0.5 * (
        np.sum((resid / sigma)**2)
        + n * np.log(2 * np.pi * sigma**2)
    )
# Express the parameters as a scalar multiplication of the other.