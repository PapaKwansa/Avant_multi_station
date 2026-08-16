import numpy as np

def primed_to_unprimed(x_prime, y_prime, x0_prime, y0_prime, theta_deg):
    """
    Transform coordinates from primed (x', y') to unprimed (x, y)
    using translation and rotation. Fully vectorized for multi-station use.

    Parameters
    ----------
    x_prime : float or array_like
        Station x' coordinates.
    y_prime : float or array_like
        Station y' coordinates.
    x0_prime : float
        Inclusion center x' coordinate.
    y0_prime : float
        Inclusion center y' coordinate.
    theta_deg : float
        Rotation angle (degrees).

    Returns
    -------
    x : ndarray
        Unprimed x-coordinates (vectorized).
    y : ndarray
        Unprimed y-coordinates (vectorized).
    """
    # Convert both scalars or lists to arrays
    x_prime = np.asarray(x_prime, dtype=float)
    y_prime = np.asarray(y_prime, dtype=float)

    # Shift first (vectorized)
    dx = x_prime - x0_prime
    dy = y_prime - y0_prime

    # Rotation
    theta = np.radians(theta_deg)
    cosθ = np.cos(theta)
    sinθ = np.sin(theta)

    x = dx * cosθ - dy * sinθ
    y = dx * sinθ + dy * cosθ

    return x, y
