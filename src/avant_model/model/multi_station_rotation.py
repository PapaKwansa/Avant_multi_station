import numpy as np

def rotate_strain_tensor(exx, eyy, exy, exz, ezy, ezz, theta_deg):
    """
    Rotates a 2D strain tensor by theta_deg (in degrees) into the primed coordinate system.
    Returns the rotated components: exx', eyy', exy',exz, ezy, ezz' (unchanged).
    """
    theta = np.radians(theta_deg)

    exx_prime = exx * np.cos(theta)**2 + eyy * np.sin(theta)**2 + 2 * exy * np.sin(theta) * np.cos(theta)
    eyy_prime = exx * np.sin(theta)**2 + eyy * np.cos(theta)**2 - 2 * exy * np.sin(theta) * np.cos(theta)
    exy_prime = 0.5 * (eyy - exx) * np.sin(2 * theta) + exy * np.cos(2 * theta)
    exz_prime = exz * np.cos(theta) + ezy * np.sin(theta)
    ezy_prime = ezy * np.cos(theta) - exz * np.sin(theta)
    ezz_prime = ezz  # unchanged in 2D rotation

    return exx_prime, eyy_prime, exy_prime, exz_prime, ezy_prime, ezz_prime