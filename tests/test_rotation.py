"""Regression tests for the AVANT strain-tensor rotation convention."""

import numpy as np

from avant_model.model.multi_station_rotation import rotate_strain_tensor


def test_zero_rotation_is_identity():
    """A zero-degree rotation must leave every strain component unchanged."""

    original = (12.0, -7.0, 3.5, 2.0, -4.0, 8.0)

    rotated = rotate_strain_tensor(
        exx=original[0],
        eyy=original[1],
        exy=original[2],
        exz=original[3],
        ezy=original[4],
        ezz=original[5],
        theta_deg=0.0,
    )

    np.testing.assert_allclose(
        rotated,
        original,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ezz_is_invariant_under_xy_rotation():
    """Rotation about z must leave the zz normal strain unchanged."""

    ezz = 17.25

    for theta in (-180.0, -90.0, -37.0, 0.0, 23.0, 90.0, 180.0):
        rotated = rotate_strain_tensor(
            exx=10.0,
            eyy=-5.0,
            exy=2.0,
            exz=3.0,
            ezy=-4.0,
            ezz=ezz,
            theta_deg=theta,
        )

        assert np.isclose(
            rotated[5],
            ezz,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_xy_trace_is_rotation_invariant():
    """
    The in-plane tensor trace exx + eyy must be invariant under rotation.
    """

    exx = 15.0
    eyy = -4.0
    exy = 6.0

    original_trace = exx + eyy

    for theta in (-90.0, -42.0, 15.0, 63.0, 90.0):
        exx_p, eyy_p, _, _, _, _ = rotate_strain_tensor(
            exx=exx,
            eyy=eyy,
            exy=exy,
            exz=0.0,
            ezy=0.0,
            ezz=9.0,
            theta_deg=theta,
        )

        assert np.isclose(
            exx_p + eyy_p,
            original_trace,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_ninety_degree_rotation_swaps_normal_components():
    """
    At 90 degrees, exx and eyy swap and the in-plane shear changes sign.
    """

    exx = 11.0
    eyy = -3.0
    exy = 4.5

    exx_p, eyy_p, exy_p, _, _, _ = rotate_strain_tensor(
        exx=exx,
        eyy=eyy,
        exy=exy,
        exz=0.0,
        ezy=0.0,
        ezz=0.0,
        theta_deg=90.0,
    )

    np.testing.assert_allclose(
        [exx_p, eyy_p, exy_p],
        [eyy, exx, -exy],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_isotropic_in_plane_strain_is_rotation_invariant():
    """
    Equal in-plane normal strains with zero shear are invariant to rotation.
    """

    for theta in (-135.0, -45.0, 0.0, 27.0, 91.0, 180.0):
        exx_p, eyy_p, exy_p, _, _, _ = rotate_strain_tensor(
            exx=8.0,
            eyy=8.0,
            exy=0.0,
            exz=0.0,
            ezy=0.0,
            ezz=2.0,
            theta_deg=theta,
        )

        np.testing.assert_allclose(
            [exx_p, eyy_p, exy_p],
            [8.0, 8.0, 0.0],
            rtol=0.0,
            atol=1.0e-12,
        )


def test_rotation_preserves_in_plane_tensor_norm():
    """
    exx^2 + eyy^2 + 2*exy^2 is invariant under an orthogonal rotation.
    """

    exx = 14.0
    eyy = -6.0
    exy = 3.25

    original_norm = exx**2 + eyy**2 + 2.0 * exy**2

    for theta in (-73.0, -15.0, 12.0, 48.0, 121.0):
        exx_p, eyy_p, exy_p, _, _, _ = rotate_strain_tensor(
            exx=exx,
            eyy=eyy,
            exy=exy,
            exz=2.0,
            ezy=-1.0,
            ezz=5.0,
            theta_deg=theta,
        )

        rotated_norm = (
            exx_p**2
            + eyy_p**2
            + 2.0 * exy_p**2
        )

        assert np.isclose(
            rotated_norm,
            original_norm,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_xz_yz_vector_norm_is_rotation_invariant():
    """
    The xz/yz pair transforms as a 2-D vector and preserves its magnitude.
    """

    exz = 7.0
    ezy = -2.5

    original_norm = exz**2 + ezy**2

    for theta in (-120.0, -31.0, 0.0, 44.0, 90.0):
        _, _, _, exz_p, ezy_p, _ = rotate_strain_tensor(
            exx=1.0,
            eyy=2.0,
            exy=0.5,
            exz=exz,
            ezy=ezy,
            ezz=3.0,
            theta_deg=theta,
        )

        rotated_norm = exz_p**2 + ezy_p**2

        assert np.isclose(
            rotated_norm,
            original_norm,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_rotation_then_inverse_recovers_original_tensor():
    """
    Applying theta followed by -theta must recover the original tensor.
    """

    original = np.array(
        [13.0, -8.0, 2.75, 4.0, -1.5, 6.0],
        dtype=float,
    )

    theta = 37.0

    rotated = rotate_strain_tensor(
        *original,
        theta_deg=theta,
    )

    recovered = rotate_strain_tensor(
        *rotated,
        theta_deg=-theta,
    )

    np.testing.assert_allclose(
        recovered,
        original,
        rtol=1.0e-12,
        atol=1.0e-12,
    )