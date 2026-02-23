"""
Unit tests for _wignerD_matrices and optimized variants.

Uses analytic properties of Wigner D matrices:
  - Identity quaternion → identity matrices
  - Unitarity: D @ D†  = I
  - Shape / dtype
  - Agreement with the old Python reference implementation (extracted from
    the dead code in precessing_surrogate.py, lines ~77-124)
  - Cross-validation between all three C implementations
"""

import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import (
    _wignerD_matrices,
    _wignerD_matrices_opt,
    _wignerD_matrices_opt_hc4,
    _assemble_powers,
)
from gwsurrogate.precessing_utils import _utils


# ---------------------------------------------------------------------------
# Python reference implementation
# (Extracted verbatim from the dead code in precessing_surrogate.py)
# ---------------------------------------------------------------------------

def _wignerD_matrices_python(q, ellMax):
    """Pure-Python reference: the original implementation before C migration."""
    ra = q[0] + 1.j * q[3]
    rb = q[2] + 1.j * q[1]
    ra_small = (abs(ra) < 1.e-12)
    rb_small = (abs(rb) < 1.e-12)
    i1 = np.where((1 - ra_small) * (1 - rb_small))[0]
    i2 = np.where(ra_small)[0]
    i3 = np.where((1 - ra_small) * rb_small)[0]

    n = len(ra)
    lvals = range(2, ellMax + 1)
    matrices = [0.j * np.zeros((2 * ell + 1, 2 * ell + 1, n)) for ell in lvals]

    for i, ell in enumerate(lvals):
        for m in range(-ell, ell + 1):
            if (ell + m) % 2 == 1:
                matrices[i][ell + m, ell - m, i2] = rb[i2] ** (2 * m)
            else:
                matrices[i][ell + m, ell - m, i2] = -1 * rb[i2] ** (2 * m)
            matrices[i][ell + m, ell + m, i3] = ra[i3] ** (2 * m)

    ra_i1 = ra[i1]
    rb_i1 = rb[i1]
    ra_pows = _assemble_powers(ra_i1, range(-2 * ellMax, 2 * ellMax + 1))
    rb_pows = _assemble_powers(rb_i1, range(-2 * ellMax, 2 * ellMax + 1))
    abs_raSqr_pows = _assemble_powers(abs(ra_i1) ** 2, range(0, 2 * ellMax + 1))
    absRRatioSquared = (abs(rb_i1) / abs(ra_i1)) ** 2
    ratio_pows = _assemble_powers(absRRatioSquared, range(0, 2 * ellMax + 1))

    for i, ell in enumerate(lvals):
        for m in range(-ell, ell + 1):
            for mp in range(-ell, ell + 1):
                factor = _utils.wigner_coef(ell, mp, m)
                factor *= ra_pows[2 * ellMax + m + mp]
                factor *= rb_pows[2 * ellMax + m - mp]
                factor *= abs_raSqr_pows[ell - m]
                rhoMin = max(0, mp - m)
                rhoMax = min(ell + mp, ell - m)
                s = 0.
                for rho in range(rhoMin, rhoMax + 1):
                    c = ((-1) ** rho) * (_utils.binom(ell + mp, rho) *
                                         _utils.binom(ell - mp, ell - rho - m))
                    s += c * ratio_pows[rho]
                matrices[i][ell + m, ell + mp, i1] = factor * s

    return matrices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)


def _random_unit_quaternions(n):
    """Return (4, n) array of random unit quaternions."""
    q = RNG.standard_normal((4, n))
    q /= np.linalg.norm(q, axis=0)
    return q


def _identity_quaternion(n=1):
    """Return (4, n) identity quaternion(s)."""
    q = np.zeros((4, n))
    q[0] = 1.0
    return q


# All three implementations to parametrize over
ALL_IMPLS = [_wignerD_matrices, _wignerD_matrices_opt, _wignerD_matrices_opt_hc4]
IMPL_IDS = ["original", "opt", "opt_hc4"]


# ---------------------------------------------------------------------------
# Tests parametrized over all implementations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_shape(impl):
    """Output is a list of length (ellMax-1) with correct shapes and dtype."""
    ellMax = 4
    N = 7
    q = _random_unit_quaternions(N)
    mats = impl(q, ellMax)
    assert len(mats) == ellMax - 1, f"Expected {ellMax - 1} matrices, got {len(mats)}"
    for i, ell in enumerate(range(2, ellMax + 1)):
        expected_shape = (2 * ell + 1, 2 * ell + 1, N)
        assert mats[i].shape == expected_shape, (
            f"ell={ell}: expected shape {expected_shape}, got {mats[i].shape}"
        )
        assert mats[i].dtype == np.complex128, (
            f"ell={ell}: expected complex128, got {mats[i].dtype}"
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_identity_quaternion(impl):
    """Identity quaternion → identity matrix for every ell."""
    ellMax = 4
    q = _identity_quaternion(n=1)
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        D = mats[i][:, :, 0]
        size = 2 * ell + 1
        np.testing.assert_allclose(
            D, np.eye(size, dtype=complex), atol=1e-12,
            err_msg=f"ell={ell}: D(identity quat) is not the identity matrix",
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_unitarity(impl):
    """D^ell @ D^ell† = I for random unit quaternions."""
    ellMax = 4
    N = 10
    q = _random_unit_quaternions(N)
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        for k in range(N):
            D = mats[i][:, :, k]
            product = D @ D.conj().T
            np.testing.assert_allclose(
                product, np.eye(size, dtype=complex), atol=1e-11,
                err_msg=f"ell={ell}, quat #{k}: D @ D† ≠ I",
            )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_matches_python_reference(impl):
    """C implementation matches the old Python reference for generic quaternions."""
    ellMax = 4
    N = 8
    q = _random_unit_quaternions(50)
    ra = q[0] + 1j * q[3]
    rb = q[2] + 1j * q[1]
    mask = (np.abs(ra) > 0.1) & (np.abs(rb) > 0.1)
    q_generic = q[:, mask][:, :N]
    assert q_generic.shape[1] == N, "Not enough generic quaternions — adjust seed"

    mats_c = impl(q_generic, ellMax)
    mats_py = _wignerD_matrices_python(q_generic, ellMax)

    for i, ell in enumerate(range(2, ellMax + 1)):
        np.testing.assert_allclose(
            mats_c[i], mats_py[i], atol=1e-12,
            err_msg=f"ell={ell}: C result disagrees with Python reference",
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_edge_case_ra_small(impl):
    """Quaternions where ra ≈ 0 — C result must still be unitary."""
    ellMax = 3
    q = np.array([[0.0], [1.0 / np.sqrt(2)], [1.0 / np.sqrt(2)], [0.0]])
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        D = mats[i][:, :, 0]
        np.testing.assert_allclose(
            D @ D.conj().T, np.eye(size, dtype=complex), atol=1e-11,
            err_msg=f"ell={ell}: unitarity fails for ra≈0 quaternion",
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_edge_case_rb_small(impl):
    """Quaternions where rb ≈ 0 — C result must still be unitary."""
    ellMax = 3
    q = np.array([[1.0 / np.sqrt(2)], [0.0], [0.0], [1.0 / np.sqrt(2)]])
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        D = mats[i][:, :, 0]
        np.testing.assert_allclose(
            D @ D.conj().T, np.eye(size, dtype=complex), atol=1e-11,
            err_msg=f"ell={ell}: unitarity fails for rb≈0 quaternion",
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_multiple_identity_quaternions(impl):
    """Batch of N identity quaternions all give identity matrices."""
    ellMax = 3
    N = 5
    q = _identity_quaternion(n=N)
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        for k in range(N):
            np.testing.assert_allclose(
                mats[i][:, :, k], np.eye(size, dtype=complex), atol=1e-12,
                err_msg=f"ell={ell}, quat #{k}: not identity",
            )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_invalid_ellmax(impl):
    """ellMax < 2 should raise ValueError."""
    q = _identity_quaternion()
    with pytest.raises(ValueError):
        impl(q, ellMax=1)


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_invalid_q_shape(impl):
    """q with wrong shape should raise ValueError."""
    with pytest.raises(ValueError):
        impl(np.ones((3, 5)), ellMax=2)


# ---------------------------------------------------------------------------
# Cross-validation: all 3 implementations agree
# ---------------------------------------------------------------------------

def test_wignerD_cross_validation_ellmax4():
    """All three implementations agree to atol=1e-13 for random quaternions, ellMax=4."""
    ellMax = 4
    N = 20
    q = _random_unit_quaternions(N)

    mats_orig = _wignerD_matrices(q, ellMax)
    mats_opt = _wignerD_matrices_opt(q, ellMax)
    mats_hc4 = _wignerD_matrices_opt_hc4(q, ellMax)

    for i, ell in enumerate(range(2, ellMax + 1)):
        np.testing.assert_allclose(
            mats_opt[i], mats_orig[i], atol=1e-13,
            err_msg=f"ell={ell}: _opt disagrees with original",
        )
        np.testing.assert_allclose(
            mats_hc4[i], mats_orig[i], atol=1e-13,
            err_msg=f"ell={ell}: _opt_hc4 disagrees with original",
        )


def test_wignerD_cross_validation_ellmax5():
    """ellMax=5: verifies _hc4 fallback path for ell>4."""
    ellMax = 5
    N = 15
    q = _random_unit_quaternions(N)

    mats_orig = _wignerD_matrices(q, ellMax)
    mats_opt = _wignerD_matrices_opt(q, ellMax)
    mats_hc4 = _wignerD_matrices_opt_hc4(q, ellMax)

    for i, ell in enumerate(range(2, ellMax + 1)):
        np.testing.assert_allclose(
            mats_opt[i], mats_orig[i], atol=1e-12,
            err_msg=f"ell={ell}: _opt disagrees with original",
        )
        np.testing.assert_allclose(
            mats_hc4[i], mats_orig[i], atol=1e-12,
            err_msg=f"ell={ell}: _opt_hc4 disagrees with original",
        )


def test_wignerD_cross_validation_ellmax8():
    """ellMax=8: tests recurrence for higher ell values."""
    ellMax = 8
    N = 10
    q = _random_unit_quaternions(N)

    mats_orig = _wignerD_matrices(q, ellMax)
    mats_opt = _wignerD_matrices_opt(q, ellMax)
    mats_hc4 = _wignerD_matrices_opt_hc4(q, ellMax)

    for i, ell in enumerate(range(2, ellMax + 1)):
        np.testing.assert_allclose(
            mats_opt[i], mats_orig[i], atol=1e-11,
            err_msg=f"ell={ell}: _opt disagrees with original",
        )
        np.testing.assert_allclose(
            mats_hc4[i], mats_orig[i], atol=1e-11,
            err_msg=f"ell={ell}: _opt_hc4 disagrees with original",
        )


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_wignerD_unitarity_ellmax8(impl):
    """Unitarity at ellMax=8 for all implementations."""
    ellMax = 8
    N = 5
    q = _random_unit_quaternions(N)
    mats = impl(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        for k in range(N):
            D = mats[i][:, :, k]
            product = D @ D.conj().T
            np.testing.assert_allclose(
                product, np.eye(size, dtype=complex), atol=1e-10,
                err_msg=f"ell={ell}, quat #{k}: D @ D† ≠ I",
            )
