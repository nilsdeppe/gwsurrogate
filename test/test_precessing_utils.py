"""
Unit tests for precessing surrogate utility functions:
  - normalize_spin
  - splinterp_many  (batch spline interpolation)
  - _splinterp_Cwrapper / _splinterp_Cwrapper_many (low-level wrappers)
  - eval_coorb_modes (C vs Python path comparison)
"""

import os
import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import normalize_spin, splinterp_many
from gwsurrogate.new.surrogate import _splinterp_Cwrapper, _splinterp_Cwrapper_many


RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# Skip condition for tests requiring the NRSur7dq4 model
# ---------------------------------------------------------------------------

def _model_path():
    import gwsurrogate as gws
    candidate = os.path.join(
        os.path.dirname(gws.__file__),
        "surrogate_downloads",
        "NRSur7dq4.h5",
    )
    return candidate if os.path.isfile(candidate) else None


_MODEL_AVAILABLE = _model_path() is not None

skip_if_no_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="NRSur7dq4.h5 not found",
)


@pytest.fixture(scope="module")
def coorb_test_data():
    """Load NRSur7dq4, run dynamics, return coorb_sur and test inputs."""
    import warnings
    import gwsurrogate as gws

    sur = gws.LoadSurrogate("NRSur7dq4")
    psur = sur._sur_dimless

    q = 2.0
    chiA0 = np.array([0.0, 0.0, 0.5])
    chiB0 = np.array([0.0, 0.0, -0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quat, orbphase, chiA_copr, chiB_copr = \
            psur.get_dynamics(q, chiA0, chiB0)

    from gwsurrogate.new.precessing_surrogate import coorb_spins_from_copr_spins
    from gwsurrogate.new.surrogate import _splinterp_Cwrapper_many

    # Interpolate to coorbital time grid
    t_ds = psur.dynamics_sur.t
    t_coorb = psur.coorb_sur.t
    chiA_copr_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, chiA_copr.T).T
    chiB_copr_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, chiB_copr.T).T
    orbphase_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, orbphase[np.newaxis, :])[0]

    chiA_coorb, chiB_coorb = coorb_spins_from_copr_spins(
        chiA_copr_coorb, chiB_copr_coorb, orbphase_coorb)

    return psur.coorb_sur, q, chiA_coorb, chiB_coorb


@skip_if_no_model
def test_eval_coorb_modes_c_vs_python(coorb_test_data):
    """C eval_coorb_modes matches Python _call_python for NRSur7dq4."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = coorb_sur.ellMax  # 4

    modes_c = coorb_sur._call_c(q, chiA, chiB, ellMax)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)

    np.testing.assert_allclose(modes_c, modes_py, rtol=1e-12, atol=1e-15,
                               err_msg="C and Python coorb modes disagree")


@skip_if_no_model
def test_eval_coorb_modes_partial_ellmax(coorb_test_data):
    """C path with ellMax=2 matches Python path."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = 2

    modes_c = coorb_sur._call_c(q, chiA, chiB, ellMax)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)

    np.testing.assert_allclose(modes_c, modes_py, rtol=1e-12, atol=1e-15,
                               err_msg="Partial ellMax: C vs Python disagree")


@skip_if_no_model
def test_eval_coorb_modes_missing_modes_zero(coorb_test_data):
    """Modes not in mode_list should remain zero."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = coorb_sur.ellMax
    modes = coorb_sur._call_c(q, chiA, chiB, ellMax)

    # Check that modes at indices not corresponding to any mode_list entry
    # are zero (comparing with Python which also zeros them)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)
    # Both should have the same zero pattern
    zero_mask_py = np.all(modes_py == 0, axis=1)
    zero_mask_c = np.all(modes == 0, axis=1)
    np.testing.assert_array_equal(zero_mask_c, zero_mask_py,
                                  err_msg="Zero mode patterns differ")


# ---------------------------------------------------------------------------
# normalize_spin
# ---------------------------------------------------------------------------

def test_normalize_spin_zero_norm_returns_unchanged():
    """chi_norm == 0 → chi returned unmodified."""
    chi = RNG.standard_normal((10, 3))
    chi_orig = chi.copy()
    result = normalize_spin(chi, chi_norm=0.0)
    np.testing.assert_array_equal(result, chi_orig,
                                  err_msg="normalize_spin modified chi when chi_norm=0")


def test_normalize_spin_rescales_rows():
    """Each row of the output has magnitude chi_norm."""
    n = 20
    chi = RNG.standard_normal((n, 3))
    # Make sure no rows are zero
    chi += 0.1
    chi_norm = 0.7

    result = normalize_spin(chi, chi_norm=chi_norm)
    row_norms = np.sqrt(np.sum(result**2, axis=1))
    np.testing.assert_allclose(row_norms, chi_norm, atol=1e-14,
                               err_msg="Row magnitudes after normalize_spin are not chi_norm")


def test_normalize_spin_unit_norm():
    """chi_norm=1.0 → all rows become unit vectors."""
    chi = RNG.standard_normal((15, 3)) + 1.0
    result = normalize_spin(chi, chi_norm=1.0)
    row_norms = np.sqrt(np.sum(result**2, axis=1))
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-14)


def test_normalize_spin_preserves_direction():
    """normalize_spin only rescales; the direction (unit vector) is unchanged."""
    chi = RNG.standard_normal((12, 3)) + 0.5
    chi_norm = 0.4
    result = normalize_spin(chi, chi_norm=chi_norm)
    orig_unit = (chi.T / np.sqrt(np.sum(chi**2, axis=1))).T
    new_unit = (result.T / np.sqrt(np.sum(result**2, axis=1))).T
    np.testing.assert_allclose(new_unit, orig_unit, atol=1e-14,
                               err_msg="normalize_spin changed the direction of chi")


def test_normalize_spin_shape_preserved():
    """Output shape equals input shape."""
    chi = RNG.standard_normal((8, 3))
    result = normalize_spin(chi, chi_norm=0.5)
    assert result.shape == chi.shape


# ---------------------------------------------------------------------------
# splinterp_many  (delegates to _splinterp_Cwrapper_many)
# ---------------------------------------------------------------------------

def _make_smooth_rows(M, t_in):
    """M rows of smooth test data sampled at t_in."""
    return np.array([np.sin((i + 1) * t_in) + 0.3 * np.cos(2 * (i + 1) * t_in)
                     for i in range(M)])


def test_splinterp_many_shape():
    """Output shape is (M, len(t_out))."""
    M, N_in, N_out = 7, 50, 80
    t_in = np.linspace(0.0, 5.0, N_in)
    t_out = np.linspace(0.2, 4.8, N_out)
    data = _make_smooth_rows(M, t_in)
    result = splinterp_many(t_out, t_in, data)
    assert result.shape == (M, N_out), f"Expected ({M},{N_out}), got {result.shape}"


def test_splinterp_many_vs_single_loop():
    """splinterp_many agrees with a loop of _splinterp_Cwrapper row by row."""
    M, N_in = 6, 40
    t_in = np.linspace(0.0, np.pi, N_in)
    t_out = np.linspace(0.1, np.pi - 0.1, 100)
    data = _make_smooth_rows(M, t_in)

    result_many = splinterp_many(t_out, t_in, data)
    result_loop = np.array([_splinterp_Cwrapper(t_out, t_in, data[i])
                            for i in range(M)])

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12,
                               err_msg="splinterp_many disagrees with single-row loop")


def test_splinterp_many_single_row():
    """Single-row splinterp_many matches _splinterp_Cwrapper."""
    t_in = np.linspace(0.0, 2.0, 30)
    t_out = np.linspace(0.1, 1.9, 50)
    y = np.sin(t_in)

    result_many = splinterp_many(t_out, t_in, y[np.newaxis, :])
    result_single = _splinterp_Cwrapper(t_out, t_in, y)

    np.testing.assert_allclose(result_many[0], result_single, rtol=1e-12)


def test_splinterp_many_reproduces_knots():
    """Interpolating at the input knots recovers the original data."""
    M, N = 5, 35
    t_in = np.linspace(0.0, 1.0, N)
    data = _make_smooth_rows(M, t_in)

    result = splinterp_many(t_in, t_in, data)
    np.testing.assert_allclose(result, data, atol=1e-12,
                               err_msg="splinterp_many does not reproduce knot values")


# ---------------------------------------------------------------------------
# _splinterp_Cwrapper_many (low-level)
# ---------------------------------------------------------------------------

def test_splinterp_Cwrapper_many_matches_loop():
    """_splinterp_Cwrapper_many agrees with row-by-row _splinterp_Cwrapper."""
    M, N_in = 9, 45
    t_in = np.linspace(-1.0, 1.0, N_in)
    t_out = np.linspace(-0.9, 0.9, 70)
    data = np.array([np.exp(-(i * 0.5) * t_in**2) for i in range(1, M + 1)])

    result_many = _splinterp_Cwrapper_many(t_out, t_in, data)
    result_loop = np.array([_splinterp_Cwrapper(t_out, t_in, data[i])
                            for i in range(M)])

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12,
                               err_msg="_splinterp_Cwrapper_many disagrees with loop")


def test_splinterp_Cwrapper_many_dtype():
    """Output dtype is float64 for real input."""
    t_in = np.linspace(0.0, 1.0, 20)
    t_out = np.linspace(0.1, 0.9, 30)
    data = np.ones((4, 20), dtype=np.float32)
    result = _splinterp_Cwrapper_many(t_out, t_in, data)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
