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
    normalize_spin(chi, chi_norm=0.0)
    np.testing.assert_array_equal(chi, chi_orig,
                                  err_msg="normalize_spin modified chi when chi_norm=0")


def test_normalize_spin_rescales_rows():
    """Each row of the output has magnitude chi_norm."""
    n = 20
    chi = RNG.standard_normal((n, 3))
    # Make sure no rows are zero
    chi += 0.1
    chi_norm = 0.7

    normalize_spin(chi, chi_norm=chi_norm)
    row_norms = np.sqrt(np.sum(chi**2, axis=1))
    np.testing.assert_allclose(row_norms, chi_norm, atol=1e-14,
                               err_msg="Row magnitudes after normalize_spin are not chi_norm")


def test_normalize_spin_unit_norm():
    """chi_norm=1.0 → all rows become unit vectors."""
    chi = RNG.standard_normal((15, 3)) + 1.0
    normalize_spin(chi, chi_norm=1.0)
    row_norms = np.sqrt(np.sum(chi**2, axis=1))
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-14)


def test_normalize_spin_preserves_direction():
    """normalize_spin only rescales; the direction (unit vector) is unchanged."""
    chi = RNG.standard_normal((12, 3)) + 0.5
    original_chi = np.copy(chi)
    chi_norm = 0.4
    normalize_spin(chi, chi_norm=chi_norm)
    orig_unit = (original_chi.T / np.sqrt(np.sum(original_chi**2, axis=1))).T
    new_unit = (chi.T / np.sqrt(np.sum(chi**2, axis=1))).T
    np.testing.assert_allclose(new_unit, orig_unit, atol=1e-14,
                               err_msg="normalize_spin changed the direction of chi")


def test_normalize_spin_shape_preserved():
    """Output shape equals input shape."""
    chi = RNG.standard_normal((8, 3))
    original_chi = np.copy(chi)
    normalize_spin(chi, chi_norm=0.5)
    assert original_chi.shape == chi.shape


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


# ---------------------------------------------------------------------------
# Tests for DynamicsSurrogate: eval_fit_batch_dydt, get_time_deriv_from_index,
# get_fit_params, get_ds_fit_x
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dynamics_test_data():
    """Load NRSur7dq4, return DynamicsSurrogate and a valid state vector."""
    import warnings
    import gwsurrogate as gws

    sur = gws.LoadSurrogate("NRSur7dq4")
    psur = sur._sur_dimless

    q = 2.0
    chiA0 = np.array([0.0, 0.0, 0.5])
    chiB0 = np.array([0.0, 0.0, -0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quat, orbphase, chiA_copr, chiB_copr, _ = \
            psur.dynamics_sur(q, chiA0, chiB0, init_orbphase=0.0,
                              init_quat=None, t_ref=psur.dynamics_sur.t[0])

    # Pick a valid state at index 100
    i0 = 100
    y = np.empty(11)
    y[:4] = quat[:, i0]
    y[4] = orbphase[i0]
    y[5:8] = chiA_copr[i0]
    y[8:11] = chiB_copr[i0]

    return psur.dynamics_sur, q, y, i0


@skip_if_no_model
def test_eval_fit_batch_dydt_matches_individual(dynamics_test_data):
    """eval_fit_batch_dydt matches individual eval_fit + assemble_dydt."""
    from gwsurrogate.precessing_utils import _utils

    ds, q, y, i0 = dynamics_test_data

    # Get fit_params via the Python path
    x = _utils.get_ds_fit_x(y, q)
    fit_params = ds._get_fit_params(x.copy())

    q_fit_offset, q_fit_slope, q_max_bfOrder, chi_max_bfOrder = ds._fit_settings

    # Batched C path
    dydt_batch = _utils.eval_fit_batch_dydt(
        ds.fit_data_batch[i0], fit_params.copy(), y,
        q_fit_offset, q_fit_slope, q_max_bfOrder, chi_max_bfOrder)

    # Individual path: evaluate 9 fits separately
    fd = ds.fit_data[i0]
    from gwsurrogate.new.precessing_surrogate import _eval_scalar_fit
    ooxy = np.array([
        _eval_scalar_fit(fd['omega_orb'][0], fit_params.copy(), ds._fit_settings),
        _eval_scalar_fit(fd['omega_orb'][1], fit_params.copy(), ds._fit_settings),
    ])
    omega = _eval_scalar_fit(fd['omega'], fit_params.copy(), ds._fit_settings)
    cAdot = np.array([
        _eval_scalar_fit(fd['chiA'][j], fit_params.copy(), ds._fit_settings)
        for j in range(3)
    ])
    cBdot = np.array([
        _eval_scalar_fit(fd['chiB'][j], fit_params.copy(), ds._fit_settings)
        for j in range(3)
    ])
    dydt_indiv = _utils.assemble_dydt(y, ooxy, omega, cAdot, cBdot)

    np.testing.assert_allclose(dydt_batch, dydt_indiv, rtol=1e-12, atol=1e-15,
                               err_msg="Batch dydt differs from individual fits")


@skip_if_no_model
def test_get_time_deriv_from_index_deterministic(dynamics_test_data):
    """get_time_deriv_from_index returns identical results on repeated calls."""
    ds, q, y, i0 = dynamics_test_data

    dydt1 = ds.get_time_deriv_from_index(i0, q, y)
    dydt2 = ds.get_time_deriv_from_index(i0, q, y)

    np.testing.assert_array_equal(dydt1, dydt2,
                                  err_msg="get_time_deriv_from_index not deterministic")


@skip_if_no_model
def test_get_fit_params_transform(dynamics_test_data):
    """NRSur7dq4 get_fit_params correctly transforms x."""
    ds, q, y, _ = dynamics_test_data

    x = np.array([q, 0.1, 0.2, 0.5, -0.1, 0.3, -0.3])
    x_orig = x.copy()
    result = ds._get_fit_params(x)

    # x[0] should be log(q)
    np.testing.assert_allclose(result[0], np.log(q), rtol=1e-14,
                               err_msg="x[0] should be log(q)")

    # x[6] should be (chi1z - chi2z)/2
    chi1z, chi2z = x_orig[3], x_orig[6]
    np.testing.assert_allclose(result[6], (chi1z - chi2z) * 0.5, rtol=1e-14,
                               err_msg="x[6] should be (chi1z-chi2z)/2")

    # x[3] should be chiHat
    eta = q / (1.0 + q)**2
    chi_wtAvg = q / (1.0 + q) * chi1z + 1.0 / (1.0 + q) * chi2z
    chiHat = (chi_wtAvg - 38.0 * eta / 113.0 * (chi1z + chi2z)) / \
             (1.0 - 76.0 * eta / 113.0)
    np.testing.assert_allclose(result[3], chiHat, rtol=1e-14,
                               err_msg="x[3] should be chiHat")

    # x[1], x[2], x[4], x[5] should be unchanged
    np.testing.assert_array_equal(result[1], x_orig[1])
    np.testing.assert_array_equal(result[2], x_orig[2])
    np.testing.assert_array_equal(result[4], x_orig[4])
    np.testing.assert_array_equal(result[5], x_orig[5])


@skip_if_no_model
def test_get_ds_fit_x_rotation(dynamics_test_data):
    """get_ds_fit_x rotates spins from coprecessing to coorbital frame."""
    from gwsurrogate.precessing_utils import _utils

    ds, q, y, _ = dynamics_test_data

    x = _utils.get_ds_fit_x(y, q)

    # x should have shape (7,)
    assert x.shape == (7,), f"Expected shape (7,), got {x.shape}"

    # x[0] should be q
    np.testing.assert_allclose(x[0], q, rtol=1e-14)

    # z-components of spins are not rotated by orbphase
    np.testing.assert_allclose(x[3], y[7], rtol=1e-14,
                               err_msg="chi1z should equal y[7]")
    np.testing.assert_allclose(x[6], y[10], rtol=1e-14,
                               err_msg="chi2z should equal y[10]")
