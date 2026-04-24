#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>

// entire block can be removed if building against numpy 1.X is dropped
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#if defined(__has_include)
#  if __has_include(<numpy/npy_2_compat.h>)
#    include <numpy/npy_2_compat.h>    /* NumPy≥2.0 shim */
#  else
     /* NumPy<2.0: stub so PyArray_ImportNumPyAPI() just calls import_array() */
#    define PyArray_ImportNumPyAPI() (import_array(), 0)
#  endif
#else
#  include <numpy/npy_2_compat.h>      // assume its there
#endif
// END numpy 1.X build support 

#include "precessing_utils.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Created in 2019 by Vijay Varma, Jonathan Blackman
 *
 * This is a utilities Python module for use by gwsurrogate that performs
 * tasks that would be slow in Python.
 *
 * You are free to redistribute and/or modify this software as needed.
 *
 * For help with C extensions for Python including NumPy, see:
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * The NumPy C-API can be found at:
 * https://docs.scipy.org/doc/numpy/reference/c-api.html
 */

/*
 * Initialize the module in a way compatible with either python 2 or 3. See:
 * https://docs.python.org/3/howto/cporting.html
 */
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif


/* Forward declarations */
static PyObject *py_rotate_waveform(PyObject *self, PyObject *args);
static PyObject *eval_coorb_modes(PyObject *self, PyObject *args);
static PyObject *py_lagrange4_interp(PyObject *self, PyObject *args);
static PyObject *py_compute_dydt_interp(PyObject *self, PyObject *args);
static PyObject *rk4_step_forward(PyObject *self, PyObject *args);
static PyObject *rk4_step_backward(PyObject *self, PyObject *args);

/* ==== Setup the python methods table === */
static PyMethodDef _utils_methods[] = {
    {"eval_fit", eval_fit, METH_VARARGS},
    {"eval_fit_batch", eval_fit_batch, METH_VARARGS},
    {"eval_fit_batch_dydt", eval_fit_batch_dydt, METH_VARARGS},
    {"normalize_y", normalize_y, METH_VARARGS},
    {"get_ds_fit_x", get_ds_fit_x, METH_VARARGS},
    {"assemble_dydt", assemble_dydt, METH_VARARGS},
    {"ab4_dy", ab4_dy, METH_VARARGS},
    {"binom", binom, METH_VARARGS},
    {"wigner_coef", wigner_coef, METH_VARARGS},
    {"wignerD_matrices", py_wignerD_matrices, METH_VARARGS},
    {"rotate_waveform", py_rotate_waveform, METH_VARARGS},
    {"eval_coorb_modes", eval_coorb_modes, METH_VARARGS},
    {"integrate_ab4_forward", integrate_ab4_forward, METH_VARARGS},
    {"integrate_ab4_backward", integrate_ab4_backward, METH_VARARGS},
    {"_lagrange4_interp", py_lagrange4_interp, METH_VARARGS},
    {"_compute_dydt_interp", py_compute_dydt_interp, METH_VARARGS},
    {"rk4_step_forward", rk4_step_forward, METH_VARARGS},
    {"rk4_step_backward", rk4_step_backward, METH_VARARGS},
    {NULL, NULL} /* Marks the end of this structure */
};

#if PY_MAJOR_VERSION >= 3

static int _utils_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _utils_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_utils",
        NULL,
        sizeof(struct module_state),
        _utils_methods,
        NULL,
        _utils_traverse,
        _utils_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit__utils(void)

#else

#define INITERROR return

void init_utils(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_utils", _utils_methods);
#endif
    if (!module)
        INITERROR;

    /* Import NumPy C-API (works on both 1.x and 2.x) */
    if (PyArray_ImportNumPyAPI() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    //fprintf(stderr,"Build log: Precessing_utils built against NumPy %s\n",NPY_FEATURE_VERSION_STRING);

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_utils.Error", NULL, NULL);
    if (!st->error) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

// Done initializing module. Actual functions start here.

double ipow(double base, long exponent) {
    if (exponent == 0) return 1.0;
    double res = base;
    while (exponent > 1) {
        res = res*base;
        exponent -= 1;
    }
    return res;
}

/*
 * This function evaluates a parametric fit.
 * Arguments (with python data types):
 *      bf_orders:  A 2d integer numpy array with shape (n_coefs, 7).
 *                  Gives the basis function orders for each coefficient
 *                  and each parameter.
 *      coefs:      A 1d float numpy array with length n_coefs.
 *                  Contains the fit coefficients.
 *      x: i        A 1d float numpy array with length 7.
 *                  Gives the parameters at which the fit should be evaluated.
 *      q_fit_offset:  A double. Gives the offset for linear transformation
 *                     from q parameter to [-1,1].
 *      q_fit_slope:  A double. Gives the slope for linear transformation
 *                     from q parameter to [-1,1].
 *      q_max_bfOrder: Max basis function order for the q parameter.
 *      chi_max_bfOrder: Max basis function order for the chi parameters.
 *
 * Computes the fit evaluation by summing up coefficients multiplied by 7 basis
 * functions, each evaluated at one component of x.
 * Returns a python float giving the fit evaluation.
 */
static PyObject *eval_fit(PyObject *self, PyObject *args) {

    PyArrayObject *bf_orders, *coefs, *x;
    int i, j, n;
    long *bf_order_data, *orders;
    double res, prod, *coef_data, *x_data, q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!ddii",
            &PyArray_Type, &bf_orders,
            &PyArray_Type, &coefs,
            &PyArray_Type, &x,
            &q_fit_offset,
            &q_fit_slope,
            &q_max_bfOrder,
            &chi_max_bfOrder)) return NULL;

    // array to store powers of fit parameters. These go into the basis
    // functions
    double x_powers[q_max_bfOrder+1 + 6*(chi_max_bfOrder+1)];

    // Point to numpy array data
    bf_order_data = (long *) PyArray_DATA(bf_orders);
    coef_data = (double *) PyArray_DATA(coefs);
    x_data = (double *) PyArray_DATA(x);
    n = PyArray_DIMS(coefs)[0];
    res = 0.0;

    // Compute all needed powers
    for (i=0; i <= q_max_bfOrder; i++){        // power of q parameter
        x_powers[i] = ipow(q_fit_offset + q_fit_slope*x_data[0], i);
    }
    int base_idx;
    for (i=0; i <= chi_max_bfOrder; i++){      // power of chi parameters
        for (j=1; j<7; j++){
            base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
            x_powers[base_idx + i] = ipow(x_data[j], i);
        }
    }

    // Sum up result
    for (i=0; i<n; i++) {
        orders = bf_order_data + i*7;   // shift address of pointer
        prod = x_powers[orders[0]];
        for (j=1; j<7; j++) {
            base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
            prod *= x_powers[base_idx+ orders[j]];
        }
        res += coef_data[i]*prod;
    }

    return Py_BuildValue("d", res);
}


/*
 * Evaluates a batch of scalar fits sharing the same x (fit_params) vector.
 * This avoids recomputing x_powers for each fit, reducing overhead by ~9x
 * compared to calling eval_fit 9 times per ODE step.
 *
 * Arguments (with python data types):
 *      fit_list:  A Python list of (bfOrders, coefs) tuples, one per fit.
 *                 bfOrders: 2d int numpy array shape (n_coefs, 7)
 *                 coefs: 1d float numpy array length n_coefs
 *      x:             A 1d float numpy array with length 7 (fit parameters).
 *      q_fit_offset:  A double.
 *      q_fit_slope:   A double.
 *      q_max_bfOrder: An int.
 *      chi_max_bfOrder: An int.
 *
 * Returns a 1d float numpy array of length len(fit_list) with results.
 */
static PyObject *eval_fit_batch(PyObject *self, PyObject *args) {

    PyObject *fit_list;
    PyArrayObject *x;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;

    if (!PyArg_ParseTuple(args, "OO!ddii",
            &fit_list,
            &PyArray_Type, &x,
            &q_fit_offset,
            &q_fit_slope,
            &q_max_bfOrder,
            &chi_max_bfOrder)) return NULL;

    double *x_data = (double *) PyArray_DATA(x);
    int n_fits = (int) PyList_GET_SIZE(fit_list);

    /* Pre-compute x_powers once for all fits in the batch */
    double x_powers[q_max_bfOrder+1 + 6*(chi_max_bfOrder+1)];
    int i, j, base_idx;
    for (i=0; i <= q_max_bfOrder; i++) {
        x_powers[i] = ipow(q_fit_offset + q_fit_slope * x_data[0], i);
    }
    for (i=0; i <= chi_max_bfOrder; i++) {
        for (j=1; j<7; j++) {
            base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
            x_powers[base_idx + i] = ipow(x_data[j], i);
        }
    }

    /* Allocate output array */
    npy_intp dims[1] = {n_fits};
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return NULL;
    double *result_data = (double *) PyArray_DATA(result);

    /* Evaluate each fit using the shared x_powers */
    int k, n;
    for (k=0; k<n_fits; k++) {
        PyObject *fit_tuple = PyList_GET_ITEM(fit_list, k);
        PyArrayObject *bf_orders = (PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 0);
        PyArrayObject *coefs_arr = (PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 1);

        long *bf_order_data = (long *) PyArray_DATA(bf_orders);
        double *coef_data   = (double *) PyArray_DATA(coefs_arr);
        n = (int) PyArray_DIMS(coefs_arr)[0];

        double res = 0.0;
        long *orders;
        double prod;
        for (i=0; i<n; i++) {
            orders = bf_order_data + i*7;
            prod = x_powers[orders[0]];
            for (j=1; j<7; j++) {
                base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
                prod *= x_powers[base_idx + orders[j]];
            }
            res += coef_data[i] * prod;
        }
        result_data[k] = res;
    }

    return PyArray_Return(result);
}


/*
 * This is a helper function to normalize unit quaternions and spin magnitudes
 * at each time step during the ODE integration.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA
 *                  y[8:11] is chiB
 *      normA:  A python float giving |chiA|
 *      normB:  A python float giving |chiB|
 * Returns a python float array giving the normalized version of y.
 */
static PyObject *normalize_y(PyObject *self, PyObject *args) {

    PyArrayObject *y, *res;
    int i;
    double *y_data, *res_data, normA, normB, nA, nB, quatNorm, sum;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!dd",
            &PyArray_Type, &y,
            &normA,
            &normB)) return NULL;

    // Initialize output array
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    res_data = (double *) PyArray_DATA(res);

    y_data = (double *) PyArray_DATA(y);

    // Compute current norms
    sum = 0.0;
    for (i=0; i<4; i++) {
        sum += y_data[i]*y_data[i];
    }
    quatNorm = sqrt(sum);
    sum = 0.0;
    for (i=5; i<8; i++) { // Yes, i=5, not i=4. i=4 is the orbital phase.
        sum += y_data[i]*y_data[i];
    }
    nA = sqrt(sum);
    sum = 0.0;
    for (i=8; i<11; i++) {
        sum += y_data[i]*y_data[i];
    }
    nB = sqrt(sum);

    // Normalize output
    for (i=0; i<4; i++) {
        res_data[i] = y_data[i] / quatNorm;
    }
    res_data[4] = y_data[4];
    for (i=5; i<8; i++) {
        res_data[i] = y_data[i] * normA / nA;
    }
    for (i=8; i<11; i++) {
        res_data[i] = y_data[i] * normB / nB;
    }

    return PyArray_Return(res);
}

/*
 * This function computes the fit input for the dynamics surrogate.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      q:     A python float giving the mass ratio q
 * Returns a python float array x with length 7 containing:
 *                  x[0]: q
 *                  x[1:4]: chiA in the coorbital frame
 *                  x[4:7]: chiB in the coorbital frame
 */
static PyObject *get_ds_fit_x(PyObject *self, PyObject *args) {

    PyArrayObject *y, *x;
    double *y_data, *x_data, q, sp, cp;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &y, &q)) return NULL;

    y_data = (double *) PyArray_DATA(y);
    dims[0] = 7;
    x = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    x_data = (double *) PyArray_DATA(x);

    // q
    x_data[0] = q;

    // chiA
    sp = sin(y_data[4]);
    cp = cos(y_data[4]);
    x_data[1] = y_data[5]*cp + y_data[6]*sp;
    x_data[2] = -1*y_data[5]*sp + y_data[6]*cp;
    x_data[3] = y_data[7];

    // chiB
    x_data[4] = y_data[8]*cp + y_data[9]*sp;
    x_data[5] = -1*y_data[8]*sp + y_data[9]*cp;
    x_data[6] = y_data[10];

    return PyArray_Return(x);
}

/*
 * Fused eval_fit_batch + assemble_dydt: evaluates 9 polynomial fits and
 * assembles the ODE right-hand-side in a single C call.
 *
 * Arguments:
 *      fit_list:   Python list of 9 tuples (bf_orders, coefs)
 *      fit_params: length-7 float numpy array (transformed fit parameters)
 *      y:          length-11 float numpy array (quat, orbphase, chiA_copr, chiB_copr)
 *      q_fit_offset, q_fit_slope: doubles for q rescaling
 *      q_max_bfOrder, chi_max_bfOrder: ints for max basis function orders
 *
 * Returns a length-11 float numpy array dydt.
 */
static PyObject *eval_fit_batch_dydt(PyObject *self, PyObject *args) {

    PyObject *fit_list;
    PyArrayObject *fit_params, *y;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    PyArrayObject *q_consts_arr = NULL;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "OO!O!ddii|O!i",
            &fit_list,
            &PyArray_Type, &fit_params,
            &PyArray_Type, &y,
            &q_fit_offset,
            &q_fit_slope,
            &q_max_bfOrder,
            &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *x_data = (double *) PyArray_DATA(fit_params);
    double *y_data = (double *) PyArray_DATA(y);

    /* Apply fit_params transform in-place if requested */
    if (fit_params_mode == 0 && q_consts_arr != NULL) {
        double *q_consts = (double *) PyArray_DATA(q_consts_arr);
        double chi1z = x_data[3], chi2z = x_data[6];
        x_data[0] = q_consts[0];  /* log(q) */
        double chi_wtAvg = q_consts[1]*chi1z + q_consts[2]*chi2z;
        x_data[3] = (chi_wtAvg - q_consts[3]*(chi1z + chi2z))
                     / q_consts[4];
        x_data[6] = (chi1z - chi2z) * 0.5;
    }
    /* fit_params_mode == 1: identity, x unchanged */
    /* fit_params_mode == -1: legacy path, x already transformed by Python */

    /* Precompute offsets for x_powers indexing */
    int i, j;
    int offsets[7];
    offsets[0] = 0;
    for (j = 1; j < 7; j++) {
        offsets[j] = q_max_bfOrder + 1 + (chi_max_bfOrder + 1) * (j - 1);
    }

    /* Pre-compute x_powers once for all fits */
    double x_powers[q_max_bfOrder+1 + 6*(chi_max_bfOrder+1)];
    for (i=0; i <= q_max_bfOrder; i++) {
        x_powers[i] = ipow(q_fit_offset + q_fit_slope * x_data[0], i);
    }
    for (i=0; i <= chi_max_bfOrder; i++) {
        for (j=1; j<7; j++) {
            x_powers[offsets[j] + i] = ipow(x_data[j], i);
        }
    }

    /* Pre-extract fit data pointers outside the hot loop */
    long *all_bf_order_data[9];
    double *all_coef_data[9];
    int all_n[9];
    int k;
    for (k = 0; k < 9; k++) {
        PyObject *fit_tuple = PyList_GET_ITEM(fit_list, k);
        all_bf_order_data[k] = (long *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 0));
        all_coef_data[k] = (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 1));
        all_n[k] = (int) PyArray_DIMS((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 1))[0];
    }

    /* Evaluate 9 fits into stack array */
    double results[9];
    for (k = 0; k < 9; k++) {
        long *bf_order_data = all_bf_order_data[k];
        double *coef_data = all_coef_data[k];
        int n = all_n[k];

        double res = 0.0;
        for (i = 0; i < n; i++) {
            long *orders = bf_order_data + i * 7;
            double prod = x_powers[orders[0]]
                * x_powers[offsets[1] + orders[1]]
                * x_powers[offsets[2] + orders[2]]
                * x_powers[offsets[3] + orders[3]]
                * x_powers[offsets[4] + orders[4]]
                * x_powers[offsets[5] + orders[5]]
                * x_powers[offsets[6] + orders[6]];
            res += coef_data[i] * prod;
        }
        results[k] = res;
    }

    /* Assemble dydt from results: ooxy=results[0:2], omega=results[2],
       cAdot=results[3:6], cBdot=results[6:9] */
    npy_intp dims[1] = {11};
    PyArrayObject *dydt = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!dydt) return NULL;
    double *dydt_data = (double *) PyArray_DATA(dydt);

    double cp = cos(y_data[4]);
    double sp = sin(y_data[4]);

    /* Rotate ooxy from coorbital to coprecessing frame */
    double ooxy_x = results[0]*cp - results[1]*sp;
    double ooxy_y = results[0]*sp + results[1]*cp;

    /* Quaternion derivative */
    dydt_data[0] = (-0.5)*y_data[1]*ooxy_x - 0.5*y_data[2]*ooxy_y;
    dydt_data[1] = (-0.5)*y_data[3]*ooxy_y + 0.5*y_data[0]*ooxy_x;
    dydt_data[2] = 0.5*y_data[3]*ooxy_x + 0.5*y_data[0]*ooxy_y;
    dydt_data[3] = 0.5*y_data[1]*ooxy_y - 0.5*y_data[2]*ooxy_x;

    /* Orbital phase derivative */
    dydt_data[4] = results[2];

    /* Spin derivatives: rotate from coorbital to coprecessing */
    dydt_data[5] = results[3]*cp - results[4]*sp;
    dydt_data[6] = results[3]*sp + results[4]*cp;
    dydt_data[7] = results[5];
    dydt_data[8] = results[6]*cp - results[7]*sp;
    dydt_data[9] = results[6]*sp + results[7]*cp;
    dydt_data[10] = results[8];

    return PyArray_Return(dydt);
}


/*
 * This function assembles the right-hand-side of the dynamics ODE integration
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      ooxy:   A length 2 float numpy array with the x and y components of
 *              \Omega^\mathrm{coorb}
 *      omega:  A python float giving the orbital frequency
 *      cAdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiA, with components in the coorbital frame.
 *      cBdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiB, with components in the coorbital frame.
 * Returns a python float array dydt with length 11 containing the time
 * derivative of y
 */
static PyObject *assemble_dydt(PyObject *self, PyObject *args) {

    PyArrayObject *y, *ooxy, *cAdot, *cBdot, *dydt;
    double *y_data, *ooxy_data, *cAdot_data, *cBdot_data, *dydt_data;
    double omega, sp, cp, ooxy_x, ooxy_y;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!dO!O!",
            &PyArray_Type, &y,      // length 11, has quat, orbphase, chiA_copr, chiB_copr
            &PyArray_Type, &ooxy,   // length 2 with x and y components of \Omega^{copr}
            &omega,                 // double, orbital frequency \omega
            &PyArray_Type, &cAdot,  // length 3, coprecessing chiA time derivative
            &PyArray_Type, &cBdot)) return NULL;

    // Initialize output array
    dims[0] = 11;
    dydt = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    y_data = (double *) PyArray_DATA(y);
    ooxy_data = (double *) PyArray_DATA(ooxy);
    cAdot_data = (double *) PyArray_DATA(cAdot);
    cBdot_data = (double *) PyArray_DATA(cBdot);
    dydt_data = (double *) PyArray_DATA(dydt);

    // Quaternion derivative
    // Omega = 2 * quat^{-1} * dqdt -> dqdt = 0.5 * quat * ooxy_quat where
    // ooxy_quat = [0, ooxy_copr_x, ooxy_copr_y, 0]
    cp = cos(y_data[4]);
    sp = sin(y_data[4]);
    ooxy_x = ooxy_data[0]*cp - ooxy_data[1]*sp;
    ooxy_y = ooxy_data[0]*sp + ooxy_data[1]*cp;
    dydt_data[0] = (-0.5)*y_data[1]*ooxy_x - 0.5*y_data[2]*ooxy_y;
    dydt_data[1] = (-0.5)*y_data[3]*ooxy_y + 0.5*y_data[0]*ooxy_x;
    dydt_data[2] = 0.5*y_data[3]*ooxy_x + 0.5*y_data[0]*ooxy_y;
    dydt_data[3] = 0.5*y_data[1]*ooxy_y - 0.5*y_data[2]*ooxy_x;

    // Orbital phase derivative
    dydt_data[4] = omega;

    // Spin derivatives need to be rotated to the coprecessing frame
    dydt_data[5] = cAdot_data[0]*cp - cAdot_data[1]*sp;
    dydt_data[6] = cAdot_data[0]*sp + cAdot_data[1]*cp;
    dydt_data[7] = cAdot_data[2];
    dydt_data[8] = cBdot_data[0]*cp - cBdot_data[1]*sp;
    dydt_data[9] = cBdot_data[0]*sp + cBdot_data[1]*cp;
    dydt_data[10] = cBdot_data[2];

    return PyArray_Return(dydt);
}

/*
 * A helper function computing the update to y using the Adams-Bashforth
 * 4th-order ODE integration scheme.
 */
static PyObject *ab4_dy(PyObject *self, PyObject *args) {

    PyArrayObject *k1, *k2, *k3, *k4, *res;
    double *k1_data, *k2_data, *k3_data, *k4_data, *res_data;
    double dt1, dt2, dt3, dt4, dt12, dt123, dt23, D1, D2, D3,
            A, B, C, D, B41, B42, B43, B4, C41, C42, C43, C4;
    int i;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",
            &PyArray_Type, &k1,
            &PyArray_Type, &k2,
            &PyArray_Type, &k3,
            &PyArray_Type, &k4,
            &dt1, &dt2, &dt3, &dt4)) return NULL;

    // Initialize output
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    k1_data = (double *) PyArray_DATA(k1);
    k2_data = (double *) PyArray_DATA(k2);
    k3_data = (double *) PyArray_DATA(k3);
    k4_data = (double *) PyArray_DATA(k4);
    res_data = (double *) PyArray_DATA(res);

    // Various time intervals
    dt12 = dt1 + dt2;
    dt123 = dt12 + dt3;
    dt23 = dt2 + dt3;

    // Denomenators and coefficients
    D1 = dt1 * dt12 * dt123;
    D2 = dt1 * dt2 * dt23;
    D3 = dt2 * dt12 * dt3;

    B41 = dt3 * dt23 / D1;
    B42 = -1 * dt3 * dt123 / D2;
    B43 = dt23 * dt123 / D3;
    B4 = B41 + B42 + B43;

    C41 = (dt23 + dt3) / D1;
    C42 = -1 * (dt123 + dt3) / D2;
    C43 = (dt123 + dt23) / D3;
    C4 = C41 + C42 + C43;

    // Polynomial coefficients and result
    for (i=0; i<11; i++) {
        A = k4_data[i];
        B = k4_data[i]*B4 - k1_data[i]*B41 - k2_data[i]*B42 - k3_data[i]*B43;
        C = k4_data[i]*C4 - k1_data[i]*C41 - k2_data[i]*C42 - k3_data[i]*C43;
        D = (k4_data[i]-k1_data[i])/D1 - (k4_data[i]-k2_data[i])/D2 + (k4_data[i]-k3_data[i])/D3;
        res_data[i] = dt4 * (A + dt4 * (0.5*B + dt4*( C/3.0 + dt4*0.25*D)));
    }

    // Sum up contributions
    return PyArray_Return(res);
}

/* ------------------------------------------------------------------ */
/*  Static helpers for fused AB4 integration (no Python allocations)   */
/* ------------------------------------------------------------------ */

/* get_ds_fit_x logic: writes 7 doubles into x_out */
static void compute_fit_x(const double *y, double q, double *x_out) {
    double sp = sin(y[4]);
    double cp = cos(y[4]);
    x_out[0] = q;
    x_out[1] = y[5]*cp + y[6]*sp;
    x_out[2] = -y[5]*sp + y[6]*cp;
    x_out[3] = y[7];
    x_out[4] = y[8]*cp + y[9]*sp;
    x_out[5] = -y[8]*sp + y[9]*cp;
    x_out[6] = y[10];
}

/* eval_fit_batch_dydt core: evaluate 9 fits + assemble dydt → writes 11 doubles into dydt_out */
static void compute_dydt_core(
    long **all_bf_order_data, double **all_coef_data, int *all_n,
    double *x, const double *y,
    double q_fit_offset, double q_fit_slope,
    int q_max_bfOrder, int chi_max_bfOrder,
    const double *q_consts, int fit_params_mode,
    double *dydt_out)
{
    int i, j, k;

    /* Apply fit_params transform if requested */
    if (fit_params_mode == 0 && q_consts != NULL) {
        double chi1z = x[3], chi2z = x[6];
        x[0] = q_consts[0];
        double chi_wtAvg = q_consts[1]*chi1z + q_consts[2]*chi2z;
        x[3] = (chi_wtAvg - q_consts[3]*(chi1z + chi2z)) / q_consts[4];
        x[6] = (chi1z - chi2z) * 0.5;
    }

    /* Precompute offsets */
    int offsets[7];
    offsets[0] = 0;
    for (j = 1; j < 7; j++) {
        offsets[j] = q_max_bfOrder + 1 + (chi_max_bfOrder + 1) * (j - 1);
    }

    /* Pre-compute x_powers */
    double x_powers[q_max_bfOrder+1 + 6*(chi_max_bfOrder+1)];
    for (i = 0; i <= q_max_bfOrder; i++) {
        x_powers[i] = ipow(q_fit_offset + q_fit_slope * x[0], i);
    }
    for (j = 1; j < 7; j++) {
      for (i = 0; i <= chi_max_bfOrder; i++) {
        x_powers[offsets[j] + i] = ipow(x[j], i);
      }
    }

    /* Evaluate 9 fits */
    double results[9];
    for (k = 0; k < 9; k++) {
        long *bf_order_data = all_bf_order_data[k];
        double *coef_data = all_coef_data[k];
        int n = all_n[k];

        double res = 0.0;
        for (i = 0; i < n; i++) {
            long *orders = bf_order_data + i * 7;
            double prod = x_powers[orders[0]]
                * x_powers[offsets[1] + orders[1]]
                * x_powers[offsets[2] + orders[2]]
                * x_powers[offsets[3] + orders[3]]
                * x_powers[offsets[4] + orders[4]]
                * x_powers[offsets[5] + orders[5]]
                * x_powers[offsets[6] + orders[6]];
            res += coef_data[i] * prod;
        }
        results[k] = res;
    }

    /* Assemble dydt */
    double cp = cos(y[4]);
    double sp = sin(y[4]);
    double ooxy_x = results[0]*cp - results[1]*sp;
    double ooxy_y = results[0]*sp + results[1]*cp;

    dydt_out[0] = (-0.5)*y[1]*ooxy_x - 0.5*y[2]*ooxy_y;
    dydt_out[1] = (-0.5)*y[3]*ooxy_y + 0.5*y[0]*ooxy_x;
    dydt_out[2] = 0.5*y[3]*ooxy_x + 0.5*y[0]*ooxy_y;
    dydt_out[3] = 0.5*y[1]*ooxy_y - 0.5*y[2]*ooxy_x;
    dydt_out[4] = results[2];
    dydt_out[5] = results[3]*cp - results[4]*sp;
    dydt_out[6] = results[3]*sp + results[4]*cp;
    dydt_out[7] = results[5];
    dydt_out[8] = results[6]*cp - results[7]*sp;
    dydt_out[9] = results[6]*sp + results[7]*cp;
    dydt_out[10] = results[8];
}

/* ab4_dy logic: writes 11 doubles into dy_out */
static void compute_ab4_dy(const double *k1, const double *k2, const double *k3, const double *k4,
                           double dt1, double dt2, double dt3, double dt4, double *dy_out)
{
    double dt12 = dt1 + dt2;
    double dt123 = dt12 + dt3;
    double dt23 = dt2 + dt3;

    double D1 = dt1 * dt12 * dt123;
    double D2 = dt1 * dt2 * dt23;
    double D3 = dt2 * dt12 * dt3;

    double B41 = dt3 * dt23 / D1;
    double B42 = -dt3 * dt123 / D2;
    double B43 = dt23 * dt123 / D3;
    double B4 = B41 + B42 + B43;

    double C41 = (dt23 + dt3) / D1;
    double C42 = -(dt123 + dt3) / D2;
    double C43 = (dt123 + dt23) / D3;
    double C4 = C41 + C42 + C43;

    int i;
    for (i = 0; i < 11; i++) {
        double A = k4[i];
        double B = k4[i]*B4 - k1[i]*B41 - k2[i]*B42 - k3[i]*B43;
        double C = k4[i]*C4 - k1[i]*C41 - k2[i]*C42 - k3[i]*C43;
        double D = (k4[i]-k1[i])/D1 - (k4[i]-k2[i])/D2 + (k4[i]-k3[i])/D3;
        dy_out[i] = dt4 * (A + dt4 * (0.5*B + dt4*(C/3.0 + dt4*0.25*D)));
    }
}

/* normalize_y logic: modifies y in-place */
static void normalize_y_inplace(double *y, const double normA,
                                const double normB) {
    int i;
    double sum = 0.0;
    for (i = 0; i < 4; i++) {
      sum += y[i]*y[i];
    }
    const double oneOverQuatNorm = 1.0 / sqrt(sum);

    sum = 0.0;
    for (i = 5; i < 8; i++) {
      sum += y[i]*y[i];
    }
    const double oneOverNormA = 1.0 / sqrt(sum);

    sum = 0.0;
    for (i = 8; i < 11; i++) {
      sum += y[i]*y[i];
    }
    const double oneOverNormB = 1.0 / sqrt(sum);

    for (i = 0; i < 4; i++) {
      y[i] *= oneOverQuatNorm;
    }
    /* y[4] unchanged (orbital phase) */
    for (i = 5; i < 8; i++) {
      y[i] *= normA * oneOverNormA;
    }
    for (i = 8; i < 11; i++) {
      y[i] *= normB  * oneOverNormB;
    }
}

/* Helper to extract fit data pointers from a Python list of 9 tuples */
static int extract_fit_data(PyObject *fit_list,
                            long **all_bf_order_data,
                            double **all_coef_data,
                            int *all_n)
{
    int k;
    for (k = 0; k < 9; k++) {
        PyObject *fit_tuple = PyList_GET_ITEM(fit_list, k);
        all_bf_order_data[k] = (long *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 0));
        all_coef_data[k] = (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 1));
        all_n[k] = (int) PyArray_DIMS((PyArrayObject *) PyTuple_GET_ITEM(fit_tuple, 1))[0];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Cubic Lagrange interpolation over 4 knots                          */
/* ------------------------------------------------------------------ */
static double lagrange4_interp(double t, const double ts[4],
                               const double vals[4]) {
    double result = 0.0;
    int i, j;
    for (i = 0; i < 4; i++) {
        double basis = vals[i];
        for (j = 0; j < 4; j++) {
            if (j != i) {
                basis *= (t - ts[j]) / (ts[i] - ts[j]);
            }
        }
        result += basis;
    }
    return result;
}

/* Python-callable wrapper for testing */
static PyObject *py_lagrange4_interp(PyObject *self, PyObject *args) {
    double t;
    PyArrayObject *ts_arr, *vals_arr;
    if (!PyArg_ParseTuple(args, "dO!O!",
            &t,
            &PyArray_Type, &ts_arr,
            &PyArray_Type, &vals_arr)) return NULL;
    double *ts = (double *) PyArray_DATA(ts_arr);
    double *vals = (double *) PyArray_DATA(vals_arr);
    double result = lagrange4_interp(t, ts, vals);
    return Py_BuildValue("d", result);
}

/* ------------------------------------------------------------------ */
/*  compute_dydt_interp: evaluate dydt at arbitrary time t via         */
/*  Lagrange interpolation of dydt at 4 nearest nodes                  */
/* ------------------------------------------------------------------ */
static void compute_dydt_interp(
    double t, const double *y, double q,
    PyObject *fit_data_batch, const double *t_array, int n_t,
    double q_fit_offset, double q_fit_slope,
    int q_max_bfOrder, int chi_max_bfOrder,
    const double *q_consts, int fit_params_mode,
    double *dydt_out)
{
    /* Find imin: the starting index of the 4-node bracket */
    int i0_best = 0;
    double min_dist = fabs(t_array[0] - t);
    int i;
    for (i = 1; i < n_t; i++) {
        double d = fabs(t_array[i] - t);
        if (d < min_dist) {
            min_dist = d;
            i0_best = i;
        }
    }
    int imin;
    if (t > t_array[i0_best]) {
        imin = i0_best - 1;
    } else {
        imin = i0_best - 2;
    }
    if (imin < 0) imin = 0;
    if (imin > n_t - 4) imin = n_t - 4;

    /* Evaluate dydt at the 4 nodes */
    double dydts[4][11];
    double ts[4];
    int ni;
    for (ni = 0; ni < 4; ni++) {
        int node_index = imin + ni;
        ts[ni] = t_array[node_index];

        /* Extract packed fit data for this node */
        PyObject *node_data = PyList_GET_ITEM(fit_data_batch, node_index);
        long *pk_orders = (long *) PyArray_DATA(
            (PyArrayObject *) PyTuple_GET_ITEM(node_data, 0));
        double *pk_coefs = (double *) PyArray_DATA(
            (PyArrayObject *) PyTuple_GET_ITEM(node_data, 1));
        int N = (int) PyLong_AsLong(PyTuple_GET_ITEM(node_data, 2));
        int *ns = (int *) PyArray_DATA(
            (PyArrayObject *) PyTuple_GET_ITEM(node_data, 3));

        long *all_bf_order_data[9];
        double *all_coef_data[9];
        int all_n[9];
        int kk;
        for (kk = 0; kk < 9; kk++) {
            all_bf_order_data[kk] = pk_orders + kk * N * 7;
            all_coef_data[kk] = pk_coefs + kk * N;
            all_n[kk] = ns[kk];
        }

        double x[7];
        compute_fit_x(y, q, x);
        compute_dydt_core(all_bf_order_data, all_coef_data, all_n,
                          x, y,
                          q_fit_offset, q_fit_slope,
                          q_max_bfOrder, chi_max_bfOrder,
                          q_consts, fit_params_mode,
                          dydts[ni]);
    }

    /* Lagrange interpolate each of 11 components */
    int c;
    for (c = 0; c < 11; c++) {
        double vals[4] = {dydts[0][c], dydts[1][c], dydts[2][c], dydts[3][c]};
        dydt_out[c] = lagrange4_interp(t, ts, vals);
    }
}

/* ------------------------------------------------------------------ */
/*  RK4 step forward: one step from i0 to i0+1                        */
/* ------------------------------------------------------------------ */
static PyObject *rk4_step_forward(PyObject *self, PyObject *args) {
    PyObject *fit_data_batch;
    PyArrayObject *y_of_t_arr, *t_array_arr;
    PyArrayObject *q_consts_arr = NULL;
    double q, normA, normB;
    int i0;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "OO!dddiO!ddii|O!i",
            &fit_data_batch,
            &PyArray_Type, &y_of_t_arr,
            &q, &normA, &normB,
            &i0,
            &PyArray_Type, &t_array_arr,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *y_of_t = (double *) PyArray_DATA(y_of_t_arr);
    double *t_array = (double *) PyArray_DATA(t_array_arr);
    int n_t = (int) PyArray_DIMS(t_array_arr)[0];
    double *q_consts = q_consts_arr ? (double *) PyArray_DATA(q_consts_arr) : NULL;

    /* Map i0 (y_of_t index) to t index */
    int i_t = i0 + 3;
    if (i0 < 3) i_t = i0 * 2;

    double t1 = t_array[i_t];
    double t2 = t_array[i_t + 1];
    if (i0 < 3) t2 = t_array[i_t + 2];
    double half_dt = 0.5 * (t2 - t1);

    double *y_cur = y_of_t + i0 * 11;
    double k1[11], k2[11], k3[11], k4[11];
    double y_tmp[11];
    int j;

    /* k1 = f(t1, y_cur) */
    compute_dydt_interp(t1, y_cur, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k1);

    /* k2 = f(t1 + half_dt, y_cur + half_dt*k1) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + half_dt * k1[j];
    compute_dydt_interp(t1 + half_dt, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k2);

    /* k3 = f(t1 + half_dt, y_cur + half_dt*k2) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + half_dt * k2[j];
    compute_dydt_interp(t1 + half_dt, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k3);

    /* k4 = f(t2, y_cur + 2*half_dt*k3) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + 2.0 * half_dt * k3[j];
    compute_dydt_interp(t2, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k4);

    /* y_{n+1} = y_n + (half_dt/3)(k1 + 2k2 + 2k3 + k4) */
    double *y_next = y_of_t + (i0 + 1) * 11;
    double coeff = half_dt / 3.0;
    for (j = 0; j < 11; j++) {
        y_next[j] = y_cur[j] + coeff * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
    }
    normalize_y_inplace(y_next, normA, normB);

    /* Return k1 as numpy array (needed for AB4 history) */
    npy_intp dims[1] = {11};
    PyArrayObject *k1_out = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!k1_out) return NULL;
    memcpy(PyArray_DATA(k1_out), k1, 11 * sizeof(double));

    return PyArray_Return(k1_out);
}

/* ------------------------------------------------------------------ */
/*  RK4 step backward: one step from i0 to i0-1                       */
/* ------------------------------------------------------------------ */
static PyObject *rk4_step_backward(PyObject *self, PyObject *args) {
    PyObject *fit_data_batch;
    PyArrayObject *y_of_t_arr, *t_array_arr;
    PyArrayObject *q_consts_arr = NULL;
    double q, normA, normB;
    int i0;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "OO!dddiO!ddii|O!i",
            &fit_data_batch,
            &PyArray_Type, &y_of_t_arr,
            &q, &normA, &normB,
            &i0,
            &PyArray_Type, &t_array_arr,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *y_of_t = (double *) PyArray_DATA(y_of_t_arr);
    double *t_array = (double *) PyArray_DATA(t_array_arr);
    int n_t = (int) PyArray_DIMS(t_array_arr)[0];
    double *q_consts = q_consts_arr ? (double *) PyArray_DATA(q_consts_arr) : NULL;

    /* Map i0 (y_of_t index) to t index */
    int i_t = i0 + 3;
    if (i0 < 3) i_t = i0 * 2;

    double t1 = t_array[i_t];
    double t2 = t_array[i_t - 1];
    if (i0 <= 3) t2 = t_array[i_t - 2];
    double half_dt = 0.5 * (t2 - t1);

    double *y_cur = y_of_t + i0 * 11;
    double k1[11], k2[11], k3[11], k4[11];
    double y_tmp[11];
    int j;

    /* k1 = f(t1, y_cur) */
    compute_dydt_interp(t1, y_cur, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k1);

    /* k2 = f(t1 + half_dt, y_cur + half_dt*k1) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + half_dt * k1[j];
    compute_dydt_interp(t1 + half_dt, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k2);

    /* k3 = f(t1 + half_dt, y_cur + half_dt*k2) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + half_dt * k2[j];
    compute_dydt_interp(t1 + half_dt, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k3);

    /* k4 = f(t2, y_cur + 2*half_dt*k3) */
    for (j = 0; j < 11; j++) y_tmp[j] = y_cur[j] + 2.0 * half_dt * k3[j];
    compute_dydt_interp(t2, y_tmp, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, k4);

    /* y_{n-1} = y_n + (half_dt/3)(k1 + 2k2 + 2k3 + k4) */
    double *y_prev = y_of_t + (i0 - 1) * 11;
    double coeff = half_dt / 3.0;
    for (j = 0; j < 11; j++) {
        y_prev[j] = y_cur[j] + coeff * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
    }
    normalize_y_inplace(y_prev, normA, normB);

    /* Return k1 as numpy array (needed for AB4 history) */
    npy_intp dims[1] = {11};
    PyArrayObject *k1_out = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!k1_out) return NULL;
    memcpy(PyArray_DATA(k1_out), k1, 11 * sizeof(double));

    return PyArray_Return(k1_out);
}

/* ------------------------------------------------------------------ */
/*  compute_dydt_interp: Python-callable wrapper for testing           */
/* ------------------------------------------------------------------ */
static PyObject *py_compute_dydt_interp(PyObject *self, PyObject *args) {
    double t, q;
    PyArrayObject *y_arr, *t_array_arr;
    PyObject *fit_data_batch;
    PyArrayObject *q_consts_arr = NULL;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "dO!dOO!ddii|O!i",
            &t,
            &PyArray_Type, &y_arr,
            &q,
            &fit_data_batch,
            &PyArray_Type, &t_array_arr,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *y = (double *) PyArray_DATA(y_arr);
    double *t_array = (double *) PyArray_DATA(t_array_arr);
    int n_t = (int) PyArray_DIMS(t_array_arr)[0];
    double *q_consts = q_consts_arr ? (double *) PyArray_DATA(q_consts_arr) : NULL;

    double dydt_out[11];
    compute_dydt_interp(t, y, q, fit_data_batch, t_array, n_t,
                        q_fit_offset, q_fit_slope,
                        q_max_bfOrder, chi_max_bfOrder,
                        q_consts, fit_params_mode, dydt_out);

    npy_intp dims[1] = {11};
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return NULL;
    memcpy(PyArray_DATA(result), dydt_out, 11 * sizeof(double));
    return PyArray_Return(result);
}

/* ------------------------------------------------------------------ */
/*  Fused AB4 forward integration                                      */
/* ------------------------------------------------------------------ */
static PyObject *integrate_ab4_forward(PyObject *self, PyObject *args) {
    PyObject *fit_data_batch;
    PyArrayObject *y_of_t_arr, *k1_arr, *k2_arr, *k3_arr, *diff_t_arr;
    PyArrayObject *q_consts_arr = NULL;
    double q, normA, normB;
    int i0;
    double dt1, dt2, dt3;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "OO!dddiO!O!O!dddO!ddii|O!i",
            &fit_data_batch,
            &PyArray_Type, &y_of_t_arr,
            &q, &normA, &normB,
            &i0,
            &PyArray_Type, &k1_arr,
            &PyArray_Type, &k2_arr,
            &PyArray_Type, &k3_arr,
            &dt1, &dt2, &dt3,
            &PyArray_Type, &diff_t_arr,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *y_of_t = (double *) PyArray_DATA(y_of_t_arr);
    double *diff_t = (double *) PyArray_DATA(diff_t_arr);
    int n_diff_t = (int) PyArray_DIMS(diff_t_arr)[0];
    int n_steps = n_diff_t - (i0 + 3);  /* matches Python: diff_t[i0+3:] */

    double *q_consts = q_consts_arr ? (double *) PyArray_DATA(q_consts_arr) : NULL;

    /* Copy initial k values to local buffers */
    double lk1[11], lk2[11], lk3[11], lk4[11];
    double x[7], dy[11];
    memcpy(lk1, PyArray_DATA(k1_arr), 11*sizeof(double));
    memcpy(lk2, PyArray_DATA(k2_arr), 11*sizeof(double));
    memcpy(lk3, PyArray_DATA(k3_arr), 11*sizeof(double));

    int i;
    for (i = 0; i < n_steps; i++) {
        int i_output = i0 + i;
        int node_index = i_output + 3;
        double dt4 = diff_t[i0 + 3 + i];
        double *y_cur = y_of_t + i_output * 11;
        double *y_next = y_of_t + (i_output + 1) * 11;

        /* Extract packed fit data for this node and set up pointer arrays */
        PyObject *node_data = PyList_GET_ITEM(fit_data_batch, node_index);
        long *pk_orders = (long *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 0));
        double *pk_coefs = (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 1));
        int N = (int) PyLong_AsLong(PyTuple_GET_ITEM(node_data, 2));
        int *ns = (int *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 3));

        long *all_bf_order_data[9];
        double *all_coef_data[9];
        int all_n[9];
        { int kk; for (kk = 0; kk < 9; kk++) {
            all_bf_order_data[kk] = pk_orders + kk * N * 7;
            all_coef_data[kk] = pk_coefs + kk * N;
            all_n[kk] = ns[kk];
        }}

        /* compute_fit_x */
        compute_fit_x(y_cur, q, x);

        /* compute_dydt_core → k4 */
        compute_dydt_core(all_bf_order_data, all_coef_data, all_n,
                          x, y_cur,
                          q_fit_offset, q_fit_slope,
                          q_max_bfOrder, chi_max_bfOrder,
                          q_consts, fit_params_mode, lk4);

        /* compute_ab4_dy → dy */
        compute_ab4_dy(lk1, lk2, lk3, lk4, dt1, dt2, dt3, dt4, dy);

        /* y_next = y_cur + dy */
        int j;
        for (j = 0; j < 11; j++) {
            y_next[j] = y_cur[j] + dy[j];
        }

        /* normalize in-place */
        normalize_y_inplace(y_next, normA, normB);

        /* shift */
        memcpy(lk1, lk2, 11*sizeof(double));
        memcpy(lk2, lk3, 11*sizeof(double));
        memcpy(lk3, lk4, 11*sizeof(double));
        dt1 = dt2; dt2 = dt3; dt3 = dt4;
    }

    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/*  Fused AB4 backward integration                                     */
/* ------------------------------------------------------------------ */
static PyObject *integrate_ab4_backward(PyObject *self, PyObject *args) {
    PyObject *fit_data_batch;
    PyArrayObject *y_of_t_arr, *k1_arr, *k2_arr, *k3_arr, *dt_array_arr;
    PyArrayObject *q_consts_arr = NULL;
    double q, normA, normB;
    int i0;
    double dt1, dt2, dt3;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;
    int fit_params_mode = -1;

    if (!PyArg_ParseTuple(args, "OO!dddiO!O!O!dddO!ddii|O!i",
            &fit_data_batch,
            &PyArray_Type, &y_of_t_arr,
            &q, &normA, &normB,
            &i0,
            &PyArray_Type, &k1_arr,
            &PyArray_Type, &k2_arr,
            &PyArray_Type, &k3_arr,
            &dt1, &dt2, &dt3,
            &PyArray_Type, &dt_array_arr,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder,
            &PyArray_Type, &q_consts_arr,
            &fit_params_mode)) return NULL;

    double *y_of_t = (double *) PyArray_DATA(y_of_t_arr);
    double *dt_array = (double *) PyArray_DATA(dt_array_arr);

    double *q_consts = q_consts_arr ? (double *) PyArray_DATA(q_consts_arr) : NULL;

    /* Copy initial k values to local buffers */
    double lk1[11], lk2[11], lk3[11], lk4[11];
    double x[7], dy[11];
    memcpy(lk1, PyArray_DATA(k1_arr), 11*sizeof(double));
    memcpy(lk2, PyArray_DATA(k2_arr), 11*sizeof(double));
    memcpy(lk3, PyArray_DATA(k3_arr), 11*sizeof(double));

    int i_output;
    for (i_output = i0 - 1; i_output >= 0; i_output--) {
        int node_index = i_output + 4;
        if (i_output < 2) {
            node_index = 2 + 2*i_output;
        }
        double dt4 = dt_array[i_output];
        double *y_next = y_of_t + (i_output + 1) * 11;  /* the "later" row */
        double *y_cur = y_of_t + i_output * 11;          /* where we write */

        /* Extract packed fit data for this node and set up pointer arrays */
        PyObject *node_data = PyList_GET_ITEM(fit_data_batch, node_index);
        long *pk_orders = (long *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 0));
        double *pk_coefs = (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 1));
        int N = (int) PyLong_AsLong(PyTuple_GET_ITEM(node_data, 2));
        int *ns = (int *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(node_data, 3));

        long *all_bf_order_data[9];
        double *all_coef_data[9];
        int all_n[9];
        { int kk; for (kk = 0; kk < 9; kk++) {
            all_bf_order_data[kk] = pk_orders + kk * N * 7;
            all_coef_data[kk] = pk_coefs + kk * N;
            all_n[kk] = ns[kk];
        }}

        /* compute_fit_x from y[i_output+1] */
        compute_fit_x(y_next, q, x);

        /* compute_dydt_core → k4 */
        compute_dydt_core(all_bf_order_data, all_coef_data, all_n,
                          x, y_next,
                          q_fit_offset, q_fit_slope,
                          q_max_bfOrder, chi_max_bfOrder,
                          q_consts, fit_params_mode, lk4);

        /* compute_ab4_dy → dy */
        compute_ab4_dy(lk1, lk2, lk3, lk4, dt1, dt2, dt3, dt4, dy);

        /* y_cur = y_next - dy */
        int j;
        for (j = 0; j < 11; j++) {
            y_cur[j] = y_next[j] - dy[j];
        }

        /* normalize in-place */
        normalize_y_inplace(y_cur, normA, normB);

        /* shift */
        memcpy(lk1, lk2, 11*sizeof(double));
        memcpy(lk2, lk3, 11*sizeof(double));
        memcpy(lk3, lk4, 11*sizeof(double));
        dt1 = dt2; dt2 = dt3; dt3 = dt4;
    }

    Py_RETURN_NONE;
}

double factorial(int n) {
    if (n <= 0) return 1.0;
    return factorial(n-1) * n;
}

double factorial_ratio(int n, int k) {
    if (n <= k) return 1.0;
    return factorial_ratio(n-1, k) * n;
}

double _binomial(int n, int k) {
    return factorial_ratio(n, k) / factorial(n-k);
}

double _wigner_coef(int ell, int mp, int m) {
    return sqrt(factorial(ell+m) * factorial(ell-m) / (factorial(ell+mp) * factorial(ell-mp)));
}

static PyObject *binom(PyObject *self, PyObject *args) {
    int n, k;
    double b;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "ii", &n, &k)) return NULL;

    // Compute result
    b = _binomial(n, k);

    // Return result
    return Py_BuildValue("d", b);
}

static PyObject *wigner_coef(PyObject *self, PyObject *args) {
    int ell, mp, m;
    double wc;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "iii", &ell, &mp, &m)) return NULL;

    // Compute result
    wc = _wigner_coef(ell, mp, m);

    // Return result
    return Py_BuildValue("d", wc);
}

/* ------------------------------------------------------------------ */
/*  Bump allocator                                                    */
/* ------------------------------------------------------------------ */

typedef struct { char *ptr, *end; } Bump;

static inline void *bump(Bump *b, size_t bytes)
{
    char *p = (char *)(((size_t)b->ptr + 63) & ~(size_t)63);
    if (p + bytes > b->end) return NULL;
    b->ptr = p + bytes;
    return p;
}

#define BUMP(b, type, count) ((type *)bump(&(b), (size_t)(count) * sizeof(type)))

/* ---- BumpSizer: exact capacity via a sizing pass ---- */
typedef struct { size_t offset, high_water; } BumpSizer;

static inline void bump_need(BumpSizer *s, size_t bytes)
{
    s->offset = (s->offset + 63) & ~(size_t)63;
    s->offset += bytes;
    if (s->offset > s->high_water) s->high_water = s->offset;
}

static inline BumpSizer bump_sizer_save(const BumpSizer *s)  { return *s; }
static inline void bump_sizer_restore(BumpSizer *s, BumpSizer saved)
{
    s->offset = saved.offset;
    /* high_water is NOT restored — it tracks the global max */
}

#define BUMP_NEED(s, type, count) bump_need(&(s), (size_t)(count) * sizeof(type))

/* ------------------------------------------------------------------ */
/*  Coefficient helpers — all take the lf table as a parameter        */
/* ------------------------------------------------------------------ */

static inline double rho_coeff(const double *lf,
                               int ell, int mp, int m, int rho)
{
    int a = ell + mp, b = ell - mp, c = ell - rho - m;
    if (rho < 0 || rho > a || c < 0 || c > b) return 0.0;
    double lc = lf[a] - lf[rho] - lf[a - rho]
              + lf[b] - lf[c]   - lf[b - c];
    return (rho & 1 ? -1.0 : 1.0) * exp(lc);
}

static inline double wigner_coef2(const double *lf, int ell, int mp, int m)
{
    return exp(0.5 * (lf[ell+m] + lf[ell-m] - lf[ell+mp] - lf[ell-mp]));
}

/* ------------------------------------------------------------------ */
/*  Complex power table builder                                       */
/* ------------------------------------------------------------------ */

static void build_cpows(const double complex *base,
                        const double complex *inv,
                        size_t len, int half,
                        double complex *out)
{
    size_t z = (size_t)half * len;
    for (size_t k = 0; k < len; ++k) out[z + k] = 1.0;
    for (int p = 1; p <= half; ++p) {
        size_t c = (size_t)(half+p)*len, pv = c - len;
        size_t d = (size_t)(half-p)*len, dv = d + len;
        for (size_t k = 0; k < len; ++k) {
            out[c+k] = out[pv+k] * base[k];
            out[d+k] = out[dv+k] * inv[k];
        }
    }
}

/* ================================================================== */
/*  Real-only power table builder (for |ra|^2 based powers)           */
/* ================================================================== */
static void build_real_pows(const double *base, size_t len, int maxp,
                            double *out)
{
    /* out[p * len + k] = base[k]^p for p = 0..maxp */
    for (size_t k = 0; k < len; ++k) out[k] = 1.0;
    for (int p = 1; p <= maxp; ++p) {
        size_t c = (size_t)p * len, pv = c - len;
        for (size_t k = 0; k < len; ++k)
            out[c + k] = out[pv + k] * base[k];
    }
}

/* ================================================================== */
/*  Unit-phase power table builder                                    */
/* ================================================================== */
static void build_unit_pows(const double complex *unit,
                            const double complex *unit_inv,
                            size_t len, int half,
                            double complex *out)
{
    /* out[(half+p)*len + k] = unit[k]^p for p = -half..+half */
    size_t z = (size_t)half * len;
    for (size_t k = 0; k < len; ++k) out[z + k] = 1.0;
    for (int p = 1; p <= half; ++p) {
        size_t c = (size_t)(half + p) * len, pv = c - len;
        size_t d = (size_t)(half - p) * len, dv = d + len;
        for (size_t k = 0; k < len; ++k) {
            out[c + k] = out[pv + k] * unit[k];
            out[d + k] = out[dv + k] * unit_inv[k];
        }
    }
}

/* ================================================================== */
/*  Three-term recurrence coefficients for Wigner d-matrix            */
/*  d^ell_{m,mp} = (A*cos(beta)*d^{ell-1} + B*d^{ell-2}) / C        */
/* ================================================================== */
static inline void recurrence_abc(int ell, int m, int mp,
                                  double *A, double *B, double *C)
{
    double ell_d = (double)ell;
    double m_d = (double)m, mp_d = (double)mp;
    double ell2 = ell_d * ell_d;
    double m2 = m_d * m_d, mp2 = mp_d * mp_d;

    /* C = ell * sqrt((ell^2 - m^2)(ell^2 - m'^2)) / (something) */
    /* Standard three-term recurrence (e.g. Risbo 1996):
       ell*sqrt((ell^2-m^2)(ell^2-mp^2)) * d^ell =
         ((2ell-1)(ell*(ell-1)*cos_beta - m*mp)) * d^{ell-1}
         - (ell-1)*sqrt(((ell-1)^2-m^2)*((ell-1)^2-mp^2)) * d^{ell-2}
    */
    double ellm1 = ell_d - 1.0;
    double ellm1_sq = ellm1 * ellm1;

    /* Correct recurrence (Varshalovich / Risbo):
       (ell-1)*sqrt((ell^2-m^2)(ell^2-mp^2)) * d^ell =
         (2ell-1)(ell*(ell-1)*cos_beta - m*mp) * d^{ell-1}
         - ell*sqrt(((ell-1)^2-m^2)*((ell-1)^2-mp^2)) * d^{ell-2}  */
    *C = ellm1 * sqrt((ell2 - m2) * (ell2 - mp2));
    *A = (2.0 * ell_d - 1.0) * ell_d * ellm1;
    *B = -ell_d * sqrt((ellm1_sq - m2) * (ellm1_sq - mp2));
}

/* ====================================================================
 * wignerD_matrices
 * ====================================================================
 *
 *  Helper: compute d_real values for all fundamental domain pairs
 *  at a given ell using hardcoded polynomial formulas.
 *  c = sqrt(c2), s = sqrt(s2), cN = c^N, sN = s^N.
 *  Stores into d_buf[m_idx * dim_d * n1 + mp_idx * n1 + k] where
 *  m_idx = m + ellMax, mp_idx = mp + ellMax.
*/

static void hc_ell2(const double * restrict cv, const double * restrict sv,
                    size_t n1, int ellMax,
                    double * restrict d_buf)
{
    int dim_d = 2 * ellMax + 1;
    /* Precompute powers: c2..c4, s2..s4 per timepoint */
    for (size_t k = 0; k < n1; ++k) {
        double c = cv[k], s = sv[k];
        double c2 = c*c, s2 = s*s;
        double c3 = c2*c, s3 = s2*s;
        double c4 = c2*c2, s4 = s2*s2;

        #define SD(m, mp, val) d_buf[(size_t)((m)+ellMax)*(size_t)dim_d*n1 + (size_t)((mp)+ellMax)*n1 + k] = (val)

        /* Fundamental domain for ell=2: mp>=0, m+mp>=0 */
        SD(-2, 2,  s4);
        SD(-1, 1,  -s4 + 3.0*c2*s2);
        SD(-1, 2,  -2.0*c*s3);
        SD( 0, 0,  s4 - 4.0*c2*s2 + c4);
        SD( 0, 1,  sqrt(6.0)*c*s3 - sqrt(6.0)*c3*s);
        SD( 0, 2,  sqrt(6.0)*c2*s2);
        SD( 1, 0,  -sqrt(6.0)*c*s3 + sqrt(6.0)*c3*s);
        SD( 1, 1,  -3.0*c2*s2 + c4);
        SD( 1, 2,  -2.0*c3*s);
        SD( 2, 0,  sqrt(6.0)*c2*s2);
        SD( 2, 1,  2.0*c3*s);
        SD( 2, 2,  c4);

        #undef SD
    }
}

static void hc_ell3(const double * restrict cv, const double * restrict sv,
                    size_t n1, int ellMax,
                    double * restrict d_buf)
{
    int dim_d = 2 * ellMax + 1;
    for (size_t k = 0; k < n1; ++k) {
        double c = cv[k], s = sv[k];
        double c2 = c*c, s2 = s*s;
        double c3 = c2*c, s3 = s2*s;
        double c4 = c2*c2, s4 = s2*s2;
        double c5 = c4*c, s5 = s4*s;
        double c6 = c3*c3, s6 = s3*s3;

        #define SD(m, mp, val) d_buf[(size_t)((m)+ellMax)*(size_t)dim_d*n1 + (size_t)((mp)+ellMax)*n1 + k] = (val)

        SD(-3, 3,  s6);
        SD(-2, 2,  -s6 + 5.0*c2*s4);
        SD(-2, 3,  -sqrt(6.0)*c*s5);
        SD(-1, 1,  s6 - 8.0*c2*s4 + 6.0*c4*s2);
        SD(-1, 2,  sqrt(10.0)*c*s5 - 2.0*sqrt(10.0)*c3*s3);
        SD(-1, 3,  sqrt(15.0)*c2*s4);
        SD( 0, 0,  -s6 + 9.0*c2*s4 - 9.0*c4*s2 + c6);
        SD( 0, 1,  -2.0*sqrt(3.0)*c*s5 + 6.0*sqrt(3.0)*c3*s3 - 2.0*sqrt(3.0)*c5*s);
        SD( 0, 2,  -sqrt(30.0)*c2*s4 + sqrt(30.0)*c4*s2);
        SD( 0, 3,  -2.0*sqrt(5.0)*c3*s3);
        SD( 1, 0,  2.0*sqrt(3.0)*c*s5 - 6.0*sqrt(3.0)*c3*s3 + 2.0*sqrt(3.0)*c5*s);
        SD( 1, 1,  6.0*c2*s4 - 8.0*c4*s2 + c6);
        SD( 1, 2,  2.0*sqrt(10.0)*c3*s3 - sqrt(10.0)*c5*s);
        SD( 1, 3,  sqrt(15.0)*c4*s2);
        SD( 2, 0,  -sqrt(30.0)*c2*s4 + sqrt(30.0)*c4*s2);
        SD( 2, 1,  -2.0*sqrt(10.0)*c3*s3 + sqrt(10.0)*c5*s);
        SD( 2, 2,  -5.0*c4*s2 + c6);
        SD( 2, 3,  -sqrt(6.0)*c5*s);
        SD( 3, 0,  2.0*sqrt(5.0)*c3*s3);
        SD( 3, 1,  sqrt(15.0)*c4*s2);
        SD( 3, 2,  sqrt(6.0)*c5*s);
        SD( 3, 3,  c6);

        #undef SD
    }
}

static void hc_ell4(const double * restrict cv, const double * restrict sv,
                    size_t n1, int ellMax,
                    double * restrict d_buf)
{
    int dim_d = 2 * ellMax + 1;
    for (size_t k = 0; k < n1; ++k) {
        double c = cv[k], s = sv[k];
        double c2 = c*c, s2 = s*s;
        double c3 = c2*c, s3 = s2*s;
        double c4 = c2*c2, s4 = s2*s2;
        double c5 = c4*c, s5 = s4*s;
        double c6 = c3*c3, s6 = s3*s3;
        double c7 = c6*c, s7 = s6*s;
        double c8 = c4*c4, s8 = s4*s4;

        #define SD(m, mp, val) d_buf[(size_t)((m)+ellMax)*(size_t)dim_d*n1 + (size_t)((mp)+ellMax)*n1 + k] = (val)

        SD(-4, 4,  s8);
        SD(-3, 3,  -s8 + 7.0*c2*s6);
        SD(-3, 4,  -2.0*sqrt(2.0)*c*s7);
        SD(-2, 2,  s8 - 12.0*c2*s6 + 15.0*c4*s4);
        SD(-2, 3,  sqrt(14.0)*c*s7 - 3.0*sqrt(14.0)*c3*s5);
        SD(-2, 4,  2.0*sqrt(7.0)*c2*s6);
        SD(-1, 1,  -s8 + 15.0*c2*s6 - 30.0*c4*s4 + 10.0*c6*s2);
        SD(-1, 2,  -3.0*sqrt(2.0)*c*s7 + 15.0*sqrt(2.0)*c3*s5 - 10.0*sqrt(2.0)*c5*s3);
        SD(-1, 3,  -3.0*sqrt(7.0)*c2*s6 + 5.0*sqrt(7.0)*c4*s4);
        SD(-1, 4,  -2.0*sqrt(14.0)*c3*s5);
        SD( 0, 0,  s8 - 16.0*c2*s6 + 36.0*c4*s4 - 16.0*c6*s2 + c8);
        SD( 0, 1,  2.0*sqrt(5.0)*c*s7 - 12.0*sqrt(5.0)*c3*s5 + 12.0*sqrt(5.0)*c5*s3 - 2.0*sqrt(5.0)*c7*s);
        SD( 0, 2,  3.0*sqrt(10.0)*c2*s6 - 8.0*sqrt(10.0)*c4*s4 + 3.0*sqrt(10.0)*c6*s2);
        SD( 0, 3,  2.0*sqrt(35.0)*c3*s5 - 2.0*sqrt(35.0)*c5*s3);
        SD( 0, 4,  sqrt(70.0)*c4*s4);
        SD( 1, 0,  -2.0*sqrt(5.0)*c*s7 + 12.0*sqrt(5.0)*c3*s5 - 12.0*sqrt(5.0)*c5*s3 + 2.0*sqrt(5.0)*c7*s);
        SD( 1, 1,  -10.0*c2*s6 + 30.0*c4*s4 - 15.0*c6*s2 + c8);
        SD( 1, 2,  -10.0*sqrt(2.0)*c3*s5 + 15.0*sqrt(2.0)*c5*s3 - 3.0*sqrt(2.0)*c7*s);
        SD( 1, 3,  -5.0*sqrt(7.0)*c4*s4 + 3.0*sqrt(7.0)*c6*s2);
        SD( 1, 4,  -2.0*sqrt(14.0)*c5*s3);
        SD( 2, 0,  3.0*sqrt(10.0)*c2*s6 - 8.0*sqrt(10.0)*c4*s4 + 3.0*sqrt(10.0)*c6*s2);
        SD( 2, 1,  10.0*sqrt(2.0)*c3*s5 - 15.0*sqrt(2.0)*c5*s3 + 3.0*sqrt(2.0)*c7*s);
        SD( 2, 2,  15.0*c4*s4 - 12.0*c6*s2 + c8);
        SD( 2, 3,  3.0*sqrt(14.0)*c5*s3 - sqrt(14.0)*c7*s);
        SD( 2, 4,  2.0*sqrt(7.0)*c6*s2);
        SD( 3, 0,  -2.0*sqrt(35.0)*c3*s5 + 2.0*sqrt(35.0)*c5*s3);
        SD( 3, 1,  -5.0*sqrt(7.0)*c4*s4 + 3.0*sqrt(7.0)*c6*s2);
        SD( 3, 2,  -3.0*sqrt(14.0)*c5*s3 + sqrt(14.0)*c7*s);
        SD( 3, 3,  -7.0*c6*s2 + c8);
        SD( 3, 4,  -2.0*sqrt(2.0)*c7*s);
        SD( 4, 0,  sqrt(70.0)*c4*s4);
        SD( 4, 1,  2.0*sqrt(14.0)*c5*s3);
        SD( 4, 2,  2.0*sqrt(7.0)*c6*s2);
        SD( 4, 3,  2.0*sqrt(2.0)*c7*s);
        SD( 4, 4,  c8);

        #undef SD
    }
}

/* ====================================================================
 * wignerD_matrices
 * ====================================================================
 *
 * Compute Wigner D-matrices D^ell_{m,mp}(q) for ell = 2..ellMax at N
 * quaternion time-samples.  Uses hardcoded polynomial formulas for
 * ell = 2, 3, 4 (via hc_ell2/3/4) and falls back to Horner evaluation
 * with three-term ell-recurrence for ell > 4.
 *
 * Optionally performs a fused matrix-vector multiply for waveform
 * rotation when h_data and out_data are non-NULL.
 *
 * Parameters
 * ----------
 * q        : const double *, shape (4, N), C-contiguous.
 *            Unit-quaternion time series [q0, q1, q2, q3] with rows
 *            indexed as q[component * N + time].
 * n        : size_t.  Number of time samples N.
 * ellMax   : int >= 2.  Maximum angular momentum quantum number.
 *            Matrices are computed for ell = 2, 3, ..., ellMax.
 * matrices : double complex **, length (ellMax - 1).
 *            Pre-allocated output arrays.  matrices[i] has shape
 *            (2*ell+1, 2*ell+1, N) for ell = i + 2, stored as
 *            mat[row * dim * N + col * N + k].
 * h_data   : const double complex * (optional, may be NULL).
 *            Coorbital mode data for fused rotation, packed as
 *            h_data[mode_offset + m_idx * N + k].  When non-NULL,
 *            the function multiplies D^ell @ h for each (ell, time)
 *            and writes the result to out_data.
 * out_data : double complex * (optional, may be NULL).
 *            Output buffer for the fused rotation result, same layout
 *            as h_data.  Ignored when h_data is NULL.
 *
 * Returns
 * -------
 *  0 on success, -1 on invalid arguments or allocation failure.
 *
 * Algorithm
 * ---------
 * 1. Quaternion → Cayley-Klein:  a = q0 + i*q3,  b = q2 + i*q1.
 *
 * 2. Classify each time point into one of three cases:
 *      general  (|a|^2 >= 1e-24 and |b|^2 >= 1e-24),
 *      edge-a   (|a|^2 < 1e-24, beta ≈ pi),
 *      edge-b   (|b|^2 < 1e-24, beta ≈ 0).
 *    Edge cases are handled via pure-phase power expansions.
 *
 * 3. For the general case, decompose a = |a| * ua,  b = |b| * ub
 *    where ua, ub are unit complex numbers, and precompute:
 *      - cv_pw[p], sv_pw[p]  : real power tables |a|^p, |b|^p
 *      - ua_pw[p], ub_pw[p]  : unit-phase tables ua^p, ub^p
 *      - R = |b|^2/|a|^2     : ratio for Horner evaluation
 *      - cos_beta = |a|^2 - |b|^2  : for ell-recurrence
 *
 * 4. For each ell, compute the real reduced d-matrix d_real(|a|,|b|)
 *    in the fundamental domain (mp >= 0, m + mp >= 0):
 *      ell <= 4 : hardcoded polynomial formulas (hc_ell2/3/4)
 *      ell >  4 : Horner evaluation in R = s^2/c^2 for boundary (m,mp)
 *                 pairs, then three-term ell-recurrence for interior pairs.
 *
 * 5. Expand to the full D-matrix via 4-fold symmetry:
 *      D^ell_{m,mp} = d_real(m,mp) * ua^{m+mp} * ub^{m-mp}
 *    with sign factors (-1)^{m-mp} for the transposed positions.
 *
 * 6. (Optional) If h_data is non-NULL, accumulate D @ h products
 *    during the ell loop to avoid a separate rotation pass.
 *
 * Memory
 * ------
 * A single malloc provides all scratch space via a bump allocator.
 * Includes: log-factorial table, Cayley-Klein arrays, index arrays,
 * power tables, d-matrix recurrence buffers.  Freed before return.
 *
 * See also: wignerD_matrices_opt (same algorithm, no hardcoded ell<=4),
 *           hc_ell2, hc_ell3, hc_ell4 (hardcoded polynomial helpers).
 * ====================================================================
 */
int wignerD_matrices(const double * restrict q, size_t n, int ellMax,
                     double complex ** restrict matrices,
                     const double complex *h_data,
                     double complex *out_data)
{
    if (!q || !matrices || ellMax < 2 || n == 0) return -1;

    int NL = ellMax - 1, P = 2 * ellMax, PC = 2 * P + 1;
    int lf_size = 2 * ellMax + 2;

    /* ---- Compute exact capacity via sizing pass ---- */
    int QP = 4 * ellMax + 1;
    BumpSizer S = {0, 0};

    BUMP_NEED(S, double, lf_size);              /* lf */
    BUMP_NEED(S, double complex, n);            /* ra_f */
    BUMP_NEED(S, double complex, n);            /* rb_f */
    BUMP_NEED(S, size_t, 3 * n);               /* idx */

    BumpSizer saved_S = bump_sizer_save(&S);

    /* Edge-case branch */
    BUMP_NEED(S, double complex, (size_t)PC * n);  /* cpw */
    BUMP_NEED(S, double complex, n);                /* cinv */

    /* General-case branch (rewind, may overlap edge-case region) */
    bump_sizer_restore(&S, saved_S);
    BUMP_NEED(S, double complex, n);                /* ua */
    BUMP_NEED(S, double complex, n);                /* ub */
    BUMP_NEED(S, double complex, n);                /* ua_inv */
    BUMP_NEED(S, double complex, n);                /* ub_inv */
    BUMP_NEED(S, double, n);                        /* c2 */
    BUMP_NEED(S, double, n);                        /* s2 */
    BUMP_NEED(S, double, n);                        /* cv_arr */
    BUMP_NEED(S, double, n);                        /* sv_arr */
    BUMP_NEED(S, double, n);                        /* R */
    BUMP_NEED(S, double, n);                        /* cos_b */
    BUMP_NEED(S, double, (size_t)QP * n);           /* cv_pw */
    BUMP_NEED(S, double, (size_t)QP * n);           /* sv_pw */
    BUMP_NEED(S, double complex, (size_t)PC * n);   /* ua_pw */
    BUMP_NEED(S, double complex, (size_t)PC * n);   /* ub_pw */
    int dim_rec = 2 * ellMax + 1;
    size_t rec_sz = (size_t)dim_rec * (size_t)dim_rec * n;
    BUMP_NEED(S, double, rec_sz);                   /* d_prev */
    BUMP_NEED(S, double, rec_sz);                   /* d_prev2 */

    size_t cap = S.high_water;

    /* Allocate cap + 63 bytes so we can 64-byte-align the usable region.
     * The BumpSizer assumes offset 0 is 64-byte aligned, but malloc may
     * return a pointer with any alignment.  By over-allocating and starting
     * the Bump arena from the first 64-byte-aligned address, the actual
     * layout matches what the sizer computed.  We keep the original malloc
     * pointer (mem) for free(). */
    char *mem = malloc(cap + 63);
    if (!mem) return -1;
    char *base = (char *)(((size_t)mem + 63) & ~(size_t)63);
    Bump B = { base, base + cap };

    /* ---- Log-factorial table ---- */
    double *lf = BUMP(B, double, lf_size);
    if (!lf) { free(mem); return -1; }
    lf[0] = 0.0;
    for (int i = 1; i < lf_size; ++i) lf[i] = lf[i-1] + log((double)i);

    /* ---- Classify time points ---- */
    double complex *ra_f = BUMP(B, double complex, n);
    double complex *rb_f = BUMP(B, double complex, n);
    size_t         *idx  = BUMP(B, size_t, 3 * n);
    if (!ra_f || !rb_f || !idx) { free(mem); return -1; }
    size_t *i1 = idx, *i2 = idx + n, *i3 = idx + 2*n;

    const double *q0=q, *q1=q+n, *q2=q+2*n, *q3=q+3*n;
    size_t n1=0, n2=0, n3=0;
    for (size_t j = 0; j < n; ++j) {
        double complex a = q0[j] + I*q3[j], b = q2[j] + I*q1[j];
        ra_f[j] = a; rb_f[j] = b;
        double a2 = creal(a)*creal(a) + cimag(a)*cimag(a);
        double b2 = creal(b)*creal(b) + cimag(b)*cimag(b);
        if      (a2 < 1e-24) i2[n2++] = j;
        else if (b2 < 1e-24) i3[n3++] = j;
        else                  i1[n1++] = j;
    }

    /* ---- Zero matrices when edge cases present ---- */
    if (n2 > 0 || n3 > 0) {
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t bytes = (size_t)(2*ell+1) * (size_t)(2*ell+1) * n
                           * sizeof(double complex);
            memset(matrices[i], 0, bytes);
        }
    }

    /* ---- Edge cases ---- */
    char *saved_ptr = B.ptr;

    double complex *cpw  = BUMP(B, double complex, (size_t)PC * n);
    double complex *cinv = BUMP(B, double complex, n);
    if (!cpw || !cinv) { free(mem); return -1; }

    #define DO_EDGE_HC4(ni, ia, base_f, row, col, sgn)                   \
    if (ni > 0) {                                                        \
        for (size_t k = 0; k < ni; ++k) cinv[k] = 1.0 / base_f[ia[k]]; \
        for (size_t k = 0; k < ni; ++k) cpw[(size_t)P*ni+k] = 1.0;     \
        for (int p = 1; p <= P; ++p) {                                   \
            size_t c=(size_t)(P+p)*ni, pv=c-ni;                          \
            size_t d=(size_t)(P-p)*ni, dv=d+ni;                          \
            for (size_t k = 0; k < ni; ++k) {                            \
                cpw[c+k] = cpw[pv+k] * base_f[ia[k]];                   \
                cpw[d+k] = cpw[dv+k] * cinv[k];                         \
            }                                                             \
        }                                                                 \
        for (int i = 0; i < NL; ++i) {                                   \
            int ell = i+2; size_t dim = (size_t)(2*ell+1);               \
            for (int m = -ell; m <= ell; ++m) {                           \
                double complex *src = cpw + (size_t)(P+2*m)*ni;          \
                double complex *dst = matrices[i]                         \
                    + (size_t)(row)*dim*n + (size_t)(col)*n;             \
                double s = sgn;                                           \
                for (size_t k = 0; k < ni; ++k) dst[ia[k]] = s*src[k];  \
            }                                                             \
        }                                                                 \
    }

    DO_EDGE_HC4(n2, i2, rb_f, ell+m, ell-m, ((ell+m)&1 ? 1.0 : -1.0))
    DO_EDGE_HC4(n3, i3, ra_f, ell+m, ell+m, 1.0)
    #undef DO_EDGE_HC4

    /* ---- General case (i1) ---- */
    if (n1 > 0) {
        B.ptr = saved_ptr;

        double complex *ua     = BUMP(B, double complex, n1);
        double complex *ub     = BUMP(B, double complex, n1);
        double complex *ua_inv = BUMP(B, double complex, n1);
        double complex *ub_inv = BUMP(B, double complex, n1);
        double *c2     = BUMP(B, double, n1);
        double *s2     = BUMP(B, double, n1);
        double *cv_arr = BUMP(B, double, n1);  /* sqrt(c2) = |ra| */
        double *sv_arr = BUMP(B, double, n1);  /* sqrt(s2) = |rb| */
        double *R      = BUMP(B, double, n1);
        double *cos_b  = BUMP(B, double, n1);
        double *cv_pw  = BUMP(B, double, (size_t)QP * n1);  /* |ra|^p */
        double *sv_pw  = BUMP(B, double, (size_t)QP * n1);  /* |rb|^p */
        double complex *ua_pw = BUMP(B, double complex, (size_t)PC * n1);
        double complex *ub_pw = BUMP(B, double complex, (size_t)PC * n1);

        int dim_rec = 2 * ellMax + 1;
        size_t rec_sz = (size_t)dim_rec * (size_t)dim_rec * n1;
        double *d_prev  = BUMP(B, double, rec_sz);
        double *d_prev2 = BUMP(B, double, rec_sz);

        if (!ua||!ub||!ua_inv||!ub_inv||!c2||!s2||!cv_arr||!sv_arr||
            !R||!cos_b||!cv_pw||!sv_pw||!ua_pw||!ub_pw||!d_prev||!d_prev2) {
            free(mem); return -1;
        }

        for (size_t k = 0; k < n1; ++k) {
            double complex a = ra_f[i1[k]], b = rb_f[i1[k]];
            double re_a = creal(a), im_a = cimag(a);
            double re_b = creal(b), im_b = cimag(b);
            double a2 = re_a*re_a + im_a*im_a;
            double b2 = re_b*re_b + im_b*im_b;
            c2[k] = a2;
            s2[k] = b2;
            cv_arr[k] = sqrt(a2);
            sv_arr[k] = sqrt(b2);
            R[k]  = b2 / a2;
            cos_b[k] = a2 - b2;
            double complex u_a = a / cv_arr[k];
            double complex u_b = b / sv_arr[k];
            ua[k] = u_a;
            ub[k] = u_b;
            ua_inv[k] = conj(u_a);
            ub_inv[k] = conj(u_b);
        }

        build_real_pows(cv_arr, n1, QP - 1, cv_pw);
        build_real_pows(sv_arr, n1, QP - 1, sv_pw);
        build_unit_pows(ua, ua_inv, n1, P, ua_pw);
        build_unit_pows(ub, ub_inv, n1, P, ub_pw);

        /* ---- Hot loop over ell ---- */
        size_t mode_offset = 0;
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t dim = (size_t)(2*ell+1);
            double complex * restrict mat = matrices[i];

            /* For ell <= 4: use hardcoded formulas to fill d_prev2 buffer */
            int used_hardcoded = 0;
            if (ell == 2) {
                hc_ell2(cv_arr, sv_arr, n1, ellMax, d_prev2);
                used_hardcoded = 1;
            } else if (ell == 3) {
                hc_ell3(cv_arr, sv_arr, n1, ellMax, d_prev2);
                used_hardcoded = 1;
            } else if (ell == 4) {
                hc_ell4(cv_arr, sv_arr, n1, ellMax, d_prev2);
                used_hardcoded = 1;
            }

            if (used_hardcoded) {
                /* Write from d_prev2 buffer to matrix using phases and symmetry */
                for (int mp = 0; mp <= ell; ++mp) {
                    int m_start = (mp > 0) ? -mp : 0;
                    for (int m = m_start; m <= ell; ++m) {
                        size_t rec_idx = (size_t)(m + ellMax) * (size_t)dim_rec * n1
                                       + (size_t)(mp + ellMax) * n1;

                        const double complex *ph_a = ua_pw + (size_t)(P + m + mp) * n1;
                        const double complex *ph_b = ub_pw + (size_t)(P + m - mp) * n1;
                        double sym_sign = ((m - mp) & 1) ? -1.0 : 1.0;

                        /* Position 1: (m, mp) */
                        {
                            double complex *dst = mat + (size_t)(ell+m)*dim*n + (size_t)(ell+mp)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = d_prev2[rec_idx + k] * ph_a[k] * ph_b[k];
                        }
                        /* Position 2: (mp, m) */
                        if (mp != m) {
                            const double complex *ph_b2 = ub_pw + (size_t)(P + mp - m) * n1;
                            double complex *dst = mat + (size_t)(ell+mp)*dim*n + (size_t)(ell+m)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a[k] * ph_b2[k];
                        }
                        /* Position 3: (-m, -mp) */
                        if (m > 0 || mp > 0) {
                            const double complex *ph_a3 = ua_pw + (size_t)(P - m - mp) * n1;
                            const double complex *ph_b3 = ub_pw + (size_t)(P - m + mp) * n1;
                            double complex *dst = mat + (size_t)(ell-m)*dim*n + (size_t)(ell-mp)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a3[k] * ph_b3[k];
                        }
                        /* Position 4: (-mp, -m) */
                        if (mp != m && (m > 0 || mp > 0)) {
                            const double complex *ph_a4 = ua_pw + (size_t)(P - m - mp) * n1;
                            double complex *dst = mat + (size_t)(ell-mp)*dim*n + (size_t)(ell-m)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = d_prev2[rec_idx + k] * ph_a4[k] * ph_b[k];
                        }
                    }
                }
            } else {
                /* ell > 4: general Horner + recurrence (same as wignerD_matrices_opt) */
                for (int mp = 0; mp <= ell; ++mp) {
                    int m_start = (mp > 0) ? -mp : 0;
                    for (int m = m_start; m <= ell; ++m) {
                        int use_recurrence = (ell >= 4 && abs(m) < ell-1 && abs(mp) < ell-1);
                        size_t rec_idx = (size_t)(m + ellMax) * (size_t)dim_rec * n1
                                       + (size_t)(mp + ellMax) * n1;

                        if (use_recurrence) {
                            double Ac, Bc, Cc;
                            recurrence_abc(ell, m, mp, &Ac, &Bc, &Cc);
                            double inv_C = 1.0 / Cc;
                            double mmp = -(2.0 * ell - 1.0) * (double)m * (double)mp;
                            size_t prev_idx = rec_idx;
                            for (size_t k = 0; k < n1; ++k) {
                                double d_new = ((Ac * cos_b[k] + mmp) * d_prev[prev_idx + k]
                                               + Bc * d_prev2[prev_idx + k]) * inv_C;
                                d_prev2[rec_idx + k] = d_new;
                            }
                        } else {
                            double coef = wigner_coef2(lf, ell, mp, m);
                            int rhoMin = (mp - m > 0) ? (mp - m) : 0;
                            int rhoMax = ell + mp;
                            if (ell - m < rhoMax) rhoMax = ell - m;
                            int nr = rhoMax - rhoMin + 1;
                            double rc_arr[32];
                            for (int r = 0; r < nr; ++r)
                                rc_arr[r] = rho_coeff(lf, ell, mp, m, rhoMin + r);
                            int c_exp = 2*ell - m + mp - 2*rhoMin;
                            int s_exp = m - mp + 2*rhoMin;
                            const double *cv_row = cv_pw + (size_t)c_exp * n1;
                            const double *sv_row = sv_pw + (size_t)s_exp * n1;
                            for (size_t k = 0; k < n1; ++k) {
                                double Rv = R[k];
                                double h = rc_arr[nr-1];
                                for (int r = nr-2; r >= 0; --r)
                                    h = h * Rv + rc_arr[r];
                                d_prev2[rec_idx + k] = coef * h * cv_row[k] * sv_row[k];
                            }
                        }

                        /* Write to matrix with phases and symmetry */
                        const double complex *ph_a = ua_pw + (size_t)(P + m + mp) * n1;
                        const double complex *ph_b = ub_pw + (size_t)(P + m - mp) * n1;
                        double sym_sign = ((m - mp) & 1) ? -1.0 : 1.0;

                        {
                            double complex *dst = mat + (size_t)(ell+m)*dim*n + (size_t)(ell+mp)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = d_prev2[rec_idx + k] * ph_a[k] * ph_b[k];
                        }
                        if (mp != m) {
                            const double complex *ph_b2 = ub_pw + (size_t)(P + mp - m) * n1;
                            double complex *dst = mat + (size_t)(ell+mp)*dim*n + (size_t)(ell+m)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a[k] * ph_b2[k];
                        }
                        if (m > 0 || mp > 0) {
                            const double complex *ph_a3 = ua_pw + (size_t)(P - m - mp) * n1;
                            const double complex *ph_b3 = ub_pw + (size_t)(P - m + mp) * n1;
                            double complex *dst = mat + (size_t)(ell-m)*dim*n + (size_t)(ell-mp)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a3[k] * ph_b3[k];
                        }
                        if (mp != m && (m > 0 || mp > 0)) {
                            const double complex *ph_a4 = ua_pw + (size_t)(P - m - mp) * n1;
                            double complex *dst = mat + (size_t)(ell-mp)*dim*n + (size_t)(ell-m)*n;
                            for (size_t k = 0; k < n1; ++k)
                                dst[i1[k]] = d_prev2[rec_idx + k] * ph_a4[k] * ph_b[k];
                        }
                    }
                }
            }

            /* Fused matmul: reordered loop for SIMD auto-vectorization.
             * Inner loop over t (stride-1) enables SSE/AVX on D and h.
             */
            if (h_data) {
                const double complex *h_base = h_data + mode_offset * n;
                double complex *out_base = out_data + mode_offset * n;
                for (int row = 0; row < (int)dim; row++) {
                    double complex *out_row = out_base + (size_t)row * n;
                    /* Zero the output row */
                    for (size_t t = 0; t < n; t++)
                        out_row[t] = 0;
                    /* Accumulate D[row,col,t] * h[col,t] */
                    for (int col = 0; col < (int)dim; col++) {
                        const double complex *D_rc = mat
                            + (size_t)row * dim * n + (size_t)col * n;
                        const double complex *h_col = h_base + (size_t)col * n;
                        for (size_t t = 0; t < n; t++)
                            out_row[t] += D_rc[t] * h_col[t];
                    }
                }
                mode_offset += (size_t)dim;
            }

            /* Rotate recurrence buffers */
            double *tmp = d_prev;
            d_prev = d_prev2;
            d_prev2 = tmp;
        }
    }

    /* Fallback matmul when n1==0 (all edge cases, no general path ran) */
    if (h_data && n1 == 0) {
        size_t mode_offset = 0;
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t dim = (size_t)(2*ell+1);
            const double complex *D = matrices[i];
            const double complex *h_base = h_data + mode_offset * n;
            double complex *out_base = out_data + mode_offset * n;
            for (int row = 0; row < (int)dim; row++) {
                double complex *out_row = out_base + (size_t)row * n;
                for (size_t t = 0; t < n; t++)
                    out_row[t] = 0;
                for (int col = 0; col < (int)dim; col++) {
                    const double complex *D_rc = D
                        + (size_t)row * dim * n + (size_t)col * n;
                    const double complex *h_col = h_base + (size_t)col * n;
                    for (size_t t = 0; t < n; t++)
                        out_row[t] += D_rc[t] * h_col[t];
                }
            }
            mode_offset += dim;
        }
    }

    free(mem);
    return 0;
}

PyObject *py_wignerD_matrices(PyObject *self, PyObject *args)
{
    PyArrayObject *q_obj;
    PyObject *mat_list;
    int ellMax;

    if (!PyArg_ParseTuple(args, "O!iO!",
            &PyArray_Type, &q_obj,
            &ellMax,
            &PyList_Type, &mat_list))
        return NULL;

    if (PyArray_NDIM(q_obj) != 2 || PyArray_DIM(q_obj, 0) != 4) {
        PyErr_SetString(PyExc_ValueError, "q must have shape (4, N)");
        return NULL;
    }
    if (ellMax < 2) {
        PyErr_SetString(PyExc_ValueError, "ellMax must be >= 2");
        return NULL;
    }

    int num_ells = ellMax - 1;
    if (PyList_GET_SIZE(mat_list) != num_ells) {
        PyErr_Format(PyExc_ValueError,
                     "matrices list must have length %d", num_ells);
        return NULL;
    }

    PyArrayObject *q_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)q_obj, NPY_DOUBLE, 2, 2);
    if (!q_arr) return NULL;

    npy_intp n = PyArray_DIM(q_arr, 1);
    const double *q_data = (const double *)PyArray_DATA(q_arr);

    double complex *mat_ptrs[num_ells];

    for (int i = 0; i < num_ells; ++i) {
        int ell = i + 2;
        int dim = 2 * ell + 1;
        PyObject *item = PyList_GET_ITEM(mat_list, i);

        if (!PyArray_Check(item)) {
            PyErr_Format(PyExc_TypeError,
                         "matrices[%d] is not a numpy array", i);
            Py_DECREF(q_arr);
            return NULL;
        }

        PyArrayObject *mat = (PyArrayObject *)item;

        if (PyArray_NDIM(mat) != 3 ||
            PyArray_DIM(mat, 0) != dim ||
            PyArray_DIM(mat, 1) != dim ||
            PyArray_DIM(mat, 2) != n) {
            PyErr_Format(PyExc_ValueError,
                         "matrices[%d] must have shape (%d, %d, %ld)",
                         i, dim, dim, (long)n);
            Py_DECREF(q_arr);
            return NULL;
        }
        if (PyArray_TYPE(mat) != NPY_COMPLEX128) {
            PyErr_Format(PyExc_TypeError,
                         "matrices[%d] must have dtype complex128", i);
            Py_DECREF(q_arr);
            return NULL;
        }
        if (!PyArray_IS_C_CONTIGUOUS(mat)) {
            PyErr_Format(PyExc_ValueError,
                         "matrices[%d] must be C-contiguous", i);
            Py_DECREF(q_arr);
            return NULL;
        }

        mat_ptrs[i] = (double complex *)PyArray_DATA(mat);
    }

    int ret;
    Py_BEGIN_ALLOW_THREADS
    ret = wignerD_matrices(q_data, (size_t)n, ellMax, mat_ptrs,
                           NULL, NULL);
    Py_END_ALLOW_THREADS

    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
}


/* ------------------------------------------------------------------ */
/* eval_coorb_modes: fused __call__ + _eval_comp for                  */
/*   CoorbitalWaveformSurrogate                                       */
/* ------------------------------------------------------------------ */

/*
 * Evaluates all coorbital waveform mode components and assembles complex modes.
 * Replaces the Python __call__ + _eval_comp loops with a single C call.
 *
 * Arguments:
 *   q:                  double, mass ratio
 *   chiA:               (N, 3) float64, spin A time series
 *   chiB:               (N, 3) float64, spin B time series
 *   comp_n_nodes:       (n_comps,) int32, number of nodes per component
 *   comp_node_offset:   (n_comps,) int32, offset into flat node arrays
 *   all_node_indices:   (total_nodes,) int32, concatenated nodeIndices
 *   node_n_coefs:       (total_nodes,) int32, number of coefs per node
 *   node_coef_offset:   (total_nodes,) int32, offset into coefs/orders
 *   all_coefs:          (total_coefs,) float64, concatenated coefficients
 *   all_orders:         (total_coefs, 7) int32, concatenated bf orders
 *   all_EI_basis:       (total_nodes, N_time) float64, stacked EI bases
 *   mode_info:          (n_groups, 8) int32, mode assembly metadata
 *   q_consts:           (5,) float64, q-dependent constants
 *   nmodes:             int, number of output modes
 *   ellMax:             int, max ell to include
 *   fit_params_mode:    int, 0=NRSur7dq4, 1=identity
 *   q_fit_offset:       double
 *   q_fit_slope:        double
 *   q_max_bfOrder:      int
 *   chi_max_bfOrder:    int
 *
 * Returns (nmodes, N_time) complex128 array.
 */
static PyObject *eval_coorb_modes(PyObject *self, PyObject *args) {

    double q_val;
    PyArrayObject *chiA_arr, *chiB_arr;
    PyArrayObject *comp_n_nodes_arr, *comp_node_offset_arr;
    PyArrayObject *all_node_indices_arr, *node_n_coefs_arr, *node_coef_offset_arr;
    PyArrayObject *all_coefs_arr, *all_orders_arr, *all_EI_basis_arr;
    PyArrayObject *mode_info_arr, *q_consts_arr;
    int nmodes, ellMax_eval, fit_params_mode;
    double q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;

    if (!PyArg_ParseTuple(args, "dO!O!O!O!O!O!O!O!O!O!O!O!iiiddii",
            &q_val,
            &PyArray_Type, &chiA_arr,
            &PyArray_Type, &chiB_arr,
            &PyArray_Type, &comp_n_nodes_arr,
            &PyArray_Type, &comp_node_offset_arr,
            &PyArray_Type, &all_node_indices_arr,
            &PyArray_Type, &node_n_coefs_arr,
            &PyArray_Type, &node_coef_offset_arr,
            &PyArray_Type, &all_coefs_arr,
            &PyArray_Type, &all_orders_arr,
            &PyArray_Type, &all_EI_basis_arr,
            &PyArray_Type, &mode_info_arr,
            &PyArray_Type, &q_consts_arr,
            &nmodes, &ellMax_eval, &fit_params_mode,
            &q_fit_offset, &q_fit_slope,
            &q_max_bfOrder, &chi_max_bfOrder)) return NULL;

    /* Extract data pointers */
    double *chiA = (double *)PyArray_DATA(chiA_arr);
    double *chiB = (double *)PyArray_DATA(chiB_arr);
    npy_int32 *comp_n_nodes = (npy_int32 *)PyArray_DATA(comp_n_nodes_arr);
    npy_int32 *comp_node_offset = (npy_int32 *)PyArray_DATA(comp_node_offset_arr);
    npy_int32 *all_node_indices = (npy_int32 *)PyArray_DATA(all_node_indices_arr);
    npy_int32 *node_n_coefs = (npy_int32 *)PyArray_DATA(node_n_coefs_arr);
    npy_int32 *node_coef_offset = (npy_int32 *)PyArray_DATA(node_coef_offset_arr);
    double *all_coefs = (double *)PyArray_DATA(all_coefs_arr);
    npy_int32 *all_orders = (npy_int32 *)PyArray_DATA(all_orders_arr);
    double *all_EI_basis = (double *)PyArray_DATA(all_EI_basis_arr);
    npy_int32 *mode_info = (npy_int32 *)PyArray_DATA(mode_info_arr);
    double *q_consts = (double *)PyArray_DATA(q_consts_arr);

    int n_comps = (int)PyArray_DIMS(comp_n_nodes_arr)[0];
    int n_groups = (int)PyArray_DIMS(mode_info_arr)[0];
    int N_time = (int)PyArray_DIMS(all_EI_basis_arr)[1];

    /* Pre-compute q-dependent powers (same for all nodes) */
    double q_transformed = (fit_params_mode == 0) ? q_consts[0] : q_val;
    double q_powers[16];  /* q_max_bfOrder <= 15 */
    for (int i = 0; i <= q_max_bfOrder; i++) {
        q_powers[i] = ipow(q_fit_offset + q_fit_slope * q_transformed, i);
    }

    /* Allocate output: (nmodes, N_time) complex128, zero-initialized */
    npy_intp out_dims[2] = {nmodes, N_time};
    PyArrayObject *modes_arr = (PyArrayObject *)PyArray_ZEROS(
            2, out_dims, NPY_COMPLEX128, 0);
    if (!modes_arr) return NULL;
    double *modes = (double *)PyArray_DATA(modes_arr);

    /* Allocate workspace for component results */
    double *comp_results = (double *)malloc(
            (size_t)n_comps * (size_t)N_time * sizeof(double));
    if (!comp_results) {
        Py_DECREF(modes_arr);
        return PyErr_NoMemory();
    }

    int chi_pw_size = 6 * (chi_max_bfOrder + 1);

    /* ---- Phase 1: Evaluate all components ---- */
    for (int c = 0; c < n_comps; c++) {
        int nn = comp_n_nodes[c];
        int offset = comp_node_offset[c];
        double nodes_buf[128];  /* max nodes per component */

        for (int j = 0; j < nn; j++) {
            int gj = offset + j;
            int ni = all_node_indices[gj];

            /* Build x[7] from q, chiA[ni], chiB[ni] */
            double x[7];
            x[0] = q_val;
            x[1] = chiA[ni * 3 + 0];
            x[2] = chiA[ni * 3 + 1];
            x[3] = chiA[ni * 3 + 2];
            x[4] = chiB[ni * 3 + 0];
            x[5] = chiB[ni * 3 + 1];
            x[6] = chiB[ni * 3 + 2];

            /* Apply fit_params transform */
            if (fit_params_mode == 0) {
                /* NRSur7dq4: x[0]=log(q), x[3]=chiHat, x[6]=chi_a */
                double chi1z = x[3], chi2z = x[6];
                x[0] = q_consts[0];  /* log(q) */
                double chi_wtAvg = q_consts[1]*chi1z + q_consts[2]*chi2z;
                x[3] = (chi_wtAvg - q_consts[3]*(chi1z + chi2z))
                        / q_consts[4];
                x[6] = (chi1z - chi2z) * 0.5;
            }
            /* mode == 1: identity, x unchanged */

            /* Compute chi_powers (q_powers already done) */
            double chi_powers[42];  /* 6*(max_bfOrder+1), max ~6*7=42 */
            for (int p = 0; p <= chi_max_bfOrder; p++) {
                for (int k = 0; k < 6; k++) {
                    chi_powers[k * (chi_max_bfOrder + 1) + p] =
                            ipow(x[k + 1], p);
                }
            }

            /* Evaluate polynomial */
            int nc = node_n_coefs[gj];
            int co = node_coef_offset[gj];
            double res = 0.0;
            for (int i = 0; i < nc; i++) {
                npy_int32 *ord = all_orders + (co + i) * 7;
                double prod = q_powers[ord[0]];
                for (int k = 0; k < 6; k++) {
                    prod *= chi_powers[k*(chi_max_bfOrder+1) + ord[k+1]];
                }
                res += all_coefs[co + i] * prod;
            }
            nodes_buf[j] = res;
        }

        /* Dot product: comp_results[c] = nodes . EI_basis */
        double *out = comp_results + (size_t)c * N_time;
        memset(out, 0, (size_t)N_time * sizeof(double));
        for (int j = 0; j < nn; j++) {
            double nj = nodes_buf[j];
            double *basis = all_EI_basis + (size_t)(offset + j) * N_time;
            for (int t = 0; t < N_time; t++) {
                out[t] += nj * basis[t];
            }
        }
    }

    /* ---- Phase 2: Assemble complex modes ---- */
    for (int g = 0; g < n_groups; g++) {
        npy_int32 *info = mode_info + g * 8;
        int ell = info[0];
        int m   = info[1];

        if (ell > ellMax_eval) continue;

        if (m == 0) {
            int mode_idx = info[2];
            int comp_re  = info[3];
            int comp_im  = info[4];
            double *re = comp_results + (size_t)comp_re * N_time;
            double *im = comp_results + (size_t)comp_im * N_time;
            double *dst = modes + (size_t)mode_idx * N_time * 2;
            for (int t = 0; t < N_time; t++) {
                dst[t * 2]     = re[t];
                dst[t * 2 + 1] = im[t];
            }
        } else {
            int idx_pos  = info[2];
            int idx_neg  = info[3];
            int comp_rep = info[4];
            int comp_rem = info[5];
            int comp_imp = info[6];
            int comp_imm = info[7];
            double *rep = comp_results + (size_t)comp_rep * N_time;
            double *rem_d = comp_results + (size_t)comp_rem * N_time;
            double *imp = comp_results + (size_t)comp_imp * N_time;
            double *imm = comp_results + (size_t)comp_imm * N_time;
            double *pos = modes + (size_t)idx_pos * N_time * 2;
            double *neg = modes + (size_t)idx_neg * N_time * 2;
            for (int t = 0; t < N_time; t++) {
                pos[t * 2]     = rep[t] - rem_d[t];
                pos[t * 2 + 1] = imm[t] - imp[t];
                neg[t * 2]     = rep[t] + rem_d[t];
                neg[t * 2 + 1] = imp[t] + imm[t];
            }
        }
    }

    free(comp_results);
    return PyArray_Return(modes_arr);
}


/* ------------------------------------------------------------------ */
/* rotate_waveform: quaternion inverse + wignerD + fused matmul in C  */
/* ------------------------------------------------------------------ */

static PyObject *py_rotate_waveform(PyObject *self, PyObject *args)
{
    PyArrayObject *q_obj, *h_obj, *out_obj;
    int ellMax;

    if (!PyArg_ParseTuple(args, "O!O!iO!",
            &PyArray_Type, &q_obj,
            &PyArray_Type, &h_obj,
            &ellMax,
            &PyArray_Type, &out_obj))
        return NULL;

    /* Validate q: shape (4, N), float64, C-contiguous */
    if (PyArray_NDIM(q_obj) != 2 || PyArray_DIM(q_obj, 0) != 4) {
        PyErr_SetString(PyExc_ValueError, "q must have shape (4, N)");
        return NULL;
    }
    npy_intp N = PyArray_DIM(q_obj, 1);

    /* Determine expected n_modes */
    int num_ells = ellMax - 1;
    npy_intp n_modes = 0;
    for (int ell = 2; ell <= ellMax; ell++)
        n_modes += 2 * ell + 1;

    /* Validate h: shape (n_modes, N), complex128, C-contiguous */
    if (PyArray_NDIM(h_obj) != 2 ||
        PyArray_DIM(h_obj, 0) != n_modes ||
        PyArray_DIM(h_obj, 1) != N) {
        PyErr_Format(PyExc_ValueError,
                     "h must have shape (%ld, %ld)", (long)n_modes, (long)N);
        return NULL;
    }
    if (PyArray_TYPE(h_obj) != NPY_COMPLEX128 || !PyArray_IS_C_CONTIGUOUS(h_obj)) {
        PyErr_SetString(PyExc_TypeError, "h must be C-contiguous complex128");
        return NULL;
    }

    /* Validate out: same shape as h, complex128, C-contiguous */
    if (PyArray_NDIM(out_obj) != 2 ||
        PyArray_DIM(out_obj, 0) != n_modes ||
        PyArray_DIM(out_obj, 1) != N) {
        PyErr_Format(PyExc_ValueError,
                     "out must have shape (%ld, %ld)", (long)n_modes, (long)N);
        return NULL;
    }
    if (PyArray_TYPE(out_obj) != NPY_COMPLEX128 || !PyArray_IS_C_CONTIGUOUS(out_obj)) {
        PyErr_SetString(PyExc_TypeError, "out must be C-contiguous complex128");
        return NULL;
    }

    /* Make a mutable copy of q for in-place quaternion inverse */
    PyArrayObject *q_tmp = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)q_obj, NPY_DOUBLE, 2, 2);
    if (!q_tmp) return NULL;
    PyArrayObject *q_arr = (PyArrayObject *)PyArray_NewCopy(q_tmp, NPY_CORDER);
    Py_DECREF(q_tmp);
    if (!q_arr) return NULL;

    double *q_mut = (double *)PyArray_DATA(q_arr);
    const double complex *h_data = (const double complex *)PyArray_DATA(h_obj);
    double complex *out_data = (double complex *)PyArray_DATA(out_obj);

    /* Quaternion inverse in-place: conjugate / |q|^2.
     * For unit quaternions this is just negating components 1,2,3.
     * We include the normalization for robustness. */
    for (npy_intp t = 0; t < N; t++) {
        double q0 = q_mut[t];
        double q1 = q_mut[N + t];
        double q2 = q_mut[2*N + t];
        double q3 = q_mut[3*N + t];
        double inv_norm_sq = 1.0 / (q0*q0 + q1*q1 + q2*q2 + q3*q3);
        q_mut[t]       =  q0 * inv_norm_sq;
        q_mut[N + t]   = -q1 * inv_norm_sq;
        q_mut[2*N + t] = -q2 * inv_norm_sq;
        q_mut[3*N + t] = -q3 * inv_norm_sq;
    }

    /* Allocate D matrices */
    double complex *mat_ptrs[num_ells];
    double complex *mat_mem = NULL;
    size_t total_elems = 0;
    for (int i = 0; i < num_ells; i++) {
        int dim = 2 * (i + 2) + 1;
        total_elems += (size_t)dim * dim * N;
    }
    mat_mem = (double complex *)malloc(total_elems * sizeof(double complex));
    if (!mat_mem) {
        Py_DECREF(q_arr);
        PyErr_NoMemory();
        return NULL;
    }
    {
        size_t offset = 0;
        for (int i = 0; i < num_ells; i++) {
            int dim = 2 * (i + 2) + 1;
            mat_ptrs[i] = mat_mem + offset;
            offset += (size_t)dim * dim * N;
        }
    }

    int ret;
    Py_BEGIN_ALLOW_THREADS

    /* Compute Wigner D matrices with fused matmul (reordered loops) */
    ret = wignerD_matrices(q_mut, (size_t)N, ellMax, mat_ptrs,
                           h_data, out_data);

    Py_END_ALLOW_THREADS

    free(mat_mem);
    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
}
