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
#include <math.h>
#include <stdio.h>

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
    {"wignerD_matrices_opt", py_wignerD_matrices_opt, METH_VARARGS},
    {"wignerD_matrices_opt_hc4", py_wignerD_matrices_opt_hc4, METH_VARARGS},
    {"rotate_waveform", py_rotate_waveform, METH_VARARGS},
    {"eval_coorb_modes", eval_coorb_modes, METH_VARARGS},
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

    if (!PyArg_ParseTuple(args, "OO!O!ddii",
            &fit_list,
            &PyArray_Type, &fit_params,
            &PyArray_Type, &y,
            &q_fit_offset,
            &q_fit_slope,
            &q_max_bfOrder,
            &chi_max_bfOrder)) return NULL;

    double *x_data = (double *) PyArray_DATA(fit_params);
    double *y_data = (double *) PyArray_DATA(y);

    /* Pre-compute x_powers once for all fits */
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

    /* Evaluate 9 fits into stack array */
    double results[9];
    int k, n;
    for (k=0; k<9; k++) {
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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

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

/* ------------------------------------------------------------------ */
/*  Core                                                              */
/* ------------------------------------------------------------------ */

#define W(mat, ell, r, c, t, n) \
    ((mat)[(size_t)(r)*(size_t)(2*(ell)+1)*(n) + (size_t)(c)*(n) + (t)])

int wignerD_matrices(const double *q, size_t n, int ellMax,
                     double complex **matrices)
{
    if (!q || !matrices || ellMax < 2 || n == 0) return -1;

    int NL = ellMax - 1, P = 2 * ellMax, PC = 2*P + 1;
    int lf_size = 2 * ellMax + 2;

    /* ---- Single allocation ---- */
    size_t cap = (size_t)lf_size * sizeof(double)             /* lf table */
      + (size_t)n * (
          2 * sizeof(double complex)                          /* ra_f, rb_f */
        + 3 * sizeof(size_t)                                  /* i1, i2, i3 */
        + 4 * sizeof(double complex)                          /* ra, rb, ia, ib */
        + 2 * (size_t)PC * sizeof(double complex)             /* rapw, rbpw */
        + sizeof(double)                                      /* arSq */
        + (size_t)(P+1) * sizeof(double)                      /* arPw */
        + sizeof(double)                                      /* absR */
        + (size_t)PC * sizeof(double complex)                 /* cpw */
        + sizeof(double complex)                              /* cinv */
      ) + 64 * 24;  /* alignment padding */

    char *mem = malloc(cap);
    if (!mem) return -1;
    Bump B = { mem, mem + cap };

    /* ---- Log-factorial table: local, no global state ---- */
    double *lf = BUMP(B, double, lf_size);
    if (!lf) { free(mem); return -1; }
    lf[0] = 0.0;
    for (int i = 1; i < lf_size; ++i) lf[i] = lf[i-1] + log((double)i);

    /* ---- Workspace arrays ---- */
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

    /* ---- Zero matrices only when edge cases present ---- */
    /* General case (i1) writes every element; edge cases only write one     */
    /* column per (ell,m) pair, so off-diagonal entries must start as zero.  */
    if (n2 > 0 || n3 > 0) {
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t bytes = (size_t)(2*ell+1) * (size_t)(2*ell+1) * n
                           * sizeof(double complex);
            memset(matrices[i], 0, bytes);
        }
    }

    /* ---- Edge cases (i2 / i3): reuse same workspace region ---- */
    /* Save bump state so we can reclaim after edge cases */
    char *saved_ptr = B.ptr;

    double complex *cpw  = BUMP(B, double complex, (size_t)PC * n);
    double complex *cinv = BUMP(B, double complex, n);
    if (!cpw || !cinv) { free(mem); return -1; }

    #define DO_EDGE(ni, ia, base_f, row, col, sgn)                      \
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

    DO_EDGE(n2, i2, rb_f, ell+m, ell-m, ((ell+m)&1 ? 1.0 : -1.0))
    DO_EDGE(n3, i3, ra_f, ell+m, ell+m, 1.0)
    #undef DO_EDGE

    /* ---- General case (i1) ---- */
    if (n1 > 0) {
        /* Reclaim edge-case workspace */
        B.ptr = saved_ptr;

        double complex *ra   = BUMP(B, double complex, n1);
        double complex *rb   = BUMP(B, double complex, n1);
        double complex *ia   = BUMP(B, double complex, n1);
        double complex *ib   = BUMP(B, double complex, n1);
        double complex *rapw = BUMP(B, double complex, (size_t)PC * n1);
        double complex *rbpw = BUMP(B, double complex, (size_t)PC * n1);
        double         *arSq = BUMP(B, double, n1);
        double         *arPw = BUMP(B, double, (size_t)(P+1) * n1);
        double         *absR = BUMP(B, double, n1);
        if (!ra||!rb||!ia||!ib||!rapw||!rbpw||!arSq||!arPw||!absR)
            { free(mem); return -1; }

        for (size_t k = 0; k < n1; ++k) {
            double complex a = ra_f[i1[k]], b = rb_f[i1[k]];
            ra[k]=a; rb[k]=b; ia[k]=1.0/a; ib[k]=1.0/b;
            double re_a=creal(a), im_a=cimag(a);
            double re_b=creal(b), im_b=cimag(b);
            double a2 = re_a*re_a + im_a*im_a;
            arSq[k] = a2;
            absR[k] = (re_b*re_b + im_b*im_b) / a2;
        }

        build_cpows(ra, ia, n1, P, rapw);
        build_cpows(rb, ib, n1, P, rbpw);

        for (size_t k = 0; k < n1; ++k) arPw[k] = 1.0;
        for (int p = 1; p <= P; ++p) {
            size_t c = (size_t)p*n1, pv = c - n1;
            for (size_t k = 0; k < n1; ++k)
                arPw[c+k] = arPw[pv+k] * arSq[k];
        }

        /* ---- Hot loop: Horner's method ---- */
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t dim = (size_t)(2*ell+1);
            double complex *mat = matrices[i];

            for (int m = -ell; m <= ell; ++m) {
                const double *arSq_row = arPw + (size_t)(ell-m)*n1;

                for (int mp = -ell; mp <= ell; ++mp) {
                    double coef = wigner_coef2(lf, ell, mp, m);
                    int rhoMin = (mp-m > 0) ? (mp-m) : 0;
                    int rhoMax = ell+mp;
                    if (ell-m < rhoMax) rhoMax = ell-m;
                    int nr = rhoMax - rhoMin + 1;

                    double rc[nr]; /* VLA — small, stack-allocated */
                    for (int r = 0; r < nr; ++r)
                        rc[r] = rho_coeff(lf, ell, mp, m, rhoMin + r);

                    const double complex *ar =
                        rapw + (size_t)(P + m+mp)*n1;
                    const double complex *br =
                        rbpw + (size_t)(P + m-mp)*n1;
                    double complex *dst = mat
                        + (size_t)(ell+m)*dim*n + (size_t)(ell+mp)*n;

                    for (size_t k = 0; k < n1; ++k) {
                        double R = absR[k];
                        double s = rc[nr-1];
                        for (int r = nr-2; r >= 0; --r)
                            s = s*R + rc[r];
                        if      (rhoMin == 1) s *= R;
                        else if (rhoMin == 2) s *= R*R;
                        else if (rhoMin > 2) {
                            double Rp = R*R;
                            for (int p = 2; p < rhoMin; ++p) Rp *= R;
                            s *= Rp;
                        }
                        dst[i1[k]] = (coef*s)*ar[k]*br[k]*arSq_row[k];
                    }
                }
            }
        }
    }

    free(mem);
    return 0;
}

/* PyObject *py_wignerD_matrices(PyObject *self, PyObject *args) */
/* { */
/*     PyArrayObject *q_obj; */
/*     int ellMax; */

/*     /\* ---- Parse arguments: q (ndarray), ellMax (int) ---- *\/ */
/*     if (!PyArg_ParseTuple(args, "O!i", */
/*             &PyArray_Type, &q_obj, */
/*             &ellMax)) */
/*         return NULL; */

/*     /\* ---- Validate q ---- *\/ */
/*     if (PyArray_NDIM(q_obj) != 2) { */
/*         PyErr_SetString(PyExc_ValueError, "q must be 2-dimensional"); */
/*         return NULL; */
/*     } */
/*     if (PyArray_DIM(q_obj, 0) != 4) { */
/*         PyErr_SetString(PyExc_ValueError, "q must have shape (4, N)"); */
/*         return NULL; */
/*     } */
/*     if (ellMax < 2) { */
/*         PyErr_SetString(PyExc_ValueError, "ellMax must be >= 2"); */
/*         return NULL; */
/*     } */

/*     /\* Ensure contiguous, row-major, float64 *\/ */
/*     PyArrayObject *q_arr = (PyArrayObject *)PyArray_ContiguousFromAny( */
/*         (PyObject *)q_obj, NPY_DOUBLE, 2, 2); */
/*     if (!q_arr) return NULL; */

/*     npy_intp n = PyArray_DIM(q_arr, 1); */
/*     const double *q_data = (const double *)PyArray_DATA(q_arr); */

/*     int num_ells = ellMax - 1; */

/*     /\* ---- Allocate output numpy arrays ---- *\/ */
/*     PyObject *result_list = PyList_New(num_ells); */
/*     if (!result_list) { Py_DECREF(q_arr); return NULL; } */

/*     /\* Array of raw data pointers to pass to C *\/ */
/*     double complex **mat_ptrs = malloc((size_t)num_ells * sizeof *mat_ptrs); */
/*     if (!mat_ptrs) { */
/*         Py_DECREF(q_arr); */
/*         Py_DECREF(result_list); */
/*         return PyErr_NoMemory(); */
/*     } */

/*     for (int i = 0; i < num_ells; ++i) { */
/*         int ell = i + 2; */
/*         npy_intp dims[3] = { 2*ell + 1, 2*ell + 1, n }; */

/*         PyObject *mat = PyArray_ZEROS(3, dims, NPY_COMPLEX128, 0); */
/*         if (!mat) { */
/*             /\* Clean up already-created arrays *\/ */
/*             for (int j = 0; j < i; ++j) */
/*                 Py_DECREF(PyList_GET_ITEM(result_list, j)); */
/*             Py_DECREF(result_list); */
/*             Py_DECREF(q_arr); */
/*             free(mat_ptrs); */
/*             return NULL; */
/*         } */

/*         /\* PyList_SET_ITEM steals the reference, so no extra DECREF needed *\/ */
/*         PyList_SET_ITEM(result_list, i, mat); */
/*         mat_ptrs[i] = (double complex *)PyArray_DATA((PyArrayObject *)mat); */
/*     } */

/*     /\* ---- Call the C core function ---- *\/ */
/*     int ret; */
/*     Py_BEGIN_ALLOW_THREADS */
/*     ret = wignerD_matrices(q_data, (size_t)n, ellMax, mat_ptrs); */
/*     Py_END_ALLOW_THREADS */

/*     free(mat_ptrs); */
/*     Py_DECREF(q_arr); */

/*     if (ret != 0) { */
/*         Py_DECREF(result_list); */
/*         PyErr_SetString(PyExc_RuntimeError, */
/*                         "wignerD_matrices failed (internal error)"); */
/*         return NULL; */
/*     } */

/*     return result_list; */
/* } */

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

    /* Stack-allocated pointer array — no malloc needed.
     * num_ells = ellMax - 1, so for any reasonable ellMax this is tiny. */
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

        /* Direct pointer into NumPy's buffer — no copy */
        mat_ptrs[i] = (double complex *)PyArray_DATA(mat);
    }

    int ret;
    Py_BEGIN_ALLOW_THREADS
    ret = wignerD_matrices(q_data, (size_t)n, ellMax, mat_ptrs);
    Py_END_ALLOW_THREADS

    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
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

/* ================================================================== */
/*  wignerD_matrices_opt                                              */
/*  Optimized: symmetry, real/complex separation, ell recurrence      */
/* ================================================================== */

int wignerD_matrices_opt(const double * restrict q, size_t n, int ellMax,
                         double complex ** restrict matrices)
{
    if (!q || !matrices || ellMax < 2 || n == 0) return -1;

    int NL = ellMax - 1, P = 2 * ellMax, PC = 2 * P + 1;
    int lf_size = 2 * ellMax + 2;

    /* ---- Single allocation ---- */
    int QP = 4 * ellMax + 1;  /* max power for cv/sv tables */
    size_t cap = (size_t)lf_size * sizeof(double)
      + (size_t)n * (
          2 * sizeof(double complex)                    /* ra_f, rb_f */
        + 3 * sizeof(size_t)                            /* i1, i2, i3 */
        + 2 * sizeof(double complex)                    /* ua, ub (unit phases) */
        + 2 * sizeof(double complex)                    /* ua_inv, ub_inv */
        + 2 * (size_t)PC * sizeof(double complex)       /* ua_pw, ub_pw */
        + 4 * sizeof(double)                            /* c2, s2, R, cos_beta */
        + 2 * sizeof(double)                            /* cv, sv */
        + 2 * (size_t)QP * sizeof(double)               /* cv_pw, sv_pw */
      )
      + (size_t)n * ((size_t)PC * sizeof(double complex) + sizeof(double complex))
      + 2 * (size_t)(2*ellMax+1) * (size_t)(2*ellMax+1) * n * sizeof(double)
      + 64 * 30;

    char *mem = malloc(cap);
    if (!mem) return -1;
    Bump B = { mem, mem + cap };

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

    /* ---- Edge cases (identical to original) ---- */
    char *saved_ptr = B.ptr;

    double complex *cpw  = BUMP(B, double complex, (size_t)PC * n);
    double complex *cinv = BUMP(B, double complex, n);
    if (!cpw || !cinv) { free(mem); return -1; }

    #define DO_EDGE_OPT(ni, ia, base_f, row, col, sgn)                  \
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

    DO_EDGE_OPT(n2, i2, rb_f, ell+m, ell-m, ((ell+m)&1 ? 1.0 : -1.0))
    DO_EDGE_OPT(n3, i3, ra_f, ell+m, ell+m, 1.0)
    #undef DO_EDGE_OPT

    /* ---- General case (i1) ---- */
    if (n1 > 0) {
        B.ptr = saved_ptr;

        /* Per-timepoint arrays */
        double complex *ua     = BUMP(B, double complex, n1);
        double complex *ub     = BUMP(B, double complex, n1);
        double complex *ua_inv = BUMP(B, double complex, n1);
        double complex *ub_inv = BUMP(B, double complex, n1);
        double *c2     = BUMP(B, double, n1);  /* |ra|^2 */
        double *s2     = BUMP(B, double, n1);  /* |rb|^2 */
        double *cv     = BUMP(B, double, n1);  /* |ra| */
        double *sv     = BUMP(B, double, n1);  /* |rb| */
        double *R      = BUMP(B, double, n1);  /* s2/c2 */
        double *cos_b  = BUMP(B, double, n1);  /* c2 - s2 = cos(beta) */
        double *cv_pw  = BUMP(B, double, (size_t)QP * n1);  /* |ra|^p */
        double *sv_pw  = BUMP(B, double, (size_t)QP * n1);  /* |rb|^p */
        double complex *ua_pw = BUMP(B, double complex, (size_t)PC * n1);
        double complex *ub_pw = BUMP(B, double complex, (size_t)PC * n1);

        int dim_rec = 2 * ellMax + 1;
        size_t rec_sz = (size_t)dim_rec * (size_t)dim_rec * n1;
        double *d_prev  = BUMP(B, double, rec_sz);
        double *d_prev2 = BUMP(B, double, rec_sz);

        if (!ua||!ub||!ua_inv||!ub_inv||!c2||!s2||!cv||!sv||!R||!cos_b||
            !cv_pw||!sv_pw||!ua_pw||!ub_pw||!d_prev||!d_prev2) {
            free(mem); return -1;
        }

        /* Precompute per-timepoint quantities */
        for (size_t k = 0; k < n1; ++k) {
            double complex a = ra_f[i1[k]], b = rb_f[i1[k]];
            double re_a = creal(a), im_a = cimag(a);
            double re_b = creal(b), im_b = cimag(b);
            double a2 = re_a*re_a + im_a*im_a;
            double b2 = re_b*re_b + im_b*im_b;
            double abs_a = sqrt(a2), abs_b = sqrt(b2);
            c2[k] = a2;
            s2[k] = b2;
            cv[k] = abs_a;
            sv[k] = abs_b;
            R[k]  = b2 / a2;
            cos_b[k] = a2 - b2;
            double complex u_a = a / abs_a;
            double complex u_b = b / abs_b;
            ua[k] = u_a;
            ub[k] = u_b;
            ua_inv[k] = conj(u_a);
            ub_inv[k] = conj(u_b);
        }

        /* Build power tables */
        build_real_pows(cv, n1, QP - 1, cv_pw);
        build_real_pows(sv, n1, QP - 1, sv_pw);
        build_unit_pows(ua, ua_inv, n1, P, ua_pw);
        build_unit_pows(ub, ub_inv, n1, P, ub_pw);

        /* ---- Hot loop over ell ---- */
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t dim = (size_t)(2*ell+1);
            double complex * restrict mat = matrices[i];

            /* Fundamental domain: mp >= 0, m+mp >= 0.
               For each (m, mp) we compute d_real (purely real)
               and write to up to 4 symmetry positions. */
            for (int mp = 0; mp <= ell; ++mp) {
                int m_start = (mp > 0) ? -mp : 0;  /* m+mp >= 0 → m >= -mp */
                for (int m = m_start; m <= ell; ++m) {
                    /* --- Compute d_real via Horner or recurrence --- */
                    int use_recurrence = (ell >= 4 && abs(m) < ell-1 && abs(mp) < ell-1);

                    /* Positions in the recurrence buffer */
                    size_t rec_idx = (size_t)(m + ellMax) * (size_t)dim_rec * n1
                                   + (size_t)(mp + ellMax) * n1;

                    if (use_recurrence) {
                        /* Three-term recurrence */
                        double Ac, mmp_term, Bc, Cc;
                        recurrence_abc(ell, m, mp, &Ac, &Bc, &Cc);
                        double inv_C = 1.0 / Cc;
                        double mmp = -(2.0 * ell - 1.0) * (double)m * (double)mp;

                        size_t prev_idx = (size_t)(m + ellMax) * (size_t)dim_rec * n1
                                        + (size_t)(mp + ellMax) * n1;
                        size_t prev2_idx = prev_idx;

                        for (size_t k = 0; k < n1; ++k) {
                            double d_prev_val  = d_prev[prev_idx + k];
                            double d_prev2_val = d_prev2[prev2_idx + k];
                            double d_new = ((Ac * cos_b[k] + mmp) * d_prev_val
                                           + Bc * d_prev2_val) * inv_C;
                            d_prev2[rec_idx + k] = d_new;  /* store for next ell (will be rotated) */
                        }
                    }

                    /* Horner evaluation (always needed for non-recurrence,
                       and also for recurrence seed) */
                    if (!use_recurrence) {
                        double coef = wigner_coef2(lf, ell, mp, m);
                        int rhoMin = (mp - m > 0) ? (mp - m) : 0;
                        int rhoMax = ell + mp;
                        if (ell - m < rhoMax) rhoMax = ell - m;
                        int nr = rhoMax - rhoMin + 1;

                        double rc_arr[32]; /* max ell=8 → max nr ~ 17 */
                        for (int r = 0; r < nr; ++r)
                            rc_arr[r] = rho_coeff(lf, ell, mp, m, rhoMin + r);

                        /* Powers of |ra| and |rb| for magnitude factor:
                           d_real = coef * horner * |ra|^c_exp * |rb|^s_exp
                           where c_exp = 2*ell-m+mp-2*rhoMin, s_exp = m-mp+2*rhoMin */
                        int c_exp = 2*ell - m + mp - 2*rhoMin;
                        int s_exp = m - mp + 2*rhoMin;
                        const double *cv_row = cv_pw + (size_t)c_exp * n1;
                        const double *sv_row = sv_pw + (size_t)s_exp * n1;

                        for (size_t k = 0; k < n1; ++k) {
                            double Rv = R[k];
                            double h = rc_arr[nr-1];
                            for (int r = nr-2; r >= 0; --r)
                                h = h * Rv + rc_arr[r];
                            /* No R^rhoMin needed — absorbed into cv/sv powers */
                            double d_val = coef * h * cv_row[k] * sv_row[k];
                            d_prev2[rec_idx + k] = d_val;
                        }
                    }

                    /* Now d_prev2[rec_idx + k] has the d_real values for this (m, mp, ell) */

                    /* --- Write to matrix positions using symmetry --- */
                    /* Phase arrays: ua_pw[P + (m+mp)] and ub_pw[P + (m-mp)] */
                    const double complex *ph_a = ua_pw + (size_t)(P + m + mp) * n1;
                    const double complex *ph_b = ub_pw + (size_t)(P + m - mp) * n1;

                    /* Position 1: (m, mp) → D = d * ua^{m+mp} * ub^{m-mp} */
                    {
                        double complex *dst1 = mat + (size_t)(ell+m)*dim*n + (size_t)(ell+mp)*n;
                        for (size_t k = 0; k < n1; ++k)
                            dst1[i1[k]] = d_prev2[rec_idx + k] * ph_a[k] * ph_b[k];
                    }

                    /* Symmetry sign: (-1)^{m-mp} */
                    double sym_sign = ((m - mp) & 1) ? -1.0 : 1.0;

                    /* Position 2: (mp, m) if mp != m → d * sym_sign,
                       phase = ua^{m+mp} * ub^{mp-m} = ua^{m+mp} * conj(ub^{m-mp}) */
                    if (mp != m) {
                        const double complex *ph_b_conj = ub_pw + (size_t)(P + mp - m) * n1;
                        double complex *dst2 = mat + (size_t)(ell+mp)*dim*n + (size_t)(ell+m)*n;
                        for (size_t k = 0; k < n1; ++k)
                            dst2[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a[k] * ph_b_conj[k];
                    }

                    /* Position 3: (-m, -mp) → d * sym_sign,
                       phase = ua^{-(m+mp)} * ub^{-(m-mp)} = conj(ph1) */
                    if (m > 0 || mp > 0) {  /* skip if m==0 && mp==0 (same as pos 1) */
                        const double complex *ph_a_neg = ua_pw + (size_t)(P - m - mp) * n1;
                        const double complex *ph_b_neg = ub_pw + (size_t)(P - m + mp) * n1;
                        double complex *dst3 = mat + (size_t)(ell-m)*dim*n + (size_t)(ell-mp)*n;
                        for (size_t k = 0; k < n1; ++k)
                            dst3[i1[k]] = sym_sign * d_prev2[rec_idx + k] * ph_a_neg[k] * ph_b_neg[k];
                    }

                    /* Position 4: (-mp, -m) → d * 1.0,
                       phase = ua^{-(m+mp)} * ub^{m-mp} → conj(ua^{m+mp}) * ub^{m-mp}
                       But this is NOT the same as conj(ph2). Let's be explicit. */
                    if (mp != m && (m > 0 || mp > 0)) {
                        const double complex *ph_a_neg = ua_pw + (size_t)(P - m - mp) * n1;
                        /* ub power for (-mp, -m): m_new - mp_new = -mp - (-m) = m - mp */
                        double complex *dst4 = mat + (size_t)(ell-mp)*dim*n + (size_t)(ell-m)*n;
                        for (size_t k = 0; k < n1; ++k)
                            dst4[i1[k]] = d_prev2[rec_idx + k] * ph_a_neg[k] * ph_b[k];
                    }
                }
            }

            /* Rotate recurrence buffers for next ell */
            /* Swap d_prev and d_prev2: current ell's values (in d_prev2) become d_prev,
               and d_prev becomes d_prev2 for the next iteration */
            double *tmp = d_prev;
            d_prev = d_prev2;
            d_prev2 = tmp;
        }
    }

    free(mem);
    return 0;
}


/* ================================================================== */
/*  wignerD_matrices_opt_hc4                                          */
/*  Same as _opt but with hardcoded formulas for ell=2,3,4            */
/* ================================================================== */

/* Helper: compute d_real values for all fundamental domain pairs
   at a given ell using hardcoded polynomial formulas.
   c = sqrt(c2), s = sqrt(s2), cN = c^N, sN = s^N.
   Stores into d_buf[m_idx * dim_d * n1 + mp_idx * n1 + k] where
   m_idx = m + ellMax, mp_idx = mp + ellMax.
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
        SD( 0, 1,  2.4494897427831781*c*s3 - 2.4494897427831781*c3*s);
        SD( 0, 2,  2.4494897427831781*c2*s2);
        SD( 1, 0,  -2.4494897427831781*c*s3 + 2.4494897427831781*c3*s);
        SD( 1, 1,  -3.0*c2*s2 + c4);
        SD( 1, 2,  -2.0*c3*s);
        SD( 2, 0,  2.4494897427831781*c2*s2);
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
        SD(-2, 3,  -2.4494897427831781*c*s5);
        SD(-1, 1,  s6 - 8.0*c2*s4 + 6.0*c4*s2);
        SD(-1, 2,  3.1622776601683795*c*s5 - 6.324555320336759*c3*s3);
        SD(-1, 3,  3.872983346207417*c2*s4);
        SD( 0, 0,  -s6 + 9.0*c2*s4 - 9.0*c4*s2 + c6);
        SD( 0, 1,  -3.4641016151377544*c*s5 + 10.392304845413264*c3*s3 - 3.4641016151377544*c5*s);
        SD( 0, 2,  -5.477225575051661*c2*s4 + 5.477225575051661*c4*s2);
        SD( 0, 3,  -4.47213595499958*c3*s3);
        SD( 1, 0,  3.4641016151377544*c*s5 - 10.392304845413264*c3*s3 + 3.4641016151377544*c5*s);
        SD( 1, 1,  6.0*c2*s4 - 8.0*c4*s2 + c6);
        SD( 1, 2,  6.324555320336759*c3*s3 - 3.1622776601683795*c5*s);
        SD( 1, 3,  3.872983346207417*c4*s2);
        SD( 2, 0,  -5.477225575051661*c2*s4 + 5.477225575051661*c4*s2);
        SD( 2, 1,  -6.324555320336759*c3*s3 + 3.1622776601683795*c5*s);
        SD( 2, 2,  -5.0*c4*s2 + c6);
        SD( 2, 3,  -2.4494897427831781*c5*s);
        SD( 3, 0,  4.47213595499958*c3*s3);
        SD( 3, 1,  3.872983346207417*c4*s2);
        SD( 3, 2,  2.4494897427831781*c5*s);
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
        SD(-3, 4,  -2.8284271247461903*c*s7);
        SD(-2, 2,  s8 - 12.0*c2*s6 + 15.0*c4*s4);
        SD(-2, 3,  3.7416573867739413*c*s7 - 11.224972160321824*c3*s5);
        SD(-2, 4,  5.291502622129181*c2*s6);
        SD(-1, 1,  -s8 + 15.0*c2*s6 - 30.0*c4*s4 + 10.0*c6*s2);
        SD(-1, 2,  -4.242640687119286*c*s7 + 21.213203435596427*c3*s5 - 14.142135623730951*c5*s3);
        SD(-1, 3,  -7.937253933193772*c2*s6 + 13.228756555322954*c4*s4);
        SD(-1, 4,  -7.483314773547883*c3*s5);
        SD( 0, 0,  s8 - 16.0*c2*s6 + 36.0*c4*s4 - 16.0*c6*s2 + c8);
        SD( 0, 1,  4.47213595499958*c*s7 - 26.832815729997478*c3*s5 + 26.832815729997478*c5*s3 - 4.47213595499958*c7*s);
        SD( 0, 2,  9.486832980505138*c2*s6 - 25.298221281347036*c4*s4 + 9.486832980505138*c6*s2);
        SD( 0, 3,  11.832159566199232*c3*s5 - 11.832159566199232*c5*s3);
        SD( 0, 4,  8.366600265340756*c4*s4);
        SD( 1, 0,  -4.47213595499958*c*s7 + 26.832815729997478*c3*s5 - 26.832815729997478*c5*s3 + 4.47213595499958*c7*s);
        SD( 1, 1,  -10.0*c2*s6 + 30.0*c4*s4 - 15.0*c6*s2 + c8);
        SD( 1, 2,  -14.142135623730951*c3*s5 + 21.213203435596427*c5*s3 - 4.242640687119286*c7*s);
        SD( 1, 3,  -13.228756555322954*c4*s4 + 7.937253933193772*c6*s2);
        SD( 1, 4,  -7.483314773547883*c5*s3);
        SD( 2, 0,  9.486832980505138*c2*s6 - 25.298221281347036*c4*s4 + 9.486832980505138*c6*s2);
        SD( 2, 1,  14.142135623730951*c3*s5 - 21.213203435596427*c5*s3 + 4.242640687119286*c7*s);
        SD( 2, 2,  15.0*c4*s4 - 12.0*c6*s2 + c8);
        SD( 2, 3,  11.224972160321824*c5*s3 - 3.7416573867739413*c7*s);
        SD( 2, 4,  5.291502622129181*c6*s2);
        SD( 3, 0,  -11.832159566199232*c3*s5 + 11.832159566199232*c5*s3);
        SD( 3, 1,  -13.228756555322954*c4*s4 + 7.937253933193772*c6*s2);
        SD( 3, 2,  -11.224972160321824*c5*s3 + 3.7416573867739413*c7*s);
        SD( 3, 3,  -7.0*c6*s2 + c8);
        SD( 3, 4,  -2.8284271247461903*c7*s);
        SD( 4, 0,  8.366600265340756*c4*s4);
        SD( 4, 1,  7.483314773547883*c5*s3);
        SD( 4, 2,  5.291502622129181*c6*s2);
        SD( 4, 3,  2.8284271247461903*c7*s);
        SD( 4, 4,  c8);

        #undef SD
    }
}

int wignerD_matrices_opt_hc4(const double * restrict q, size_t n, int ellMax,
                             double complex ** restrict matrices)
{
    if (!q || !matrices || ellMax < 2 || n == 0) return -1;

    int NL = ellMax - 1, P = 2 * ellMax, PC = 2 * P + 1;
    int lf_size = 2 * ellMax + 2;

    /* ---- Single allocation ---- */
    int QP = 4 * ellMax + 1;
    size_t cap = (size_t)lf_size * sizeof(double)
      + (size_t)n * (
          2 * sizeof(double complex)
        + 3 * sizeof(size_t)
        + 2 * sizeof(double complex)
        + 2 * sizeof(double complex)
        + 2 * (size_t)PC * sizeof(double complex)
        + 4 * sizeof(double)   /* c2, s2, R, cos_beta */
        + 2 * sizeof(double)   /* cv, sv */
        + 2 * (size_t)QP * sizeof(double)  /* cv_pw, sv_pw */
      )
      + (size_t)n * ((size_t)PC * sizeof(double complex) + sizeof(double complex))
      + 2 * (size_t)(2*ellMax+1) * (size_t)(2*ellMax+1) * n * sizeof(double)
      + 64 * 32;

    char *mem = malloc(cap);
    if (!mem) return -1;
    Bump B = { mem, mem + cap };

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

            /* Rotate recurrence buffers */
            double *tmp = d_prev;
            d_prev = d_prev2;
            d_prev2 = tmp;
        }
    }

    free(mem);
    return 0;
}


/* ================================================================== */
/*  Python wrappers for optimized implementations                     */
/* ================================================================== */

static PyObject *py_wignerD_matrices_opt(PyObject *self, PyObject *args)
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
    ret = wignerD_matrices_opt(q_data, (size_t)n, ellMax, mat_ptrs);
    Py_END_ALLOW_THREADS

    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices_opt failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *py_wignerD_matrices_opt_hc4(PyObject *self, PyObject *args)
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
    ret = wignerD_matrices_opt_hc4(q_data, (size_t)n, ellMax, mat_ptrs);
    Py_END_ALLOW_THREADS

    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices_opt_hc4 failed (internal error)");
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
/* rotate_waveform: compute wignerD + matrix-vector multiply in C     */
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

    PyArrayObject *q_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)q_obj, NPY_DOUBLE, 2, 2);
    if (!q_arr) return NULL;

    const double *q_data = (const double *)PyArray_DATA(q_arr);
    const double complex *h_data = (const double complex *)PyArray_DATA(h_obj);
    double complex *out_data = (double complex *)PyArray_DATA(out_obj);

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

    /* Step 1: compute Wigner D matrices */
    ret = wignerD_matrices_opt_hc4(q_data, (size_t)N, ellMax, mat_ptrs);

    if (ret == 0) {
        /* Step 2: matrix-vector multiply per ell block, per time step.
         * D layout: (dim, dim, N) C-contiguous via D[row * dim * N + col * N + t]
         * h layout: (n_modes, N) C-contiguous, so h[mode][t] = h_data[mode * N + t]
         */
        size_t mode_offset = 0;
        for (int i = 0; i < num_ells; i++) {
            int ell = i + 2;
            int dim = 2 * ell + 1;
            const double complex *D = mat_ptrs[i];

            for (int row = 0; row < dim; row++) {
                double complex *out_row = out_data + (mode_offset + row) * N;
                const double complex *h_base = h_data + mode_offset * N;

                for (size_t t = 0; t < (size_t)N; t++) {
                    double complex acc = 0;
                    for (int col = 0; col < dim; col++) {
                        acc += D[(size_t)row * dim * N + (size_t)col * N + t]
                             * h_base[(size_t)col * N + t];
                    }
                    out_row[t] = acc;
                }
            }
            mode_offset += dim;
        }
    }

    Py_END_ALLOW_THREADS

    free(mat_mem);
    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices_opt_hc4 failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
}
