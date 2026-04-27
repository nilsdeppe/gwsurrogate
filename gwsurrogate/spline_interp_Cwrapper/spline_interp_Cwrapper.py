import ctypes
from ctypes import c_double, c_long, POINTER, c_int, c_void_p
import numpy as np
import os
from glob import glob

def _load_c_func(dll_path, function_name, argtypes, restype=None):
    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = argtypes
    if restype is not None:
        func.restype = restype
    return func

dll_dir = os.path.dirname(os.path.realpath(__file__))

# --- Load cubic spline library ---
dll_path_glob = '%s/_spline_interp*so'%dll_dir
spline_libs = glob(dll_path_glob)
if len(spline_libs) == 0:
  all_files = glob('%s/*'%dll_dir)
  msg = '_spline_interp library not found! Searched in path %s which has files...\n'%dll_dir
  for all_file in all_files:
    msg += all_file+"\n"
  raise Exception(msg)
elif len(spline_libs) > 1:
  raise Exception('there should be only one _spline_interp library!')
else:
  _MULTI_COMPLEX_ARGS = [c_long, c_long, c_long,
                         c_void_p, c_void_p, c_void_p, c_void_p]
  c_interp = _load_c_func(spline_libs[0], 'spline_interp', [
      c_long, c_long,
      c_void_p, c_void_p,
      c_void_p, c_void_p,
  ], restype=c_int)
  c_interp_multi = _load_c_func(spline_libs[0], 'spline_interp_multi',
      _MULTI_COMPLEX_ARGS, restype=c_int)
  c_interp_multi_complex = _load_c_func(spline_libs[0],
      'spline_interp_multi_complex', _MULTI_COMPLEX_ARGS, restype=c_int)

# --- Load higher-order interpolation library ---
_ho_funcs = {}
quintic_libs = glob('%s/_quintic_interp*so' % dll_dir)
if len(quintic_libs) == 1:
    _ho_lib = quintic_libs[0]
    _ho_funcs['quintic_hermite'] = _load_c_func(_ho_lib,
        'quintic_interp_many_complex', _MULTI_COMPLEX_ARGS, restype=c_int)
    for n in (4, 6, 8, 10, 12):
        _ho_funcs['lagrange%d' % n] = _load_c_func(_ho_lib,
            'lagrange%d_interp_many_complex' % n,
            _MULTI_COMPLEX_ARGS, restype=c_int)
    # Floater-Hormann barycentric rational (blend orders 3, 5, 7)
    for d in (3, 5, 7):
        _ho_funcs['floater_hormann%d' % d] = _load_c_func(_ho_lib,
            'floater_hormann%d_interp_many_complex' % d,
            _MULTI_COMPLEX_ARGS, restype=c_int)
    # Quintic B-spline
    _ho_funcs['bspline5'] = _load_c_func(_ho_lib,
        'bspline5_interp_many_complex', _MULTI_COMPLEX_ARGS, restype=c_int)

_SPLINE_ERRORS = {
    1: 'Memory allocation failed in spline interpolation',
    2: 'Evaluation point outside data range (extrapolation not supported)',
    3: 'Zero-length interval in input x data (duplicate or non-monotonic knots)',
    4: 'data_size must be >= 3 (need at least 3 knots for cubic spline)',
    5: 'out_size must be > 0 (no evaluation points provided)',
    6: 'num_datasets must be > 0 (no datasets provided)',
}

def _check_spline_rc(rc):
    if rc != 0:
        msg = _SPLINE_ERRORS.get(rc, 'Unknown spline error (code %d)' % rc)
        raise RuntimeError(msg)


# ======================================================================
# Low-level wrappers (unchanged API for backward compatibility)
# ======================================================================

def interpolate(xnew, x, y):
    x = x.astype('float64', copy=False)
    y = y.astype('float64')
    xnew = xnew.astype('float64', copy=False)

    x_p = x.ctypes.data
    y_p = y.ctypes.data
    xnew_p = xnew.ctypes.data

    ynew  = np.empty(xnew.shape[0])
    ynew_p = ynew.ctypes.data

    rc = c_interp(x.shape[0],xnew.shape[0],x_p,y_p,xnew_p,ynew_p)
    _check_spline_rc(rc)

    return ynew

def interpolate_many(xnew, x, y):
    """Interpolate multiple real-valued datasets sharing the same x-grid.

    Uses natural cubic spline interpolation (y'' = 0 at boundaries).

    Parameters
    ----------
    xnew : (M,) array_like, float64
        Evaluation points. Must lie within ``[x[0], x[-1]]`` (no
        extrapolation). Sorted ascending is optimal but not required.
    x : (N,) array_like, float64
        Knot x-coordinates, strictly monotonically increasing (no
        duplicates). Must have N >= 3.
    y : (num_datasets, N) array_like, float64
        Data values. Each row is an independent dataset sampled at `x`.
        Must be C-contiguous; will be copied if not already float64.

    Returns
    -------
    ynew : (num_datasets, M) ndarray, dtype float64
        Interpolated values at `xnew` for each dataset.

    Raises
    ------
    RuntimeError
        If the C interpolation routine returns a non-zero error code:
        out-of-bounds evaluation point, duplicate/non-monotonic knots,
        fewer than 3 knots, or memory allocation failure.
    """
    x = x.astype('float64', copy=False)
    y = np.ascontiguousarray(y, dtype='float64')
    xnew = xnew.astype('float64', copy=False)

    num_datasets, n_x = y.shape

    x_p = x.ctypes.data
    xnew_p = xnew.ctypes.data

    ynew = np.empty((y.shape[0], xnew.shape[0]), dtype='float64')

    n_datasets = y.shape[0]
    row_bytes  = y.shape[1]    * y.itemsize
    out_bytes  = xnew.shape[0] * ynew.itemsize

    VoidPtrArr = c_void_p * n_datasets
    y_base    = y.ctypes.data
    ynew_base = ynew.ctypes.data
    y_ptrs    = VoidPtrArr(*(y_base    + d * row_bytes for d in range(n_datasets)))
    ynew_ptrs = VoidPtrArr(*(ynew_base + d * out_bytes for d in range(n_datasets)))

    rc = c_interp_multi(x.shape[0], xnew.shape[0], n_datasets, x_p, y_ptrs, xnew_p,
                        ynew_ptrs)
    _check_spline_rc(rc)

    return ynew

def interpolate_many_complex(xnew, x, y):
    """Interpolate multiple complex128 datasets using cubic spline.

    Parameters
    ----------
    xnew : (M,) float64 array — evaluation points
    x    : (N,) float64 array — knot x-coordinates
    y    : (num_datasets, N) complex128 array — data values

    Returns
    -------
    ynew : (num_datasets, M) complex128 array
    """
    return _call_many_complex(c_interp_multi_complex, xnew, x, y)


# ======================================================================
# Unified higher-order interpolation API
# ======================================================================

def _call_many_complex(c_func, xnew, x, y):
    """Shared implementation for all batch complex128 interpolation calls."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    xnew = np.ascontiguousarray(xnew, dtype=np.float64)

    n_datasets, n_x = y.shape
    n_out = xnew.shape[0]

    ynew = np.empty((n_datasets, n_out), dtype=np.complex128)

    row_bytes = n_x  * 16
    out_bytes = n_out * 16

    VoidPtrArr = c_void_p * n_datasets
    y_base    = y.ctypes.data
    ynew_base = ynew.ctypes.data
    y_ptrs    = VoidPtrArr(*(y_base    + d * row_bytes for d in range(n_datasets)))
    ynew_ptrs = VoidPtrArr(*(ynew_base + d * out_bytes for d in range(n_datasets)))

    rc = c_func(n_x, n_out, n_datasets,
                x.ctypes.data, y_ptrs,
                xnew.ctypes.data, ynew_ptrs)
    _check_spline_rc(rc)

    return ynew


# Available interpolation methods for batch complex128 data.
# Each entry maps a method name to a callable with signature
# (xnew, x, y) -> ynew.
INTERP_METHODS = {
    'cubic': interpolate_many_complex,
}

# Register higher-order methods if the library was loaded
for _name, _cfunc in _ho_funcs.items():
    def _make_wrapper(cf):
        return lambda xnew, x, y: _call_many_complex(cf, xnew, x, y)
    INTERP_METHODS[_name] = _make_wrapper(_cfunc)


def get_interp_func(method='cubic'):
    """Return the batch complex128 interpolation function for a given method.

    Parameters
    ----------
    method : str
        One of: 'cubic', 'lagrange4', 'lagrange6', 'lagrange8',
        'lagrange10', 'lagrange12', 'quintic_hermite'.

    Returns
    -------
    callable : (xnew, x, y) -> ynew
    """
    if method not in INTERP_METHODS:
        available = ', '.join(sorted(INTERP_METHODS.keys()))
        raise ValueError(
            "Unknown interpolation method '%s'. Available: %s" % (method, available))
    return INTERP_METHODS[method]


# Backward-compatible aliases
def quintic_interpolate_many_complex(xnew, x, y):
    return INTERP_METHODS['quintic_hermite'](xnew, x, y)

def lagrange6_interpolate_many_complex(xnew, x, y):
    return INTERP_METHODS['lagrange6'](xnew, x, y)
