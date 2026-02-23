import ctypes
from ctypes import c_double, c_long, POINTER, c_int
import numpy as np
import os
from glob import glob

def _load_spline_interp(dll_path):
    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.spline_interp
    func.argtypes = [c_long, c_long,
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double)]
    return func

def _load_spline_interp_multi(dll_path,function_name):

    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.spline_interp_multi
    # func.restype  = c_int
    func.argtypes = [
        c_long, c_long, c_long,
        POINTER(c_double),          # data_x
        POINTER(POINTER(c_double)), # data_y
        POINTER(c_double),          # out_x
        POINTER(POINTER(c_double)), # out_y   (array of pointers)
    ]
    return func

dll_dir = os.path.dirname(os.path.realpath(__file__))
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
  c_interp = _load_spline_interp(spline_libs[0])
  c_interp_multi = _load_spline_interp_multi(spline_libs[0], 'spline_interp_multi')

  # Load complex variant — same signature (pointer-of-pointer to doubles),
  # but data_y/out_y point to interleaved (re,im) pairs.
  _dll = ctypes.CDLL(spline_libs[0], mode=ctypes.RTLD_GLOBAL)
  c_interp_multi_complex = _dll.spline_interp_multi_complex
  c_interp_multi_complex.argtypes = [
      c_long, c_long, c_long,
      POINTER(c_double),          # data_x
      POINTER(POINTER(c_double)), # data_y (interleaved re,im)
      POINTER(c_double),          # out_x
      POINTER(POINTER(c_double)), # out_y  (interleaved re,im)
  ]

def interpolate(xnew, x, y):
    if np.min(xnew) < np.min(x) or np.max(xnew) > np.max(x):
        raise Exception('Extrapolation not allowed')

    x = x.astype('float64', copy=False)
    y = y.astype('float64')
    xnew = xnew.astype('float64', copy=False)

    x_p = x.ctypes.data_as(POINTER(c_double))
    y_p = y.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew  = np.empty(xnew.shape[0])
    ynew_p = ynew.ctypes.data_as(POINTER(c_double))

    c_interp(x.shape[0],xnew.shape[0],x_p,y_p,xnew_p,ynew_p)

    return ynew

def interpolate_many(xnew, x, y):
    x = x.astype('float64', copy=False)
    y = np.ascontiguousarray(y, dtype='float64')
    xnew = xnew.astype('float64', copy=False)

    num_datasets, n_x = y.shape

    x_p = x.ctypes.data_as(POINTER(c_double))
    y_p = y.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew = np.empty((y.shape[0], xnew.shape[0]), dtype='float64')

    n_datasets = y.shape[0]
    row_bytes  = y.shape[1]    * y.itemsize
    out_bytes  = xnew.shape[0] * ynew.itemsize

    PtrArr  = POINTER(c_double) * n_datasets
    y_ptrs    = PtrArr(*(ctypes.cast(y.ctypes.data    + d * row_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))
    ynew_ptrs = PtrArr(*(ctypes.cast(ynew.ctypes.data + d * out_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))

    c_interp_multi(x.shape[0], xnew.shape[0], n_datasets, x_p, y_ptrs, xnew_p,
                   ynew_ptrs)

    return ynew

def interpolate_many_complex(xnew, x, y):
    """Interpolate multiple complex128 datasets sharing the same x-grid.

    Parameters
    ----------
    xnew : (M,) float64 array — evaluation points
    x    : (N,) float64 array — knot x-coordinates
    y    : (num_datasets, N) complex128 array — data values

    Returns
    -------
    ynew : (num_datasets, M) complex128 array
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    xnew = np.ascontiguousarray(xnew, dtype=np.float64)

    n_datasets, n_x = y.shape
    n_out = xnew.shape[0]

    x_p    = x.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew = np.empty((n_datasets, n_out), dtype=np.complex128)

    # Each complex128 row is 2*n doubles interleaved (re,im,re,im,...)
    row_bytes = n_x  * 16  # sizeof(complex128) = 16
    out_bytes = n_out * 16

    PtrArr = POINTER(c_double) * n_datasets
    y_ptrs    = PtrArr(*(ctypes.cast(y.ctypes.data    + d * row_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))
    ynew_ptrs = PtrArr(*(ctypes.cast(ynew.ctypes.data + d * out_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))

    c_interp_multi_complex(n_x, n_out, n_datasets, x_p, y_ptrs, xnew_p,
                           ynew_ptrs)

    return ynew
