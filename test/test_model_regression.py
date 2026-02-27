"""
Test every gwsurrogate model.

Each model should test the following at a small handful of parameter values:

* all modes
* summation of all modes
* each of the two above, for dimensionless and physical waveforms

Running this script as a regression test (with pytest) will download
regression data from Dropbox.

Regression data should NOT be generated locally unless a new regression
file is being created to upload to Dropbox. In this case do

>>> python test_model_regression.py

from the folder test.

NOTES
=====
(1) Single precision regression data

Waveform regression data is saved with single precision in order to,

     (i) reduce the size of the regression file and
     (ii) allow h5diff to not fail due to round-off error (Still fails! switched comparison to np.allclose)

this is OK because the models themselves are not accurate to better than single precision.


(2) Dynamics surrogate not (directly) tested

No regression on dynamics surrogate output. This is probably OK since coorb
full surrogate uses dynamics output.

(3) As of 6/2023, attempt to log versions of some key packages that are known
    to impact regression data


    (i) NRHybSur3dq8_CCE
            sklearn 1.2.2
            numpy   1.24.3
            GSL     2.4
            python  3.11
"""


from __future__ import division
import numpy as np
import gwsurrogate as gws
from gwsurrogate.new import surrogate
import h5py
import os
import time
import warnings
import requests
import pytest

import hashlib

def md5(fname):
  """ Compute has from file. code taken from
  https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

# set global tolerances for floating point comparisons.
# From documentation on np.testing.assert_allclose
# the comparison is
#
#     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
#
# Setting atol=0 means we are only testing relative errors
#
atol = 0.0
# why a high tolerance? For some reason, a high tolerance is needed when
# comparing to regression data on different machines
# TODO: explore the origin of these large discrepancies
#       note that regression data model_regression_data.h5 is saved in single precision (see above)
#       largest relative errors seem to be post-merger
#       only seems to affect models that use gpr fits and/or gsl calls
rtol_default           = 1.e-11
rtols = {'NRHybSur3dq8':  3.4e-5,
         'NRHybSur2dq15': 2.e-5,
         'NRHybSur3dq8_CCE': 7.e-5,  # higher modes for this GPR-fit model require a bit higher tolerance
         'NRHybSur3dq8Tidal': 3.e-4,
         'SpEC_q1_10_NoSpin_linear_alt': 3.e-8, # needed for (8,7) mode to pass. Other modes pass with 1e-11 tolerance
         }

# TODO: new and old surrogate interfaces should be similar enough to avoid
#       model-specific cases like below

# Old surrogate interface
surrogate_old_interface = ["SpEC_q1_10_NoSpin","EOBNRv2_tutorial","EOBNRv2","SpEC_q1_10_NoSpin_linear","EMRISur1dq1e4","BHPTNRSur1dq1e4"]

# news loader class
surrogate_loader_interface = ["NRHybSur3dq8","NRHybSur3dq8Tidal","NRSur7dq4","NRHybSur2dq15","NRHybSur3dq8_CCE","SEOBNRv4PHMSur"]

# Most models are randomly sampled, but in some cases its useful to provide
# test points to activate specific code branches. This is done by mapping
# a model (named in the surrogate catalog) to a function that will return
# 3 unique parameter points.
#
# Each sampler should return a list [q, chiA, chiB] and a dictionary tidOpts
# and a dictionary precesingOpts
model_sampler = {}

def NRHybSur3dq8Tidal_samples(i):
  """ sample points for the NRHybSur3dq8Tidal model
  the ith sample point to evaluate the model at
  samples are returned as [q, chiA, chiB], tidOpts """

  assert i in [0,1,2]

  if i==0:
    return [1.2, [0.,0.,.1], [0.,0.,.1]], {'Lambda1': 1000.0, 'Lambda2': 4000.0}, None
  elif i==1:
    return [1.2, [0.,0.,.4], [0.,0.,-.4]], {'Lambda1': 0.0, 'Lambda2': 9000.0}, None
  elif i==2:
    return [1.2, [0.,0.,.0], [0.,0.,.0]], {'Lambda1': 0.0, 'Lambda2': 0.0}, None

model_sampler["NRHybSur3dq8Tidal"] = NRHybSur3dq8Tidal_samples


def NRSur7dq4_samples(i):
  """ sample points for the NRSur7dq4 model
  the ith sample point to evaluate the model at
  samples are returned as [q, chiA, chiB], precessingOpts """

  assert i in [0,1,2]

  if i==0:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [2., chiA, chiB], None, precessing_opts
  elif i==1:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 1.0}
                       #'use_lalsimulation_conventions': False}
    return [3., chiA, chiB], None, precessing_opts
  elif i==2:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [5., chiA, chiB], None, precessing_opts

model_sampler["NRSur7dq4"] = NRSur7dq4_samples

def SEOBNRv4PHMSur_samples(i):
  """ sample points for the SEOBNRv4PHMSur model
  the ith sample point to evaluate the model at
  samples are returned as [q, chiA, chiB], precessingOpts """

  assert i in [0,1,2]

  if i==0:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [2., chiA, chiB], None, precessing_opts
  elif i==1:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.8]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 1.0}
                       #'use_lalsimulation_conventions': False}
    return [14., chiA, chiB], None, precessing_opts
  elif i==2:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [20., chiA, chiB], None, precessing_opts

model_sampler["SEOBNRv4PHMSur"] = SEOBNRv4PHMSur_samples

def BHPTNRSur1dq1e4_samples(i):
  """ sample points for the BHPTNRSur1dq1e4 model """

  assert i in [0,1,2]

  if i==0:
    return [3.0, [0,0,0],[0,0,0]], None, None
  elif i==1:
    return [100.0, [0,0,0],[0,0,0]], None, None
  elif i==2:
    return [10000.0, [0,0,0],[0,0,0]], None, None

model_sampler["BHPTNRSur1dq1e4"] = BHPTNRSur1dq1e4_samples


def NRHybSur2dq15_samples(i):
  """ sample points for the NRHybSur2dq15 model """

  assert i in [0,1,2]

  if i==0:
    return [2.0, [0,0,-.4],[0,0,0]], None, None
  elif i==1:
    return [11.0, [0,0,.7],[0,0,0]], None, None
  elif i==2:
    return [18.0, [0,0,.4],[0,0,0]], None, None

model_sampler["NRHybSur2dq15"] = NRHybSur2dq15_samples


def NRHybSur3dq8_CCE_samples(i):
  """ sample points for the NRHybSur3dq8_CCE model """

  assert i in [0,1,2]

  if i==0:
    return [2.0, [0,0,-.4],[0,0,0.5]], None, None
  elif i==1:
    return [7.0, [0,0,.7],[0,0,-0.8]], None, None
  elif i==2:
    return [10.0, [0,0,0],[0,0,0.2]], None, None

model_sampler["NRHybSur3dq8_CCE"] = NRHybSur3dq8_CCE_samples


def flatten_params(x):
  """ Convert [q, chiA, chiB] to [q, chiAx, chiAy, chiAz, chiBx, chiBy, chiBz].

  This function is only used when writting samples from specific model samplers
  to HDF5 file. """

  return [x[0],x[1][0],x[1][1],x[1][2],x[2][0],x[2][1],x[2][2]]


# Models to skip (don't have data or too large)
dont_test = [# "SEOBNRv4PHMSur",
             # "NRHybSur2dq15",
             # "BHPTNRSur1dq1e4",
             # "EMRISur1dq1e4",
             # "NRHybSur3dq8_CCE",
             "NRSur4d2s_TDROM_grid12",  # 10 GB file
             "NRSur4d2s_FDROM_grid12",  # 10 GB file
             # "SpEC_q1_10_NoSpin_linear_alt",
             # "SpEC_q1_10_NoSpin_linear",
             "EOBNRv2",  # TODO: this is two surrogates in one. Break up?
             # "SpEC_q1_10_NoSpin",
             # "EOBNRv2_tutorial",
             # "NRHybSur3dq8",
             # "NRHybSur3dq8Tidal",
             # "NRSur7dq4"
             ]


def _get_models_to_test():
  """Return dict {model_name: datafile_or_None} for models to test.

  Models with available data get their datafile path. Models that are missing
  data AND not in dont_test get None as their datafile — these will fail at
  test time with a descriptive error message."""

  surrogate_path = gws.catalog.download_path()

  models = [model for model in gws.catalog._surrogate_world]

  models_to_test = {}
  for model in models:
    surrogate_data = surrogate_path+os.path.basename(gws.catalog._surrogate_world[model][0])
    if os.path.isfile(surrogate_data):
      if model == 'EOBNRv2':
        models_to_test[model] = surrogate_path+'EOBNRv2'
      else:
        models_to_test[model] = surrogate_data
    elif model in dont_test:
      pass  # silently skip models listed in dont_test
    else:
      # Include with None datafile so the test is collected and fails
      models_to_test[model] = None

  # also test the tutorial surrogate
  models_to_test["EOBNRv2_tutorial"] = gws.__path__[0] + "/../tutorial/TutorialSurrogate/EOB_q1_2_NoSpin_Mode22/"

  # remove models from testing
  for i in dont_test:
    models_to_test.pop(i, None)

  return models_to_test


models_to_test = _get_models_to_test()


def _download_regression_data():
  """Download regression data if not already present. Returns the file path."""
  regression_path = "test/model_regression_data.h5"
  if not os.path.isfile(regression_path):
    print("Downloading regression data...")
    url = "https://www.dropbox.com/scl/fi/g12562mas9x4ujxdcdu6e/model_regression_data.h5?rlkey=l21gsvokca5svtjy4nod2dysq&dl=1"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(regression_path, "wb") as file:
      for chunk in response.iter_content(chunk_size=8192):
        if chunk:
          file.write(chunk)
  regression_hash = md5(regression_path)
  print("hash of model_regression_data.h5 is ", regression_hash)
  return regression_path


@pytest.fixture(scope="session")
def regression_data():
  """Download (if needed) and open regression HDF5 file."""
  regression_path = _download_regression_data()
  fp = h5py.File(regression_path, 'r')
  yield fp
  fp.close()


@pytest.mark.parametrize("model", list(models_to_test.keys()), ids=list(models_to_test.keys()))
def test_model_regression(model, regression_data, tmp_path):
  """Test a single surrogate model against regression data."""

  datafile = models_to_test[model]
  fp_regression = regression_data

  if datafile is None:
    surrogate_path = gws.catalog.download_path()
    surrogate_data = surrogate_path+os.path.basename(gws.catalog._surrogate_world[model][0])
    msg = "Surrogate missing!!!\n"
    msg += "Surrogate data assumed to be in the path %s.\n"%surrogate_data
    msg += "If the data is somewhere else, move the file or create a symbolic link.\n"
    msg += "To download this surrogate, from python do\n\n >>> gws.catalog.pull(\"%s\")\n"%model
    pytest.fail(msg)

  # reproducibility per model
  np.random.seed(0)

  print("Testing model = %s"%model)
  print(datafile)

  # Load model
  if model in surrogate_old_interface:
    sur = gws.EvaluateSurrogate(datafile)
    p_mins = sur.param_space.min_vals()
    p_maxs = sur.param_space.max_vals()
  elif model in surrogate_loader_interface:
    sur = gws.LoadSurrogate(model)
    try:
      p_mins = sur._sur_dimless.param_space.min_vals()
      p_maxs = sur._sur_dimless.param_space.max_vals()
    except AttributeError:
      p_mins = None
      p_maxs = None
  else:
    sur = surrogate.FastTensorSplineSurrogate()
    sur.load(datafile)
    p_mins = sur.param_space.min_vals()
    p_maxs = sur.param_space.max_vals()

  # NOTE: NRHybSur3dq8_CCE and NRHybSur3dq8 will report different max/min
  #       because NRHybSur3dq8 defines its interval in q while NRHybSur3dq8_CCE
  #       defines its interval in log(q). This change in the hdf5 file's
  #       behavior is due a change in convert_to_gwsurrogate.py (building code).
  #       Neither NRHybSur3dq8_CCE nor NRHybSur3dq8 use
  #       sur._sur_dimless.param_space for model evaluations. See ParamSpace's docstring

  print("parameter minimum values for model %s"%model, p_mins)
  print("parameter maximum values for model %s"%model, p_maxs)

  # Get parameter samples
  param_samples = []
  for i in range(3):
    if model in list(model_sampler.keys()):
      custom_sampler = model_sampler[model]
      x, tidOpts, pecOpts = custom_sampler(i)
    else:
      print(model+"/parameter%i/parameter"%i)
      x = list(fp_regression[model+"/parameter%i/parameter"%i][:])
      tidOpts = None
      pecOpts = None
    param_samples.append([x, tidOpts, pecOpts])

  # Write comparison data to a temporary HDF5 file
  h5_file = str(tmp_path / "comparison_data.h5")
  fp = h5py.File(h5_file, "w")
  model_grp = fp.create_group(model)

  for i, ps in enumerate(param_samples):
    x = ps[0]
    tidOpts = ps[1]
    pecOpts = ps[2]
    if model in surrogate_old_interface:
      ps_float = x[0]
      modes, t, hp, hc = sur(q=ps_float, mode_sum=False, fake_neg_modes=True)
    else:
      if model in surrogate_loader_interface:
        q = x[0]
        if type(x[1]) is np.float64 or type(x[1]) is float:
          chiA = np.array([0, 0, x[1]])
          chiB = np.array([0, 0, x[2]])
        elif len(x[1])==3:
          chiA = np.array(x[1])
          chiB = np.array(x[2])
        else:
          raise ValueError
        try:
            if model in ["NRSur7dq4"]:
              with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
            else:
              t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
        except ValueError:
          t, h, dyanmics = sur(q, chiA, chiB, dt = 0.25, f_low=3.e-3, tidal_opts=tidOpts, precessing_opts=pecOpts)
      else:
        h = sur(x)
        try:
          t = fp_regression[model+"/parameter%i/time"%i][:]
        except KeyError:
          t = sur.domain  # FastTensorSplineSurrogate exposes time grid as .domain
      try:
        modes = sur.mode_list
        h_np = [h[mode] for mode in modes]
      except AttributeError:
        modes = sur._sur_dimless.mode_list
        h_np = [h[mode] for mode in modes]

      h_np = np.vstack(h_np)
      hp = np.real(h_np)
      hc = np.imag(h_np)
    samplei = model_grp.create_group("parameter"+str(i))
    if model in list(model_sampler.keys()):
      x = flatten_params(x)
    samplei.create_dataset("parameter", data=x)
    samplei.create_dataset("hp", data=hp, dtype='float32')
    samplei.create_dataset("hc", data=hc, dtype='float32')
    samplei.create_dataset("time", data=t, dtype='float32')
    samplei.create_dataset("modes", data=np.array(modes), dtype='int')

  fp.close()

  # Compare against regression data
  fp = h5py.File(h5_file, "r")
  print("testing model %s ..."%model)
  for i in range(3):
    hp_regression = fp_regression[model+"/parameter%i/hp"%i][:]
    hc_regression = fp_regression[model+"/parameter%i/hc"%i][:]
    hp_comparison = fp[model+"/parameter%i/hp"%i][:]
    hc_comparison = fp[model+"/parameter%i/hc"%i][:]

    t_indx = fp[model+"/parameter%i/time"%i][:].shape[0]

    local_rtol = rtols.get(model, rtol_default)
    print("Model %s uses a relative error tolerance of %e"%(model, local_rtol))

    for j, mode in enumerate(fp[model+"/parameter%i/modes"%i][:]):
      err_msg="Failed: model %s for mode index %i (ell = %i,m = %i)"%(model,j,mode[0],mode[1])

      h_regression = hp_regression[j,:t_indx] + 1.0j*hc_regression[j,:t_indx]
      h_comparison = hp_comparison[j,:t_indx] + 1.0j*hc_comparison[j,:t_indx]
      np.testing.assert_allclose(h_regression, h_comparison, rtol=local_rtol, atol=atol, err_msg=err_msg)

  fp.close()
  print("model %s passed"%model)


def generate_regression_data():
  """Generate regression data file. Run from the test/ directory."""

  h5_file = "model_regression_data_new.h5"
  print("Generating regression data file... Make sure this step is done BEFORE making any code changes!\n")
  print(os.path.exists(h5_file))
  if os.path.exists(h5_file):
    raise RuntimeError("Refusing to overwrite a regression file!")

  surrogate_path = gws.catalog.download_path()
  np.random.seed(0)

  models = [model for model in gws.catalog._surrogate_world]
  print(models)

  gen_models_to_test = {}
  for model in models:
    surrogate_data = surrogate_path+os.path.basename(gws.catalog._surrogate_world[model][0])
    if os.path.isfile(surrogate_data):
      if model == 'EOBNRv2':
        gen_models_to_test[model] = surrogate_path+'EOBNRv2'
      else:
        gen_models_to_test[model] = surrogate_data
    elif model in dont_test:
      msg = "WARNING: skipping model %s which is listed as dont test\n"%model
    else:
      msg = "WARNING: Surrogate missing!!!\n"
      msg += "Surrogate data assumed to be in the path %s.\n"%surrogate_data
      msg += "If the data is somewhere else, change the path or move the file.\n\n"
      msg +="To download this surrogate, from ipython do\n\n >>> gws.catalog.pull(%s)\n"%model
      print(msg)
      time.sleep(1)
      assert(False)

  gen_models_to_test["EOBNRv2_tutorial"] = gws.__path__[0] + "/../tutorial/TutorialSurrogate/EOB_q1_2_NoSpin_Mode22/"

  for i in dont_test:
    try:
      gen_models_to_test.pop(i)
      print("model %s removed from testing"%i)
    except KeyError:
      print("model %s cannot be removed"%i)

  fp = h5py.File(h5_file, "w")

  param_samples_tested = []
  for model, datafile in gen_models_to_test.items():

    print("Generating data for model = %s"%model)
    print(datafile)

    if model in surrogate_old_interface:
      sur = gws.EvaluateSurrogate(datafile)
      p_mins = sur.param_space.min_vals()
      p_maxs = sur.param_space.max_vals()
    elif model in surrogate_loader_interface:
      sur = gws.LoadSurrogate(model)
      try:
        p_mins = sur._sur_dimless.param_space.min_vals()
        p_maxs = sur._sur_dimless.param_space.max_vals()
      except AttributeError:
        p_mins = None
        p_maxs = None
    else:
      sur = surrogate.FastTensorSplineSurrogate()
      sur.load(datafile)
      p_mins = sur.param_space.min_vals()
      p_maxs = sur.param_space.max_vals()

    print("parameter minimum values for model %s"%model, p_mins)
    print("parameter maximum values for model %s"%model, p_maxs)

    param_samples = []
    for i in range(3):
      if model in list(model_sampler.keys()):
        custom_sampler = model_sampler[model]
        x, tidOpts, pecOpts = custom_sampler(i)
      else:
        x = []
        for j in range(len(p_mins)):
          xj_min = p_mins[j]
          xj_max = p_maxs[j]
          tmp = float(np.random.uniform(xj_min, xj_max, size=1))
          x.append(tmp)
        tidOpts = None
        pecOpts = None
      param_samples.append([x, tidOpts, pecOpts])

    param_samples_tested.append(param_samples)

    model_grp = fp.create_group(model)
    for i, ps in enumerate(param_samples):
      x = ps[0]
      tidOpts = ps[1]
      pecOpts = ps[2]
      if model in surrogate_old_interface:
        ps_float = x[0]
        modes, t, hp, hc = sur(q=ps_float, mode_sum=False, fake_neg_modes=True)
      else:
        if model in surrogate_loader_interface:
          q = x[0]
          if type(x[1]) is np.float64 or type(x[1]) is float:
            chiA = np.array([0, 0, x[1]])
            chiB = np.array([0, 0, x[2]])
          elif len(x[1])==3:
            chiA = np.array(x[1])
            chiB = np.array(x[2])
          else:
            raise ValueError
          try:
              if model in ["NRSur7dq4"]:
                with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
              else:
                t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
          except ValueError:
            t, h, dyanmics = sur(q, chiA, chiB, dt = 0.25, f_low=3.e-3, tidal_opts=tidOpts, precessing_opts=pecOpts)
        else:
          h= sur(x)
        try:
          modes = sur.mode_list
          h_np = [h[mode] for mode in modes]
        except AttributeError:
          modes = sur._sur_dimless.mode_list
          h_np = [h[mode] for mode in modes]

        h_np = np.vstack(h_np)
        hp = np.real(h_np)
        hc = np.imag(h_np)
      samplei = model_grp.create_group("parameter"+str(i))
      if model in list(model_sampler.keys()):
        x = flatten_params(x)
      samplei.create_dataset("parameter", data=x)
      samplei.create_dataset("hp", data=hp, dtype='float32')
      samplei.create_dataset("hc", data=hc, dtype='float32')
      samplei.create_dataset("time", data=t, dtype='float32')
      samplei.create_dataset("modes", data=np.array(modes), dtype='int')

  fp.close()
  print("models tested... ")


#------------------------------------------------------------------------------
if __name__ == "__main__":
  generate_regression_data()
