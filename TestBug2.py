import gwsurrogate
import cProfile
import pstats
from pstats import SortKey

sur7dq4 = gwsurrogate.LoadSurrogate('NRSur7dq4')

# Configure coarse-grid rotation: lagrange8 interpolation with stride 4
# (rotates on ~501 of 2000 time nodes, interpolates to user times)
sur7dq4._sur_dimless.set_interp_method('lagrange8', stride=4)

q=2.0
chiA = [0.2, 0.3, -0.4]
chiB = [0.1, -0.3, 0.2]
f_ref = 40
f_low = 20
dist_mpc = 100
# dt = 1./4096
dt = 1./32
ellMax = 4
M=60.0

s = sur7dq4._sur_dimless
print(f"Interpolation method: {s._interp_method_name}, stride={s._rotation_stride}, "
      f"rotation points={len(s._coarse_rotation_idx)}/{len(s.t_coorb)}")

t, h, dyn = sur7dq4(q, chiA, chiB, f_low=f_low, f_ref=f_ref, dist_mpc=dist_mpc, dt=dt, ellMax=ellMax, M=M, units='mks')   # dyn stands for dynamics, do dyn.keys() to see contents

N = 1000

# 1) Produce N independent profile outputs
for i in range(N):
    prof = cProfile.Profile()
    prof.runcall(sur7dq4, q, chiA, chiB, dt=dt, f_low=f_low, f_ref=f_ref, dist_mpc=dist_mpc, ellMax=ellMax, M=M, units='mks')  # avoids eval of a string
    prof.dump_stats(f"restats_{i}.pstats")

# 2) Aggregate them
stats = pstats.Stats("restats_0.pstats")
# stats.print_callers('numpy.array', 50)  # shows WHO is calling np.array
# stats.print_callers('numpy.asarray', 50)  # shows WHO is calling np.asarray
for i in range(1, N):
    stats.add(f"restats_{i}.pstats")

# 3) Show totals (over N runs)
stats.strip_dirs().sort_stats(SortKey.TIME).print_stats()

# 4) Print averaged timing per function (cumtime and tottime divided by N)
for func, (cc, nc, tt, ct, callers) in stats.stats.items():
    stats.stats[func] = (cc, nc, tt / N, ct / N, callers)

print("\nAveraged over", N, "runs (times in milliseconds):")
for func, (cc, nc, tt, ct, callers) in stats.stats.items():
    stats.stats[func] = (cc, nc, tt * 1000, ct * 1000, callers)
stats.strip_dirs().sort_stats(SortKey.TIME).print_stats()

# 5) Clean up .pstats files
import glob, os
for f in glob.glob("restats_*.pstats"):
    os.remove(f)
