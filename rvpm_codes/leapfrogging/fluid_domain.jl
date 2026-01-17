using FLOWVPM
using GeometricTools

nums = 0:10:200        # time steps you want
read_path = "/home/dysco/FLOWUnsteady/leapfrog_simulation00/" # folder containing .h5 files
file_pref = "vortexring"  # prefix used during simulation


P_min = [-2.0, -2.0, -2.0]
P_max = [ 2.0,  2.0,  2.0]
NDIVS = [80, 80, 80]   # resolution



computefluiddomain(
    300_000,             # max number of particles
    P_min, P_max, NDIVS,
    nums,
    read_path,
    file_pref;
    save_path = "fluiddomain",
    file_pref = "velocity",
    add_Uinf = false,
    add_J = false,
    add_Wapprox = false
)