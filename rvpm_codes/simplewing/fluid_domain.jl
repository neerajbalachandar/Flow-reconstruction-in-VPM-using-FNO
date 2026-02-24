import FLOWUnsteady as uns
import FLOWUnsteady: vlm, vpm, gt, Im, dot, norm

run_name        = "simplewing"                  # Name of this simulation
save_path       = "/home/dysco/Neeraj/Flow-reconstruction-in-VPM-using-FNO/data/raw data/poisson_dataset2_simplewing/input/"

# Grid parameters for fluid domain computation

L_domain        = 2.5                  # (m) reference length for domain
dx_domain       = L_domain/32               # (m) cell size in each direction
dy_domain       = L_domain/32
dz_domain       = L_domain/32 

# Domain bounds
Pmin_domain     = L_domain*[-0.5, -1.1, -0.8]   # (m) minimum bound of outline cuboid
Pmax_domain     = L_domain*[ 4, 1.1, 0.8]   # (m) maximum bound of outline cuboid
 

NDIVS_domain    = ceil.(Int, (Pmax_domain .- Pmin_domain)./[dx_domain, dy_domain, dz_domain])
nnodes_domain   = prod(NDIVS_domain .+ 1)        # Total number of nodes

# VPM settings for fluid domain
maxparticles_domain = Int(2.5e6 + nnodes_domain) # 1.5e06
fmm_domain      = vpm.FMM(; p=4, ncrit=50, theta=0.4)
f_sigma_domain  = 0.5                       # Smoothing of node particles
maxsigma_domain = L_domain/10               # Maximum particle size was 20
maxmagGamma_domain = Inf                    # Maximum vortex strength

# File naming for fluid domain
pfield_prefix   =     "simple-wing_pfield"     
staticpfield_prefix = "simple-wing_staticpfield" 
fdom_prefix     =     "simple-wing"           


last_cycle_start = 50
fluid_domain_nums = collect(last_cycle_start:10:200) # temporal resolution
println("   Processing $(length(fluid_domain_nums)) time steps for fluid domain")
println("   Domain grid: $(NDIVS_domain) cells, $(nnodes_domain) nodes")

# Fluid domain save path
fdom_save_path = joinpath(save_path, "fluid_domain")

# Create save path
gt.create_path(fdom_save_path, false)  # No prompt for automated execution

# Grid orientation (aligned with vehicle)
Oaxis_domain = gt.rotation_matrix2(0, 0, 0)

# Include static particles for complete analysis
include_staticparticles = true
other_file_prefs = include_staticparticles ? [staticpfield_prefix] : []
other_read_paths = [save_path for i in 1:length(other_file_prefs)]

# Generate preprocessing function for particle field
preprocessing_pfield = uns.generate_preprocessing_fluiddomain_pfield(
    maxsigma_domain, maxmagGamma_domain;
    verbose=true, v_lvl=1
)

# ----------------- COMPUTE FLUID DOMAIN -----------------------------------
println("\n Computing fluid domain...")

# Process fluid domain computation
uns.computefluiddomain(
    Pmin_domain, Pmax_domain, NDIVS_domain,
    maxparticles_domain,
    fluid_domain_nums, save_path, pfield_prefix;
    Oaxis=Oaxis_domain,
    fmm=fmm_domain,
    f_sigma=f_sigma_domain,
    save_path=fdom_save_path,
    file_pref=fdom_prefix, 
    grid_names=["_fdom"],
    other_file_prefs=other_file_prefs,
    other_read_paths=other_read_paths,
    userfunction_pfield=preprocessing_pfield,
    verbose=true, 
    v_lvl=0
)

println("   Fluid domain computation completed")


println("\n Simulation summary:")
println("    • Main results: $save_path")
if compute_fluid_domain
    println("    • Fluid domain: $(joinpath(save_path, "fluid_domain"))")
end
println("    • VTK files generated for:")
println("      - Vehicle geometry and wake")
println("      - Particle field (pfield)")
println("      - Static particle field (staticpfield)")
if compute_fluid_domain
    println("      - Fluid domain (fdom)")
end

# ----------------- PARAVIEW VISUALIZATION ---------------------------------
if paraview
    println("\n Opening results in ParaView...")
    
    # Find the main VTK file
    vtk_files = filter(x -> endswith(x, ".vtk"), readdir(save_path))
    if !isempty(vtk_files)
        main_vtk = joinpath(save_path, vtk_files[1])
        println("    Opening: $main_vtk")
        try
            run(`paraview --data=$main_vtk`, wait=false)
        catch e
            println("    Could not launch ParaView automatically: $e")
            println("    Please open $main_vtk manually in ParaView")
        end
    end
    
    # Also mention fluid domain files
    if compute_fluid_domain
        fdom_path = joinpath(save_path, "fluid_domain")
        fdom_files = filter(x -> endswith(x, ".vtk"), readdir(fdom_path))
        if !isempty(fdom_files)
            println("    Fluid domain files available in: $fdom_path")
        end
    end
end

println("\n" * "="^80)
println("Simple Wing Simulation Complete")
println("="^80)