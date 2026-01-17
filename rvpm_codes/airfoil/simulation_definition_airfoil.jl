# Resolution is 64 x 64 x 20 
# Input-Output pairs of properties of vortex particles and the velcoity field associated with the fluid domain in wake

#-----------------------------------------------------------NEERAJ BALACHANDAR------------------------------------------------------------------
__authors__ = "NEERAJ BALACHANDAR"
__contact__ = "neerajbalachandar@gmail.com"

import FLOWUnsteady as uns
import FLOWUnsteady: vlm, vpm, gt, Im, dot, norm

include("maneuver_definition_airfoil.jl")      
include("vehicle_definition_airfoil.jl")     
include("monitor_definition_airfoil.jl")    


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# ----------------- GENERAL SIMULATION SETUP ----------------------------------
run_name        = "airfoil"
save_path       = "/home/dysco/FLOWUnsteady/FLOWUnsteady/Unsteady Aerodynamics/Simulations/codes/data_gen/poisson pde/dataset1"  # Where to save results
paraview        = true                           
compute_fluid_domain = true                       
save_horseshoes = true



# ----------------- GEOMETRY PARAMETERS ----------------------------------------
add_wings       = true        
add_rotors      = false             

wingspan        = 4.0                      
chord = 1.0

# ----------------- FLIGHT PARAMETERS ------------------------------------------
vehicle_velocity = 0.01                     
angle_of_attack = 30.0                   

# Flight conditions
rho             = 1.200                     # (kg/m^3) air density
mu              = 1.82e-5                   # (kg/ms) air dynamic viscosity

# Time parameters                   
ttot            = 2.0
nsteps          = 1000         
dt              = ttot/nsteps

# Simulation time window
tstart          = 0.00 * ttot              
tquit           = 1.00 * ttot               

Vinf(X, t)      = 1.0*[1.0, 0.0, 0.0]


# ----------------- SOLVER PARAMETERS ------------------------------------------

# Aerodynamic solver
VehicleType     = uns.VLMVehicle           # Unsteady solver for flapping wings

# VPM particle shedding
p_per_step      = 2                         # Cause for increased simulation time   here
shed_starting   = true                     # No starting vortex needed
shed_unsteady   = true                      # Shed vorticity from unsteady loading
unsteady_shedcrit = 0.001                    # Circulation fluctuation threshold
wake_coupled = false

# Regularization parameters
sigma_vlm_surf  = wingspan/60              # VLM-on-VPM smoothing radius
sigma_rotor_surf = 0.0001
sigma_vlm_solver = 0.002
lambda_vpm      = 2.0                     # VPM core overlap 
sigma_vpm_overwrite = 0.1
sigmafactor_vpmonvlm = 5.0                    # VPM-on-VLM velocity factor

# Wing solver parameters
vlm_rlx         = 0.5                       
vlm_vortexsheet = true                     # ALM/ASM
vlm_vortexsheet_overlap = 2.125
vlm_vortexsheet_distribution = uns.g_pressure
vlm_vortexsheet_sigma_tbv = sigma_vpm_overwrite
vlm_vortexsheet_maxstaticparticle = 1000000

# Force calculation parameters
KJforce_type    = "weighted"                 
include_trailingboundvortex = true
include_unsteadyforce = true                
add_unsteadyforce = false
include_parasiticdrag = true
add_skinfriction = true
calc_cd_from_cl = false
wing_polar_file = "naca0012.csv" 

# VPM solver settings
vpm_integration = vpm.euler                 # Temporal integration scheme
vpm_viscous     = vpm.Inviscid()           # Viscous diffusion

# Subfilter-scale model HIGH FID
vpm_SFS = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
                         alpha=0.999, maxC=1.0,
                         clippings=[vpm.clipping_backscatter],
                         controls=[vpm.control_directional, vpm.control_magnitude])

# vpm_SFS = vpm.SFS_none # Low and Mid fid... here

# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

# ----------------- 1) VEHICLE DEFINITION --------------------------------------
println("\n1) Generating airfoil geometry...")

vehicle = airfoil_geometry()

println("   Vehicle generated with $(length(vehicle.system.wings)) wings")
println("   Total panels: $(sum(uns.vlm.get_m(wing) for wing in vehicle.system.wings))")

# ----------------- 2) MANEUVER DEFINITION -------------------------------------
println("\n2) Generating flapping maneuver...")

maneuver = airfoil_maneuver(;
    disp_plot = true,
    add_wings = add_wings,
    vehicle_velocity = vehicle_velocity,
    angle_of_attack = angle_of_attack
)


# ----------------- 3) SIMULATION DEFINITION -----------------------------------
println("\n3) Setting up simulation...")

# Reference parameters
Vref = vehicle_velocity
RPMref = 120              

# Initial conditions
t0 = tstart / ttot
Vinit = Vref * maneuver.Vvehicle(t0)
Winit = pi/180 * (maneuver.anglevehicle(t0 + 1e-12) - maneuver.anglevehicle(t0)) / (ttot * 1e-12)

# Create simulation object
simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, ttot;
                           Vinit=Vinit, Winit=Winit, t=tstart)

# Memory allocation for particles
max_particles = ceil(Int, (nsteps + 2) * (2 * vlm.get_m(vehicle.wake_system) * (p_per_step + 1) + p_per_step))
max_particles = min(3000000, max_particles) # here

println("   Simulation configured: $nsteps steps, max $max_particles particles")


function generate_calc_aerodynamicforce(; add_parasiticdrag=true,
                                          add_skinfriction=true,
                                          airfoilpolar="naca0012.csv",
                                          parasiticdrag_args=(),
                                          )
    kuttajoukowski = generate_aerodynamicforce_kuttajoukowski("weighted",
                                                                1.0, 0.0001,
                                                                true, 2.125,
                                                                uns.g_pressure,vehicle)

    function calc_aerodynamicforce(vlm_system, args...; per_unit_span=false, optargs...)

        fieldname = per_unit_span ? "ftot" : "Ftot"
        if fieldname in keys(vlm_system.sol)
            pop!(vlm_system.sol, fieldname)
        end
        # Calculate Kutta-Joukowski force
        Ftot = kuttajoukowski(vlm_system, args...; per_unit_span=per_unit_span, optargs...)
        return Ftot
    end
    
    return calc_aerodynamicforce
end

function generate_aerodynamicforce_kuttajoukowski(KJforce_type::String,
                                sigma_vlm_surf, sigma_rotor_surf,
                                vlm_vortexsheet, vlm_vortexsheet_overlap,
                                vlm_vortexsheet_distribution,
                                vlm_vortexsheet_sigma_tbv;
                                vehicle=vehicle
                                )

    function calc_aerodynamicforce_kuttajoukowski(vlm_system::Union{vlm.Wing, vlm.WingSystem},
                                    prev_vlm_system, pfield, Vinf, dt, rho; t=0.0,
                                    per_unit_span=false, spandir=[0, 1, 0],
                                    include_trailingboundvortex=true,
                                    Vout=nothing, lenout=nothing,
                                    lencrit=-1, debug=false)

        m = vlm.get_m(vlm_system)    # Number of horseshoes

        if vehicle != nothing && !("Gamma" in keys(vlm_system.sol))
            # Get the global Gamma solution from the overall vehicle system
            global_gamma_solution = vehicle.system.sol["Gamma"]

            # Determine the starting index for the current vlm_system within the global Gamma array.
            idx_start = 1

            if vlm_system === vehicle.tilting_systems[1]  # fore_left
                idx_start = 1
            elseif vlm_system === vehicle.tilting_systems[2]  # fore_right
                m_fore_left = vlm.get_m(vehicle.tilting_systems[1])
                idx_start = m_fore_left + 1
            elseif vlm_system === vehicle.tilting_systems[3]  # hind_left
                m_fore_left = vlm.get_m(vehicle.tilting_systems[1])
                m_fore_right = vlm.get_m(vehicle.tilting_systems[2])
                idx_start = m_fore_left + m_fore_right + 1
            elseif vlm_system === vehicle.tilting_systems[4]  # hind_right
                m_fore_left = vlm.get_m(vehicle.tilting_systems[1])
                m_fore_right = vlm.get_m(vehicle.tilting_systems[2])
                m_hind_left = vlm.get_m(vehicle.tilting_systems[3])
                idx_start = m_fore_left + m_fore_right + m_hind_left + 1
            elseif vlm_system === vehicle.system
                # If it's the full system, it should already have Gamma
                @warn "Full vehicle system passed but Gamma is missing. This is unexpected."
                return nothing
            else
                # If for some reason vlm_system is not identified
                error("Cannot determine global Gamma slice for the given vlm_system. " *
                      "It is not identified as any of the four wing systems, and 'Gamma' is missing.")
            end

            # Calculate the end index for the slice.
            m = vlm.get_m(vlm_system)
            idx_end = idx_start + m - 1

            # Basic sanity check to prevent out-of-bounds access
            if idx_end > length(global_gamma_solution)
                error("Calculated Gamma slice indices ($idx_start to $idx_end) exceed the bounds of the global Gamma solution (length: $(length(global_gamma_solution))).")
            end

            # Manually assign the relevant slice of Gamma to the current vlm_system's solution dictionary.
            vlm_system.sol["Gamma"] = global_gamma_solution[idx_start:idx_end]

        elseif vehicle == nothing && !("Gamma" in keys(vlm_system.sol))
            error("Gamma solution not found in vlm_system.sol and the vehicle object is not provided for global Gamma access. Cannot proceed with Kutta-Joukowski calculation.")
        end

        # Nodes of every horseshoe
        Ap = uns._get_Xs(vlm_system, "Ap")
        A = uns._get_Xs(vlm_system, "A")
        B = uns._get_Xs(vlm_system, "B")
        Bp = uns._get_Xs(vlm_system, "Bp")

        # Midpoints of bound vortices
        ApA = (Ap .+ A)/2
        AB = (A .+ B)/2
        BBp = (B .+ Bp)/2

        # Evaluate VPM on each midpoint
        Xs = vcat(ApA, AB, BBp)

        # Evaluate VLM on each midpoint
        Vvlm = vlm.Vind.(Ref(vlm_system), Xs; t=t, ign_col=true, ign_infvortex=true)

        # Evaluate Vinf on each midpoint
        Vinfs = Vinf.(Xs, t)

        # Evaluate kinematic velocity on each node
        Vtran = uns._Vkinematic(vlm_system, prev_vlm_system, dt; t=t,
                                                    targetX=["Ap", "A", "B", "Bp"])

        # Pre-allocate memory
        Ftot = [zeros(3) for i in 1:m]
        if debug
            Vvpms, Vvlms, Vtranmids = ([zeros(3) for i in 1:m] for j in 1:3)
        end

        # Bound vortices to include in force calculation
        vortices = include_trailingboundvortex ? (1:3) : (2:2)  # 1==ApA, 2==AB, 3==BBp



        if !vlm_vortexsheet || KJforce_type=="weighted"

            ## NOTE: Instead of calling the VPM, we use what was calculated
            ## by `solve()`, which includes Rotor-on-VLM velocities
            Vvpm = Vvpm_on_Xs(pfield, Xs; dt=dt)
            #Vvpm = vcat(vlm_system.sol["Vvpm_ApA"], vlm_system.sol["Vvpm_AB"], vlm_system.sol["Vvpm_BBp"])

        elseif vehicle == nothing

            error("KJ force calculation based on vortex sheet has been
                    requested, but vehicle has not been provided.")

        else

            # ----- Estimate VPM velocity (including rotors) on vortex sheet

            # Generate function of static particles
            static_particles_function = generate_static_particle_fun(pfield, pfield,
                                            vehicle,
                                            sigma_vlm_surf, sigma_rotor_surf;
                                            vlm_vortexsheet=vlm_vortexsheet,
                                            vlm_vortexsheet_overlap=vlm_vortexsheet_overlap,
                                            vlm_vortexsheet_distribution=vlm_vortexsheet_distribution,
                                            vlm_vortexsheet_sigma_tbv=vlm_vortexsheet_sigma_tbv)

            # Pre-calculate direction of lifting bound vortices
            BVdir = [zeros(3) for i in 1:m]

            for i in 1:m
                BVdir[i] .= B[i]
                BVdir[i] .-= A[i]
                BVdir[i] ./= norm(BVdir[i])
            end

            # Add static particles representing the vortex sheet
            org_np = vpm.get_np(pfield)
            static_particles_function(pfield)

            # Nullify strength of vortex sheet particles to avoid double accounting for VLM velocity
            for P in vpm.iterate(pfield; include_static=true)

                if P.static[1] && 1 <= P.index[1] <= m

                    # Save strength for future reference
                    for i in 1:3; P.M[i] = P.Gamma[i]; end;

                    # Save index before it gets overwritten by the FMM
                    P.vol[1] = P.index[1]

                    # Zero it out
                    P.Gamma .*= 1e-14
                end
            end

            # Evaluate velocity in entire particle field
            vpm._reset_particles(pfield)
            pfield.UJ(pfield)

            # Collect velocities
            weights = zeros(m)
            Vvpm = [zeros(3) for i in 1:m]

            for P in vpm.iterate(pfield; include_static=true)

                # if P.static[1] && 1 < P.index[1] <= m
                if P.static[1] && 1 <= P.vol[1] <= m

                    # ind = P.index[1]
                    ind = Int(P.vol[1])

                    # Calculate weight of this static particle as projection to lifting BV
                    weight = KJforce_type=="weighted" ? abs(dot(view(P.M, 1:3), BVdir[ind])) :
                             KJforce_type=="averaged" ? 1.0 :
                             error("Invalid KJforce_type. Options are `\"regular\"`, `\"weighted\"`, or `\"averaged\"`; got $(KJforce_type).")

                    # Add weighted velocity
                    for i in 1:3
                        Vvpm[ind][i] += weight*P.U[i]
                    end
                    weights[ind] += weight

                end
            end

            # Divide by total weight to get the final weighted velocity
            Vvpm ./= weights

            Vvpm = vcat(Vvpm, Vvpm, Vvpm)
            # ------
        end


        # Calculate KJ force
        for i in 1:m                 # Iterate over horseshoes
            for j in vortices        # Iterate over bound vortices (1==ApA, 2==AB, 3==BBp)

                # Bound vortex' length vector
                if j==1
                    l = (A[i]-Ap[i])
                elseif j==2
                    l = (B[i]-A[i])
                else
                    l = (Bp[i]-B[i])
                end

                # Bound vortex' midpoint
                # X = Xs[i + m*(j-1)]

                # Kinematic velocity at midpoint
                Vtranmid = (Vtran[i + m*(j-1)] + Vtran[i + m*j])/2

                # Effective velocity at midpoint
                V = Vvpm[i + m*(j-1)] + Vvlm[i + m*(j-1)] + Vinfs[i + m*(j-1)] + Vtranmid
                if Vout != nothing
                    push!(Vout, [Vvpm[i + m*(j-1)], Vvlm[i + m*(j-1)], Vinfs[i + m*(j-1)], Vtranmid])
                end

                # Circulation
                Gamma = vlm_system.sol["Gamma"][i]

                # Normalization factor
                if per_unit_span
                    len = abs((B[i][1]-A[i][1])*spandir[1] + (B[i][2]-A[i][2])*spandir[2] + (B[i][3]-A[i][3])*spandir[3])
                else
                    len = 1
                end
                if lenout != nothing
                    push!(lenout, per_unit_span==false || len>lencrit ? len : -10*lencrit)
                end

                # Kutta–Joukowski theorem: F = rho * V × vecGamma
                if per_unit_span==false || len > lencrit # NOTE: Use lencrit to avoid dividing by zero
                    Ftot[i][1] += rho * Gamma * (V[2]*l[3] - V[3]*l[2]) / len
                    Ftot[i][2] += rho * Gamma * (V[3]*l[1] - V[1]*l[3]) / len
                    Ftot[i][3] += rho * Gamma * (V[1]*l[2] - V[2]*l[1]) / len

                    if debug && j==2
                        Vtranmids[i] .+= Vtranmid
                        Vvpms[i] .+= Vvpm[i + m*(j-1)]
                        Vvlms[i] .+= Vvlm[i + m*(j-1)]
                    end
                end

            end
        end


        if vlm_vortexsheet && KJforce_type!="regular"
            # Remove static particles
            for pi in vpm.get_np(pfield):-1:org_np+1
                vpm.remove_particle(pfield, pi)
            end
        end

        if debug
            # Save Kutta-Joukowski force as a solution field
            vlm._addsolution(vlm_system, (per_unit_span ? "f" : "F")*"rk-vector", deepcopy(Ftot); t=t)
            vlm._addsolution(vlm_system, "Vtranmid-vector", Vtranmids; t=t)
            vlm._addsolution(vlm_system, "Vvpm-vector", Vvpms; t=t)
            vlm._addsolution(vlm_system, "Vvlm-vector", Vvlms; t=t)
        end

        # Add Kutta-Joukowski force to any existing force calculation
        fieldname = per_unit_span ? "ftot" : "Ftot"
        if fieldname in keys(vlm_system.sol)
            Ftot .+= vlm_system.sol[fieldname]
        end

        # Save total force as a solution field
        vlm._addsolution(vlm_system, fieldname, Ftot; t=t)

        return Ftot
    end

    return calc_aerodynamicforce_kuttajoukowski
end



# ----------------- 4) FORCE CALCULATION SETUP --------------------------------
println("\n4) Setting up force calculations...")

# Create unified force calculation function that matches monitor requirements
calc_aerodynamicforce_fun = generate_calc_aerodynamicforce(;
    add_parasiticdrag=include_parasiticdrag,
    add_skinfriction=add_skinfriction,
    airfoilpolar=wing_polar_file,
)

# Initialize forces array for simulation
forces = []

# Kutta-Joukowski force (primary lift mechanism) - simplified approach
kuttajoukowski = generate_aerodynamicforce_kuttajoukowski(
    "weighted",            
    sigma_vlm_surf, 0.000000001,         # No rotor surface
    true, 2.125,                
    uns.g_pressure, sigma_vpm_overwrite;
    vehicle=vehicle,
)
push!(forces, kuttajoukowski)


# Combined force calculation function for simulation runtime
function simulation_calc_aerodynamicforce_fun(vlm_system, args...; per_unit_span=false, optargs...)
    fieldname = per_unit_span ? "ftot" : "Ftot"
    
    # Clear previous force calculations
    if fieldname in keys(vlm_system.sol)
        pop!(vlm_system.sol, fieldname)
    end
    
    Ftot = nothing
    
    # Apply each force component sequentially
    for (i, force) in enumerate(forces)
        try
            Ftot = force(vlm_system, args...; per_unit_span=per_unit_span, optargs...)
        catch e
            println("Warning: Force component $i failed: $e")
        end
    end
    
    return Ftot
end





# ----------------- 5) MONITORS SETUP ------------------------------------------
println("\n5) Setting up monitoring functions...")

# Wing monitor options
wingmonitor_optargs = (
    include_trailingboundvortex=include_trailingboundvortex,
    calc_aerodynamicforce_fun=calc_aerodynamicforce_fun
)

# Generate monitoring functions
monitors = airfoil_monitor(
    vehicle, rho, Vinf, nsteps, save_path;
    add_wings=add_wings,
    wingmonitor_optargs=wingmonitor_optargs
)






# ----------------- 6) WAKE TREATMENT ------------------------------------------
println("\n6) Configuring wake treatment...")

# Removal based on strength of particles
rmv_strength = 2 * 2 / p_per_step * dt / (1 / 12)
minmaxGamma = rmv_strength * [0.0001, 0.15]
wake_treatment_strength = uns.remove_particles_strength(
    minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=10
)

# Removal based on size of particles
minmaxsigma = sigma_vpm_overwrite * [0.1, 6] # [0.1,6]
wake_treatment_sigma = uns.remove_particles_sigma(
    minmaxsigma[1], minmaxsigma[2]; every_nsteps=10 
)

# Remove distant particles
wake_treatment_sphere = uns.remove_particles_sphere(
    (2.5 * wingspan)^2, 1; Xoff=[0.5 * wingspan, 0, 0]
) 

# Combine wake treatments
wake_treatment = uns.concatenate(
    wake_treatment_sphere, 
    wake_treatment_strength, 
    wake_treatment_sigma
)

# Combined runtime function
runtime_function = uns.concatenate(wake_treatment, monitors)






# ----------------- 7) RUN SIMULATION ------------------------------------------
println("\n7) Running simulation...")
println("   Forward velocity: $(vehicle_velocity) m/s")
println("   Angle of attack: $(angle_of_attack)°")
# println("   Reynolds number: $(round(Re, digits=0))")

# Execute simulation
uns.run_simulation(simulation, nsteps;
    # Simulation options
    Vinf=Vinf,
    rho=rho, mu=mu, 
    tquit=tquit,
    
    # Solver options
    p_per_step=p_per_step,
    max_particles=max_particles,
    max_static_particles=vlm_vortexsheet_maxstaticparticle,
    vpm_integration=vpm_integration,
    vpm_viscous=vpm_viscous,
    vpm_SFS=vpm_SFS,
    # vpm_fmm = vpm.FMM(; p=4, ncrit=100, theta=0.4, phi=0.5),
    sigma_vlm_surf=sigma_vlm_surf,
    sigma_vlm_solver = sigma_vlm_solver, 
    sigma_rotor_surf=0.000000001,
    sigma_vpm_overwrite=sigma_vpm_overwrite,
    sigmafactor_vpmonvlm=sigmafactor_vpmonvlm,
    vlm_vortexsheet=vlm_vortexsheet,
    vlm_vortexsheet_overlap=vlm_vortexsheet_overlap,
    vlm_vortexsheet_distribution=vlm_vortexsheet_distribution,
    vlm_vortexsheet_sigma_tbv=vlm_vortexsheet_sigma_tbv,
    vlm_rlx=vlm_rlx,
    vlm_init=true,
    wake_coupled=wake_coupled,
    hubtiploss_correction=nothing,
    shed_starting=shed_starting,
    shed_unsteady=shed_unsteady,
    unsteady_shedcrit=unsteady_shedcrit,
    extra_runtime_function=monitors,
    
    # Output options
    save_horseshoes=save_horseshoes,
    save_path=save_path,
    run_name=run_name,
    save_wopwopin=false,
)

println("   Simulation completed successfully!")



# =============================================================================
# FLUID DOMAIN COMPUTATION
# =============================================================================


# ----------------- FLUID DOMAIN PARAMETERS --------------------------------
# Grid parameters for fluid domain computation
L_domain        = wingspan                  # (m) reference length for domain
dx_domain       = L_domain/64               # (m) cell size in each direction
dy_domain       = L_domain/64
dz_domain       = L_domain/64 

# Domain bounds
Pmin_domain     = L_domain*[ 0, -1.0, -1.0]   # (m) minimum bound of outline cuboid
Pmax_domain     = L_domain*[ 2, 1.0, 1.0]   # (m) maximum bound of outline cuboid
 

NDIVS_domain    = ceil.(Int, (Pmax_domain .- Pmin_domain)./[dx_domain, dy_domain, dz_domain])
nnodes_domain   = prod(NDIVS_domain .+ 1)        # Total number of nodes

# VPM settings for fluid domain
maxparticles_domain = Int(2.5e6 + nnodes_domain) # 1.5e06
fmm_domain      = vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.3)
f_sigma_domain  = 0.5                       # Smoothing of node particles
maxsigma_domain = L_domain/10               # Maximum particle size was 20
maxmagGamma_domain = Inf                    # Maximum vortex strength

# File naming for fluid domain
pfield_prefix   = "airfoil_pfield"     
staticpfield_prefix = "airfoil_staticpfield" 
fdom_prefix     = "airfoil_fdom"           


 
if compute_fluid_domain
    
    # ----------------- FLUID DOMAIN SETUP -------------------------------------
    
    # Time steps to process (select representative steps)
    # Process every 10th step for the last cycle
    last_cycle_start = 100
    fluid_domain_nums = collect(last_cycle_start:10:500)

    println("   Processing $(length(fluid_domain_nums)) time steps for fluid domain")
    println("   Domain grid: $(NDIVS_domain) cells, $(nnodes_domain) nodes")
    
    # Fluid domain save path
    fdom_save_path = joinpath(save_path, "fluid_domain")
    
    # Create save path
    gt.create_path(fdom_save_path, false)  # No prompt for automated execution
    
    # Grid orientation (aligned with vehicle)
    Oaxis_domain = gt.rotation_matrix2(0, angle_of_attack, 0)
    
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
    println("\n9) Computing fluid domain...")
    
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
    
    println("   Fluid domain computation completed!")
    println("   Results saved to: $fdom_save_path")
    
else
    println("\n   Skipping fluid domain computation (compute_fluid_domain = false)")
end






# =============================================================================
# POST-PROCESSING AND VISUALIZATION
# =============================================================================

println("\n10) Simulation summary:")
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
    println("\n11) Opening results in ParaView...")
    
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
println("Stationary Airfoil Simulation Complete")
println("="^80)