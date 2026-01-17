# --------------------------------------------------------------------------------NEERAJ BALACHANDAR-----------------------------------------------------------------------------------------------------
__authors__ = "NEERAJ BALACHANDAR"
__contact__ = "neerajbalachandar@gmail.com"

function airfoil_monitor(vehicle, rho, Vinf, nsteps, save_path;
                                    add_wings=true,
                                    wingmonitor_optargs=[])
    
    # Create output directory
    isdir(save_path) || mkpath(save_path)


    function Vvpm_on_Xs(pfield::vpm.ParticleField, Xs::Array{T, 1}; static_particles_fun=(args...)->nothing, dt=0, fsgm=1) where {T}

        if length(Xs)!=0 && vpm.get_np(pfield)!=0
            # Omit freestream
            Uinf = pfield.Uinf
            pfield.Uinf = (t)->zeros(3)

            org_np = vpm.get_np(pfield)             # Original particles

            # Singularize particles to correct tip loss
            # NOTE: This doesn't include static particles, but there shouldn't be
            #       any in the field at this point anyways
            if abs(fsgm) != 1
                for P in vpm.iterator(pfield)
                    P.sigma .*= fsgm
                end
            end

            # Add static particles
            static_particles_fun(pfield, pfield.t, dt)

            sta_np = vpm.get_np(pfield)             # Original + static particles

            # Add probes
            for X in Xs
                add_probe(pfield, X)
            end

            # Evaluate velocity field
            pfield.UJ(pfield)

            # Retrieve velocity at probes
            Vvpm = [Array(P.U) for P in vpm.iterator(pfield; start_i=sta_np+1)]

            # Remove static particles and probes
            for pi in vpm.get_np(pfield):-1:(org_np+1)
                vpm.remove_particle(pfield, pi)
            end

            # De-singularize particles
            if abs(fsgm) != 1
                for P in vpm.iterator(pfield)
                    P.sigma ./= fsgm
                end
            end

            # Restore freestream
            pfield.Uinf = Uinf
        else
            Vvpm = [zeros(3) for i in 1:length(Xs)]
        end

        return Vvpm
    end

    add_probe(pfield::vpm.ParticleField, X) = vpm.add_particle(pfield, X, zeros(3), 1e-6; vol=0)


    function generate_calc_aerodynamicforce(; add_parasiticdrag=true,
                                              add_skinfriction=true,
                                              airfoilpolar="xf-n0012-il-500000-n5.csv",
                                              parasiticdrag_args=(),
                                              )


        # Aerodynamic force from Kutta-Joukowski's theorem
        kuttajoukowski = generate_aerodynamicforce_kuttajoukowski("regular",
                                                                    nothing, nothing,
                                                                    false, nothing,
                                                                    nothing, vehicle)


        function calc_aerodynamicforce(vlm_system, args...; per_unit_span=false, optargs...)

            # Delete any previous force field
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

            # === START OF FIX ===
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
            # === END OF FIX ===

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



            if !vlm_vortexsheet || KJforce_type=="regular"

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

    # -------------------- WING MONITORS ---------------------------------------
    if add_wings
    
        system = vehicle.system
        #Dimensionalised
        b_ref = 1.0
        ar_ref = 1.0
        qinf = 1.0

        # Force component directions (Z-up, X-forward)
        L_dir = [0, 0, 1]  
        D_dir = [-1, 0, 0] 

        calc_aerodynamicforce_fun = generate_calc_aerodynamicforce(;
        add_parasiticdrag=true,
        add_skinfriction=true,
        airfoilpolar="naca0012.csv",
        parasiticdrag_args=()
        )

        monitors = []

        wing_names = ["airfoil"]
        wings = [vlm.get_wing(vehicle.vlm_system, name) for name in wing_names]

        for (wing, name) in zip(wings, wing_names)
            wing_monitor = uns.generate_monitor_wing(
                wing, Vinf, b_ref, ar_ref, rho, qinf, nsteps;
                calc_aerodynamicforce_fun=calc_aerodynamicforce_fun,
                save_path=save_path,
                run_name=name,
                figname="$(name) Forces",
                L_dir=L_dir,
                D_dir=D_dir
            )
            push!(monitors, wing_monitor)
        end
    end

    # # -------------------- FLAPPING KINEMATICS --------------------------------
    # function monitor_flapping(pfield, vehicle, t, dt)
    #     # Track flapping angles
    #     fore_angle = vehicle.tilting_systems[1].angle(t)
    #     hind_angle = vehicle.tilting_systems[2].angle(t)
        
    #     # Store in global data structure
    #     global flapping_data = get!(flapping_data, t, Dict())
    #     flapping_data[t] = (fore=fore_angle, hind=hind_angle)
        
    #     return false  # Don't stop simulation
    # end

    # # -------------------- VORTEX METRICS ----------------------------------
    # function monitor_vortex_formation(pfield, vehicle, t, dt)
    #     # Calculate LEV characteristics
    #     global vortex_data = get!(vortex_data, t, Dict())
        
    #     # Implementation for vortex tracking would go here
    #     # (Requires access to flow field particles)
        
    #     return false
    # end

    # -------------------- STANDARD MONITORS -----------------------------------
    # Vehicle state variables
    state_monitor = uns.generate_monitor_statevariables(;
        save_path=save_path,
        figname="vehicle_states"
    )

    # Flow field enstrophy
    enstrophy_monitor = uns.generate_monitor_enstrophy(;
        save_path=save_path,
        figname="flow_energy"
    )

    # Combine all monitors
    all_monitors = uns.concatenate(
        state_monitor,
        enstrophy_monitor,
        monitors...
    )

    return all_monitors
end