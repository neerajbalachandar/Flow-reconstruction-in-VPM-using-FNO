#-----------------------------------------------------------NEERAJ BALACHANDAR-----------------------------------------------------------------------------------------------------
__authors__ = "NEERAJ BALACHANDAR"
__contact__ = "neerajbalachandar@gmail.com"

#-----------------------------------------------------------FLOWUnsteady_openvsp.jl-----------------------------
function airfoil_geometry()
    
    # Read degenerate geometry components
    components = uns.read_degengeom(joinpath(@__DIR__, "airfoil_DegenGeom.csv"))

    airfoil_component = components[findfirst(c -> c.name == "airfoil", components)]
    
    airfoil = uns.import_vsp(airfoil_component; geomType="wing")
  
    println("airfoil geometry imported")

#-------------------------------------------------------------FLOWUnsteady_vehicle_vlm_unsteady.jl--------------------
    
    # Build comprehensive wing system with proper hierarchy
    system = uns.vlm.WingSystem()
    
    # Add wings to main vehicle system
    uns.vlm.addwing(system, "airfoil", airfoil)
    
    # Configure tilting systems as WingSystem subgroups
    tilting_system = uns.vlm.WingSystem()
    
    # Reference wings in tilting systems
    uns.vlm.addwing(tilting_system, "Main", airfoil)
    
    # Initialize vehicle with proper kinematic hierarchy
    vehicle = uns.UVLMVehicle(
        system;                      # Single WingSystem representing the full vehicle
        tilting_systems = (tilting_system,),  # Tuple of one tilting system
        rotor_systems = (),                   # Tuple of rotor groups (empty if none)
        vlm_system = system,                  # System solved with VLM
        wake_system = system,                 # System shedding wake
        V = zeros(3),                        # Initial linear velocity
        W = zeros(3),                        # Initial angular velocity
        prev_data = [
            deepcopy(system),                # Previous VLM system state
            deepcopy(system),                # Previous wake system state
            ()                              # Empty rotor systems
        ]
    )

    
    # Validate panel counts with proper indexing
    for (i, wing) in enumerate(vehicle.system.wings)
        m_panels = uns.vlm.get_m(wing)
        println("Wing $i: $m_panels panels")
        if m_panels < 50
            @warn "Low panel count ($m_panels) detected in wing $i"
        end
    end
    
    println("UVLM vehicle configuration successful")
    return vehicle

    #EXPORTING---------------
    # Define output directory
    save_geom_path = "/home/dysco/FLOWUnsteady/FLOWUnsteady/Unsteady Aerodynamics/Simulations/codes/data_gen/poisson pde/dataset1/geom"

    # Clean/create directory
    if isdir(save_geom_path)
        rm(save_geom_path; recursive=true, force=true)
    end
    mkdir(save_geom_path)

    # Set freestream velocity for visualization context
    uns.vlm.setVinf(system, Vinf)

    # Save VTK files and get master PVD file name
    pvd_file = uns.save_vtk(vehicle, ""; 
                            path=save_geom_path,
                            save_wopwopin=false) # Acoustic input

    # Open in ParaView using the master PVD file
    paraview_file = joinpath(save_geom_path, pvd_file)
    run(`paraview --data=$paraview_file`, wait=false)

end
#----------------------------------------------------------------------------------------------------------------------