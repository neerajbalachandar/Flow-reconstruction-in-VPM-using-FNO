#--------------------------------------------------------------------------------NEERAJ BALACHANDAR-----------------------------------------------------------------------------------------------------
__authors__ = "NEERAJ BALACHANDAR"
__contact__ = "neerajbalachandar@gmail.com"


function airfoil_maneuver(; disp_plot = true, 
                         add_wings = true, 
                         vehicle_velocity::Real=0.01,
                         angle_of_attack::Real=30.0)


    chord = 1.0
    wingspan = 4.0

    # reduced_frequency=3.0
    # h_amplitude=0.10
    # pitch_amplitude=25.0
    # phase_offset=π/2

    # freq = (reduced_frequency)/(2*pi*chord)
    # plunge_amp = h_amplitude*chord
    # pitch_amp = pitch_amplitude * (π/180)


    vehicle_velocity_func(t) = [vehicle_velocity, 0, 0]
    vehicle_angle_func(t) = [0, angle_of_attack, 0]  

    airfoil_angle(t) = begin
    # Return [Ax, Ay, Az] in degrees
    [0.0, 30.0, 0.0]
    end
        
    airfoil_angles = (airfoil_angle,)   
    rotor_rpms = ()                     
    
    maneuver = uns.KinematicManeuver(airfoil_angles, rotor_rpms, 
                                 vehicle_velocity_func, vehicle_angle_func)

    
    if disp_plot
        uns.plot_maneuver(maneuver)
    end
    
    return maneuver
end