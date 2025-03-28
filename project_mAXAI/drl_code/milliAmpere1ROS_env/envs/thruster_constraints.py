import numpy as np

def enforce_thruster_constraints(action):

    # Define angle constraints in radians
    constraints = [
        (np.pi/2, np.pi),       # Thruster 1: [90°, 180°]
        (-np.pi, -np.pi/2),     # Thruster 2: [-180°, -90°]
        (-np.pi/2, 0),          # Thruster 3: [-90°, 0°]
        (0, np.pi/2)            # Thruster 4: [0°, 90°]
    ]

    # Define opposite quadrants for each thruster
    opposite_quadrants = [
        (-np.pi/2, 0),          # Opposite of Thruster 1: [-90°, 0°]
        (0, np.pi/2),           # Opposite of Thruster 2: [0°, 90°]
        (np.pi/2, np.pi),       # Opposite of Thruster 3: [90°, 180°]
        (-np.pi, -np.pi/2)      # Opposite of Thruster 4: [-180°, -90°]
    ]

    constrained_actions = np.copy(action)
    thrusters = np.zeros(4)
    angles = np.zeros(4)

    # Process each thruster
    for i in range(4):
        x = constrained_actions[i*2]
        y = constrained_actions[i*2 + 1]

        if x == 0 and y == 0:
            # add 0 thrust and angle
            thrusters[i] = 0
            angles[i] = angle_prev[i]
            continue

        angle = np.arctan2(y, x)
        thrust = np.sqrt(x**2 + y**2)

        min_angle, max_angle = constraints[i]
        opp_min_angle, opp_max_angle = opposite_quadrants[i]

        min_minus_45 = ssa(min_angle - np.pi/4)
        max_plus_45 = ssa(max_angle + np.pi/4)

        # CASE 1: Thruster angle is within constraints
        if min_angle <= angle <= max_angle:
            pass
        
        # CASE 2: Thruster angle is in opposite quadrant
        elif opp_min_angle <= angle <= opp_max_angle:
            # Flip angle by 180° and negate thrust
            angle = ssa(angle + np.pi)
            thrust = -thrust

        # CASE 3: Thruster angle is within 45° less than min_angle
        elif min_minus_45 <= angle < ssa_alt(min_angle):
            #print("Thruster", i+1, "is within 45° of constraints")
            if i == 0 or i == 2:
                # Thruster 1 or 3
                angle = min_angle
                thrust = abs(y)
            else:
                # Thruster 2 or 4
                angle = min_angle
                thrust = abs(x)

        # CASE 4: Thruster angle is within 45° more than max_angle + 45°
        elif max_plus_45 < angle < ssa(max_angle)+np.pi/2:
            if i == 0 or i == 2:
                # Thruster 1 or 3
                angle = min_angle
                thrust = -abs(y)
            else:
                # Thruster 2 or 4
                angle = min_angle
                thrust = -abs(x)

        # CASE 5: Thruster angle is within 45° more than max_angle
        elif ssa(max_angle) < angle <= max_plus_45:
            if i == 0 or i == 2:
                # Thruster 1 or 3
                angle = max_angle
                thrust = abs(x)
            else:
                # Thruster 2 or 4
                angle = max_angle
                thrust = abs(y)
        
        # CASE 6: Thruster angle is within 45° less than min_angle - 45°
        else:
            if i == 0 or i == 2:
                # Thruster 1 or 3
                angle = max_angle
                thrust = -abs(x)
            else:
                # Thruster 2 or 4
                angle = max_angle
                thrust = -abs(y)
        
        angles[i] = angle/np.pi*180
        thrusters[i] = thrust

    return thrusters, angles
        

def ssa(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def ssa_alt(angle):
    if angle == -np.pi:
        return np.pi
    return (angle+np.pi) % (2*np.pi) - np.pi

def ssa_degrees(angle):
    return (angle+180) % 360 - 180

def is_in_range(angle, start, end):
    angle = ssa(angle)
    start = ssa(start)
    end = ssa(end)

    if start <= end:
        return start <= angle <= end
    else:
        return angle >= start or angle <= end
    
    
print(ssa_degrees(180))
print(ssa_degrees(-180))
print(ssa(np.pi))
print(ssa(-np.pi))

print(enforce_thruster_constraints([1, 1, 1, 1, 1, 1, 1, 1]))
print(enforce_thruster_constraints([3, 1, 3, 1, 3, 1, 3, 1]))
print(enforce_thruster_constraints([-1, -3, -1, -3, -1, -3, -1, -3]))
print(enforce_thruster_constraints([-3, 1, -3, 1, -3, 1, -3, 1]))
