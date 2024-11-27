import numpy as np

### Manual for using the function.
# Checks if planet 1 and planet 2 will collide during the next time step.
# r1 and r2 are the positions and v1 and v2 the velocities of planet 1 and 2 respectively (they should be numpy arrays). 
# R1 and R2 are the radii of planet 1 and 2 respectively.
# delta_t is the time-step size.
# epsilon is a safety parameter making sure all planets near enough eachother are checked.
# resolution_factor determines with what factor the time_step is reduced for the interpolation of the planet trajectories (based on which it is checked wether they collide).

### Comments.
# For a single planet, its position may be interpolated multiple times, depending on how many planets are nearby.
# Thus if the simulation is densely populated with planets, it might be more efficient to interpolate the position of every planet globally in the simulation 
# (instead of locally inside this function), and write a function that only determines whether a collision will take place based on this interpolation 
# (instead of calculating it over and over for a single planet).

def check_collision(r1, v1, R1, r2, v2, R2, delta_t, epsilon = 1.1, resolution_factor = 10):
    # If planet 1 and 2 are nowhere near eachother, it is not necessary to check for a collision.
    # Check if the distance the two planets will travel during the next time step is larger than or equal to the distance between the two planets.
    distance = np.linalg.norm(r1 - r2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if epsilon*delta_t*(norm_v1 + norm_v2) >= distance:
        # Check for a collision between planet 1 and 2.
        
        # Interpolate the position of planet 1
        increment_vector_1 = v1*delta_t / resolution_factor # displacement of planet 1 after delta_t / resolution_factor seconds
        indices_1 = np.arange(resolution_factor).reshape(-1, 1) # column vector [0, 1, 2, ..., n-1]
        r1_interpolated = r1 + indices_1*increment_vector_1

        # Interpolate the position of planet 2
        increment_vector_2 = v2*delta_t / resolution_factor
        indices_2 = np.arange(resolution_factor).reshape(-1, 1)
        r2_interpolated = r2 + indices_2*increment_vector_2

        # Interpolate the distance between planet 1 and 2 
        distance_interpolated = np.linalg.norm(r1_interpolated - r2_interpolated, axis=1)

        print(r1_interpolated)
        print(r2_interpolated)
        print("\n" + str(r1_interpolated - r2_interpolated))
        print(distance_interpolated)

        # Check if a collision occurs
        collision_check = np.any(distance_interpolated <= R1 + R2)

        return collision_check
    else:
        return False