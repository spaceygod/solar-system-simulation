# Define Euler integration
def euler_update(positions, velocities, forces, masses, dt):
    """
    Performs an Euler update.
    
    Parameters:
    - positions: Current positions of particles (N x 3 CuPy array).
    - velocities: Current velocities of particles (N x 3 CuPy array).
    - forces: Current forces acting on particles (N x 3 CuPy array).
    - masses: Masses of particles (N CuPy array).
    - dt: Time step.
    
    Returns:
    - Updated positions and velocities.
    """

    # Update velocities
    velocities += forces * dt / masses[:, None]
    # Update positions
    positions += velocities * dt
    return positions, velocities


# Define Leapfrog integration
def leapfrog_update(positions, velocities, forces, masses, dt):
    """
    Performs a Leapfrog update.
    
    Parameters:
    - positions: Current positions of particles (N x 3 CuPy array).
    - velocities: Current velocities of particles (N x 3 CuPy array).
    - forces: Current forces acting on particles (N x 3 CuPy array).
    - masses: Masses of particles (N CuPy array).
    - dt: Time step.
    
    Returns:
    - Updated positions and velocities.
    """

    # Half-step velocity update
    velocities_half = velocities + 0.5 * forces * dt / masses[:, None]
    # Full-step position update
    positions += velocities_half * dt
    # Full-step force update happens externally
    return positions, velocities_half

def runge_kutta_update(positions, velocities, forces_func, masses, dt):
    """
    Performs a 4th-order Runge-Kutta update.
    
    Parameters:
    - positions: Current positions of particles (N x 3 CuPy array).
    - velocities: Current velocities of particles (N x 3 CuPy array).
    - forces_func: Function to compute forces given positions.
    - masses: Masses of particles (N CuPy array).
    - dt: Time step.
    
    Returns:
    - Updated positions and velocities.
    """
    
    k1_v = forces_func(positions) / masses[:, None]
    k1_r = velocities

    k2_v = forces_func(positions + 0.5 * k1_r * dt) / masses[:, None]
    k2_r = velocities + 0.5 * k1_v * dt

    k3_v = forces_func(positions + 0.5 * k2_r * dt) / masses[:, None]
    k3_r = velocities + 0.5 * k2_v * dt

    k4_v = forces_func(positions + k3_r * dt) / masses[:, None]
    k4_r = velocities + k3_v * dt

    velocities += dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
    positions += dt * (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6.0
    return positions, velocities


# Feel free to add more integration methods here