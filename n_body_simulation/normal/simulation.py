import cupy as cp
from tqdm import tqdm
from n_body_simulation.normal.tree import build_tree
from n_body_simulation.normal.force import compute_force, compute_all_forces
from utils.integration_methods import euler_update, leapfrog_update, runge_kutta_update

# Main simulation loop
def simulate(positions, masses, velocities, steps, dt, method="leapfrog", theta=0.5, G=1.0, softening=1e-2):
    """
    Main simulation loop for N-body simulation using a parallelized Barnes-Hut algorithm.
    
    Parameters:
    - positions: Initial positions of particles (N x 3 CuPy array).
    - masses: Masses of particles (N CuPy array).
    - velocities: Initial velocities of particles (N x 3 CuPy array).
    - steps: Number of simulation steps.
    - dt: Time step for numerical integration.
    - method: Integration method (default "leapfrog").
    - theta: Barnes-Hut parameter for approximation (default 0.5).
    - G: Gravitational constant (default 1.0).
    - softening: Softening parameter to avoid division by zero (default 1e-2).
    """

    # Initialize progress bar
    with tqdm(total=steps, desc="Simulation Progress", unit="step") as pbar:
        for step in range(steps):
            # Build the tree
            root_center = cp.mean(positions, axis=0)
            root_size = cp.max(positions) - cp.min(positions) # maximum distance between particles along any dimension, which ensures that the root node (cube) covers all particles
            root = build_tree(positions, masses, root_center, root_size)

            # Compute forces
            if method != "runge-kutta": # For RK4, forces are computed inside the integration function
                # Compute forces
                forces = cp.zeros_like(positions)
                for i in range(len(positions)):
                    forces[i] = compute_force(root, positions[i], masses[i], theta, G, softening)

            # Choose the integration method
            if method == "euler":
                positions, velocities = euler_update(positions, velocities, forces, masses, dt)
            elif method == "leapfrog":
                if step == 0:  # Initialize half-step velocities for Leapfrog
                    velocities += 0.5 * forces * dt / masses[:, None]
                positions, velocities = leapfrog_update(positions, velocities, forces, masses, dt)
            elif method == "runge-kutta":
                # For RK4, pass the force computation as a function because RK4 requires multiple force evaluations (k1, k2, k3, k4)
                positions, velocities = runge_kutta_update(positions, velocities, lambda pos: compute_all_forces(pos, masses), masses, dt)

            # Update the progress bar
            pbar.update(1)

    return positions, velocities
    

def simulate_with_trajectories(positions, masses, velocities, steps, dt, method="leapfrog", theta=0.5, G=1.0, softening=1e-2):
    """
    Simulation loop that tracks trajectories of particles for visualization.
    
    Parameters:
    - positions: Initial positions of particles (N x 3 CuPy array).
    - masses: Masses of particles (N CuPy array).
    - velocities: Initial velocities of particles (N x 3 CuPy array).
    - steps: Number of simulation steps.
    - dt: Time step for numerical integration.
    - method: Integration method (default "leapfrog").
    - theta: Barnes-Hut parameter for approximation (default 0.5).
    - G: Gravitational constant (default 1.0).
    - softening: Softening parameter to avoid division by zero (default 1e-2).
    
    Returns:
    - trajectories: List of positions (N x 3 CuPy arrays) at each timestep.
    """
    trajectories = []  # To store positions at each step

    # Initialize progress bar
    with tqdm(total=steps, desc="Simulation Progress", unit="step") as pbar:
        for step in range(steps):
            # Store current positions for visualization
            trajectories.append(positions.get())

            # Build the tree
            root_center = cp.mean(positions, axis=0)
            root_size = cp.max(positions) - cp.min(positions)  # Ensure the root node covers all particles
            root = build_tree(positions, masses, root_center, root_size)

            # Compute forces
            if method != "runge-kutta":  # For RK4, forces are computed inside the integration function
                forces = cp.zeros_like(positions)
                for i in range(len(positions)):
                    forces[i] = compute_force(root, positions[i], masses[i], theta, G, softening)

            # Choose the integration method
            if method == "euler":
                positions, velocities = euler_update(positions, velocities, forces, masses, dt)
            elif method == "leapfrog":
                if step == 0:  # Initialize half-step velocities for Leapfrog
                    velocities += 0.5 * forces * dt / masses[:, None]
                positions, velocities = leapfrog_update(positions, velocities, forces, masses, dt)
            elif method == "runge-kutta":
                positions, velocities = runge_kutta_update(
                    positions, velocities, lambda pos: compute_all_forces(pos, masses), masses, dt
                )

            # Update the progress bar
            pbar.update(1)
            
    return [pos for pos in trajectories]
