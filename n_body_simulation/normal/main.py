import cupy as cp
from n_body_simulation.normal.simulation import simulate, simulate_with_trajectories
from utils.visualization import visualize_simulation_plotly, visualize_simulation_matplotlib, visualize_simulation_with_controls

# Number of particles
N = 10  # 1 star + 9 planets
dim = 3  # 3D space

# Other parameters
G = 1.0
theta = 0.5
dt = 1000  # Smaller time step for better stability
softening = 1e-2
steps = 1000

# Positions: Star at the center, planets in a disk-like configuration
positions = cp.zeros((N, dim))
positions[1:, 0] = cp.linspace(1e5, 5e5, N - 1)  # Planets along the x-axis

# Velocities: Tangential velocities for circular orbits
velocities = cp.zeros((N, dim))
star_mass = 1e6  # Mass of the central star
for i in range(1, N):
    r = positions[i, 0]  # Distance from the star
    velocities[i, 1] = cp.sqrt(G * star_mass / r)  # Tangential velocity along y-axis

# Masses: Central star is much heavier than the planets
masses = cp.ones(N)
masses[0] = star_mass  # Central mass is much heavier

trajectories = simulate_with_trajectories(positions, masses, velocities, steps=steps, dt=0.01, method="leapfrog", theta=theta, G=G, softening=softening)
visualize_simulation_with_controls(trajectories, downsample_factor=10) # Use adequate downsample factor for better performance
