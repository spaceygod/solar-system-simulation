import rebound
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.custom_collision_resolve import custom_collision_resolve

try:
    from utils.visualization import visualize_simulation_with_controls
except ImportError:
    print("Couldn't import the visualization function in utils.visualization. Make sure the path is correct and if it is run 'py -m  n_body_simulation.normal.Nbodyproblem.py' from the main directory.")
    exit()

# Number of particles
N_planetesimals = 10000
M_sun = 1.0
G = 1 # Use normalized units because we already use AU, yr, Msun and Rebound expects G = 1 (see https://rebound.readthedocs.io/en/latest/units/)
sigma0 = 1e-5 # Check what value would be good

years = 50000
box_limit = 25000

# Simulator Function
def setup_simulation():
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.configure_box(box_limit, 1, 1)
    sim.boundary = "open"
    sim.collision = "linetree"
    sim.collision_resolve = custom_collision_resolve
    sim.add(m=M_sun)  # Sun
    
    # Add particles
    a = np.sqrt(np.random.uniform(0.1**2, 50**2, N_planetesimals)) # Semi-major axis (uniformly distributed in 2D space)
    e = np.random.uniform(0, 0.1, N_planetesimals) # Eccentricity
    inc = np.random.uniform(0, 0.05, N_planetesimals) # Inclination
    omega = np.random.uniform(0, 2 * np.pi, N_planetesimals) # Argument of pericenter
    Omega = np.random.uniform(0, 2 * np.pi, N_planetesimals) # Longitude of ascending node
    f = np.random.uniform(0, 2 * np.pi, N_planetesimals) # True anomaly
    
    # Radius of particles
    mean_log_r = np.log(0.1)
    sigma_log_r = 0.4
    radius = np.random.lognormal(mean=mean_log_r, sigma=sigma_log_r, size=N_planetesimals)

    # Assume density (rho) is constant for all particles
    density = 1e-3  # Example: density in Msun / AU^3
    mass = (4/3) * np.pi * (radius**3) * density  # Mass proportional to volume

    for i in range(N_planetesimals):
        # print(a[i], e[i], inc[i], omega[i], Omega[i], f[i], mass[i], radius[i])
        sim.add(a=a[i], e=e[i], inc=inc[i], omega=omega[i], Omega=Omega[i], f=f[i], m=mass[i], r=radius[i])
    
    sim.move_to_com()
    sim.integrator = "leapfrog"
    sim.dt = 0.1
    return sim

# Visualiser function
def plot_simulation(sim, title, filename=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    particles = sim.particles[1:]  # Exclude sun
    x = [p.x for p in particles]
    y = [p.y for p in particles]
    radii = [p.r for p in particles]  # Get radii of particles
    sizes = [r**2 * 1000 for r in radii]  # Scale marker size with squared radius

    ax.scatter(x, y, s=sizes, c='blue', alpha=0.5, label="Particles")
    ax.scatter(sim.particles[0].x, sim.particles[0].y, color='yellow', s=100, label="Central Mass")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.legend()
    if filename:
        plt.savefig(filename)
    # plt.show()
    plt.close(fig)

# Main simulation
sim = setup_simulation()

# Initial state
plot_simulation(sim, "Initial State", filename="images/n_body_initial.png")

def disk_force(sim_pointer):
    # Access the simulation object
    sim = sim_pointer.contents

    # Calculate current sigma
    t = sim.t  # Current simulation time
    sigma = sigma0 * np.exp(-t / years)  # Exponential decay of sigma over time

    # Apply the same force to all particles (excluding the Sun)?
    forces = G * sigma / 2
    for particle in sim.particles[1:]:  # Exclude the central mass (Sun)
        particle.ax += forces

sim.additional_forces = disk_force

# Collect positions over time for interactive visualization
trajectories = []
radii_trajectory = []

print("Simulating for interactive visualization...")
for t in tqdm(range(years)):
    sim.integrate(t)
    # Visualization
    positions = np.array([[p.x, p.y] for p in sim.particles if p.m > 0])
    radii = np.array([p.r for p in sim.particles if p.m > 0])  # Filter only non-zero-mass particles
    trajectories.append(positions)
    radii_trajectory.append(radii)

# Final state
print(f"Particles left after {years} years: {len(sim.particles)}")

visualize_simulation_with_controls(trajectories, radii_trajectory, downsample_factor=1, dynamic_axis_limits=True)

name = "After " + str(years) + " years"
plot_simulation(sim, name, filename="images/n_body_simulation.png")
