import rebound
import numpy as np

# Ask the user whether they want an orbit plot or interactive visualization
visualization = input("Do you want an orbit plot (o) or interactive visualization (v)? ")

# Create a new simulation
sim = rebound.Simulation()

# Set units to solar masses, astronomical units (AU), and years
sim.units = ('yr', 'AU', 'Msun')

# Add the Sun
sim.add(m=1.0)  # Mass of the Sun in solar masses

# Add planets (masses in solar masses, positions in AU, velocities in AU/year)
# Data from NASA's planetary fact sheets (approximate initial conditions)
planets = [
    {"m": 3.003e-6, "a": 0.39, "e": 0.205, "inc": 7.0},  # Mercury
    {"m": 4.867e-6, "a": 0.72, "e": 0.007, "inc": 3.39},  # Venus
    {"m": 3.003e-6, "a": 1.0, "e": 0.017, "inc": 0.0},  # Earth
    {"m": 3.213e-7, "a": 1.52, "e": 0.093, "inc": 1.85},  # Mars
    {"m": 9.547e-4, "a": 5.2, "e": 0.049, "inc": 1.31},  # Jupiter
    {"m": 2.858e-4, "a": 9.58, "e": 0.056, "inc": 2.49},  # Saturn
    {"m": 4.365e-5, "a": 19.22, "e": 0.046, "inc": 0.77},  # Uranus
    {"m": 5.15e-5, "a": 30.05, "e": 0.009, "inc": 1.77}   # Neptune
]

for p in planets:
    sim.add(m=p["m"], a=p["a"], e=p["e"], inc=p["inc"])

# Set up the integrator and timestep
sim.integrator = "whfast"  # Wisdom-Holman integrator for efficiency
sim.dt = 0.01  # Time step in years

# Run the simulation for 100 years
times = [i for i in range(101)]  # Output every year
orbits = []
trajectories = []

print("Simulating 100 years...")
for time in times:
    sim.integrate(time)

    if visualization == "o":
        orbits.append(sim.particles)  # Save particle states for visualization
    elif visualization == "v":
        positions = np.array([[p.x, p.y, p.z] for p in sim.particles])
        trajectories.append(positions)

# Visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required for visualization. Install it with 'pip install matplotlib'")
    exit()

# Visualize the simulation
if visualization == "o":
    fig = rebound.OrbitPlot(sim)
    plt.title("Solar System Simulation (100 years)")
    plt.show()
elif visualization == "v":
    # Visualization
    try:
        from utils.visualization import visualize_simulation_with_controls
    except ImportError:
        print("Couldn't import the visualization function in utils.visualization. Make sure the path is correct and if it is run 'py -m  n_body_simulation.rebound.simple_test.py' from the main folder.")
        exit()
    visualize_simulation_with_controls(trajectories, downsample_factor=1)
