# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:26:20 2024

@author: Harsh Mishra
"""

import rebound
import numpy as np
import matplotlib.pyplot as plt

# Number of particles
N_planetesimals = 10000
M_sun = 1.0
G = 6.67e-11 #gravitation constant, 
#we have to check whether this works good together with AU, do the order of magnitude match?
#or should G be 1?
sigma0 = 1e-5 #to check what value would be good
#homogenous disk btw


# Simulator Function
def setup_simulation():
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')  # Astronomical units 
    sim.collision = "line" # Collision detection along line
    sim.collision_resolve = "merge" # Collision resolves, merges 2 particles without conserving energy
    sim.add(m=M_sun)  # Sun
    
    # Add N particles to the simulation
    for i in range(N_planetesimals):
        a = np.random.uniform(0.1, 50)  # Semi-major axis 
        e = np.random.uniform(0, 0.1)   # Eccentricity
        
        #check what this means
        inc = np.random.uniform(0, 0.05)  # Inclination (radians)
        omega = np.random.uniform(0, 2*np.pi)  # Argument of pericenter
        Omega = np.random.uniform(0, 2*np.pi)  # Longitude of ascending node
        f = np.random.uniform(0, 2*np.pi)      # True anomaly
        sim.add(a=a, e=e, inc=inc, omega=omega, Omega=Omega, f=f, m=0)
        print(i+1)
    
    sim.move_to_com()  # CENTER OF MASS
    sim.integrator = "leapfrog"  # INTEGRATOR
    sim.dt = 0.1  # Time step in years
    return sim

# Visualiser function
def plot_simulation(sim, title, filename=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    particles = sim.particles[1:]  # Exclude sun
    x = [p.x for p in particles]
    y = [p.y for p in particles]
    ax.scatter(x, y, s=1, c='blue', alpha=0.5, label="Particles")
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
    plt.show()

# Main simulation
sim = setup_simulation()
plot_simulation(sim, "Initial State")

# Define additional Disk Force
#btw now force for sun, but later on only for neptune right?
# def diskForce(reb_sim):
#     for i in range(1, len(sim.particles)):  #  use reb_sim.particles
#                                                 #  if there are less than N_particles due to collission
#         sim.particles[i].ax += G * sigma0 / 2  #all planets exclude the sun right?     
#     #also which direction should this be, or 
#     #like statistical physics, 1/3 in every direction, or in the direction of movement
#     #aangezien M_sun de massa is delen we die weg voor de versnelling neem ik aan?


#         #disk oneindig of eindig maar wel uitleggen
#         #disk dichtheid updaten
# sim.additional_forces = diskForce
# Evolve the simulation
years = 180
sim.integrate(years)  
name = "After " + str(years) + " years"
plot_simulation(sim, name, filename="n_body_simulation.png")