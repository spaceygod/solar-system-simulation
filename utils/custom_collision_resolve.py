import logging
logging.basicConfig(level=logging.INFO)

def custom_collision_resolve(sim_pointer, collision):
    """
    Custom collision resolution for merging particles.
    """
    # Access the simulation object
    sim = sim_pointer.contents

    # Get the two colliding particles
    p1 = sim.particles[collision.p1]
    p2 = sim.particles[collision.p2]

    # Log details of the collision
    # logging.info(f"Collision detected between Particle {collision.p1} and Particle {collision.p2}")
    # logging.info(f"They have radius {p1.r} and {p2.r} respectively")
    # logging.info(f"Pre-Collision Masses: p1={p1.m}, p2={p2.m}")

    # Calculate the combined mass
    new_mass = p1.m + p2.m

    # If the mass is zero, do nothing
    if p1.m == 0 or p2.m == 0:
        return 1  # Return 1 to indicate failure

    # Calculate the new position (center of mass)
    new_x = (p1.m * p1.x + p2.m * p2.x) / new_mass
    new_y = (p1.m * p1.y + p2.m * p2.y) / new_mass
    new_z = (p1.m * p1.z + p2.m * p2.z) / new_mass

    # logging.info(f"New Position after Collision: {new_x}, {new_y}, {new_z}")

    # Calculate the new velocity (momentum conservation)
    new_vx = (p1.m * p1.vx + p2.m * p2.vx) / new_mass
    new_vy = (p1.m * p1.vy + p2.m * p2.vy) / new_mass
    new_vz = (p1.m * p1.vz + p2.m * p2.vz) / new_mass

    # Calculate the new radius based on volume conservation
    new_radius = (p1.r**3 + p2.r**3)**(1/3)

    # logging.info(f"New Radius after Collision: {new_radius}")

    # Update p1 with new properties
    p1.m = new_mass
    p1.x = new_x
    p1.y = new_y
    p1.z = new_z
    p1.vx = new_vx
    p1.vy = new_vy
    p1.vz = new_vz
    p1.r = new_radius

    # Mark p2 for removal by setting its mass to zero
    p2.m = 0

    return 0  # Return 0 to indicate success