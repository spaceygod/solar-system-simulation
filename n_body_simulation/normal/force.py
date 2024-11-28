import cupy as cp
from n_body_simulation.normal.tree import build_tree

# Force calculation using the Barnes-Hut tree
def compute_force(node, position, mass, theta, G, softening):
    """
    Recursively computes the gravitational force acting on a particle from a node in the Barnes-Hut tree.
    
    Parameters:
    - node: Current node in the Barnes-Hut tree.
    - position: Position of the particle.
    - mass: Mass of the particle.
    - theta: Threshold for Barnes-Hut approximation.
    - G: Gravitational constant.
    - softening: Softening parameter to avoid division by zero.
    
    Returns:
    - Total force acting on the particle.
    """

    if node.is_leaf(): # A leaf node is a node with 0 or 1 particles (so no children)
        # Compute pairwise force directly
        diff = node.center_of_mass - position
        dist = cp.linalg.norm(diff) + softening # Avoid division by zero
        return G * node.mass * mass * diff / (dist**3) # F_vector = G * m1 * m2 / r^2 * r_hat = G * m1 * m2 / r^2 * (r_vector / r)
    
    # Check the Barnes-Hut condition
    size_over_dist = node.size / cp.linalg.norm(node.center - position) # Side length of cube / distance between cube center and particle
    # Means the cube is far enough away from the particle that it "looks like a single particle"
    if size_over_dist < theta:
        ### Explanation of the above condition:
        # The total gravitational potential from a group of particles can be approximated as a multipole series expansion:
        # The first term is the monopole term, which represents the gravitational potential of the group as a single particle at the center of mass
        # Higher-order terms are the dipole, quadrupole, etc. terms, which represent the deviations from the monopole approximation.
        # When s/d is small, the monopole term dominates and the higher-order terms can be neglected.
        # Approximate using the center of mass
        diff = node.center_of_mass - position
        dist = cp.linalg.norm(diff) + softening
        return G * node.mass * mass * diff / (dist**3)
    else:
        # Traverse children as we can't approximate now because the cube is too close to the particle and cannot be approximated as a monopole
        force = cp.zeros(3)
        for child in node.children:
            force += compute_force(child, position, mass, theta, G, softening)
        return force
    
def compute_all_forces(positions, masses, theta, G, softening):
    """
    Computes the gravitational forces on all particles using the Barnes-Hut tree.
    
    Parameters:
    - positions: Current positions of all particles (N x 3 CuPy array).
    - masses: Masses of all particles (N CuPy array).
    - theta: Threshold for Barnes-Hut approximation.
    - G: Gravitational constant.
    - softening: Softening parameter to avoid division by zero.
    
    Returns:
    - forces: Total forces acting on each particle (N x 3 CuPy array).
    """
    # Build the Barnes-Hut tree
    root_center = cp.mean(positions, axis=0)
    root_size = cp.max(positions) - cp.min(positions)
    root = build_tree(positions, masses, root_center, root_size)

    # Initialize the forces array
    forces = cp.zeros_like(positions)

    # Compute forces for each particle
    for i in range(len(positions)):
        forces[i] = compute_force(root, positions[i], masses[i], theta, G, softening)

    return forces
