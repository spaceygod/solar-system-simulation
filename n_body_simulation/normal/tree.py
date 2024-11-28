import cupy as cp

# Define a Node class for the Barnes-Hut tree
class Node:
    def __init__(self, center, size):
        self.center = center  # Center of the cube (3D space)
        self.size = size      # Size of the cube (length of one side)
        self.mass = 0.0       # Total mass in this region
        self.center_of_mass = cp.zeros(3)  # Center of mass
        self.children = []    # Child nodes
        self.particles = []   # Particles in this region

    def is_leaf(self):
        return len(self.children) == 0


# Build the Barnes-Hut tree
def build_tree(positions, masses, center, size, depth=0):
    """
    Recursively builds the Barnes-Hut tree for efficient force computation.
    
    Parameters:
    - positions: Positions of all particles.
    - masses: Masses of all particles.
    - center: Center of the current cube.
    - size: Size of the current cube.
    - depth: Current depth of the tree.
    
    Returns:
    - The root node of the tree.
    """
    
    node = Node(center, size)
    if len(positions) == 1:  # Base case: one particle
        node.mass = masses[0]
        node.center_of_mass = positions[0]
        node.particles = positions
        return node

    # Subdivide into 8 octants (for 3D)
    half_size = size / 2.0 # Half the size of the cube is the size of an octant

    # Each octant has a center offset from the center of the cube of +- half_size/2 along each dimension
    offsets = cp.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], # Corners of a 2 x 2 x 2 cube centered at the origin
                        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * half_size / 2.0
    
    ### Explanation of the above offsets calculation:
    # These corners are multiplied by half_size/2 to get the corners of the cube centered at the origin with half_size/2 side length
    # Which gives exactly the offset of the centers of the 8 octants from the center of the cube

    for offset in offsets:
        sub_center = center + offset
        # Identify all particles in this octant
        indices = cp.all((positions >= sub_center - half_size / 2) & (positions < sub_center + half_size / 2), axis=1) # Check which particles are between the "back left bottom" and "front right top" corners of the octant
        # The above check is done such that we do not unncessarily split an octant even further into more octants if there are no particles in it
        if cp.any(indices):
            # Obviously this just recursively builds the tree further and further until each octant has either 0 (empty node) or 1 (base case) particle
            child = build_tree(positions[indices], masses[indices], sub_center, half_size, depth + 1)
            node.children.append(child)

    # Compute mass and center of mass for this node
    node.mass = cp.sum(masses)
    if node.mass > 0:
        # Center of mass r_cm = sum(m_i * r_i) / sum(m_i)
        node.center_of_mass = cp.sum(positions.T * masses, axis=1) / node.mass

    return node
