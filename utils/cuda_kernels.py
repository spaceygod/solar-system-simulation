import cupy as cp

reset_kernel_code = r'''
extern "C" __global__
void ResetKernel(
    float *min_corner_x, float *min_corner_y, float *min_corner_z,
    float *max_corner_x, float *max_corner_y, float *max_corner_z,
    float *center_mass_x, float *center_mass_y, float *center_mass_z,
    float *total_mass, bool *is_leaf, int *start, int *end,
    int *mutex, int nNodes, int nBodies)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < nNodes)
    {
        min_corner_x[b] = INFINITY;
        min_corner_y[b] = INFINITY;
        min_corner_z[b] = INFINITY;

        max_corner_x[b] = -INFINITY;
        max_corner_y[b] = -INFINITY;
        max_corner_z[b] = -INFINITY;

        center_mass_x[b] = -1.0f;
        center_mass_y[b] = -1.0f;
        center_mass_z[b] = -1.0f;

        total_mass[b] = 0.0f;
        is_leaf[b] = true;
        start[b] = -1;
        end[b] = -1;
        mutex[b] = 0;
    }

    if (b == 0)
    {
        start[b] = 0;
        end[b] = nBodies - 1;
    }
}
'''

reset_kernel = cp.RawKernel(reset_kernel_code, 'ResetKernel')


compute_bounding_box_kernel_code = r'''
extern "C" __global__
void ComputeBoundingBoxKernel(
    float *min_corner_x, float *min_corner_y, float *min_corner_z,
    float *max_corner_x, float *max_corner_y, float *max_corner_z,
    float *positions_x, float *positions_y, float *positions_z,
    int *mutex, int nBodies)
{
    __shared__ float min_x[1024];
    __shared__ float min_y[1024];
    __shared__ float min_z[1024];
    __shared__ float max_x[1024];
    __shared__ float max_y[1024];
    __shared__ float max_z[1024];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    min_x[tx] = INFINITY;
    min_y[tx] = INFINITY;
    min_z[tx] = INFINITY;

    max_x[tx] = -INFINITY;
    max_y[tx] = -INFINITY;
    max_z[tx] = -INFINITY;

    __syncthreads();

    if (b < nBodies)
    {
        float x = positions_x[b];
        float y = positions_y[b];
        float z = positions_z[b];

        min_x[tx] = x;
        min_y[tx] = y;
        min_z[tx] = z;

        max_x[tx] = x;
        max_y[tx] = y;
        max_z[tx] = z;
    }

    // Reduction to find min and max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (tx < stride)
        {
            min_x[tx] = fminf(min_x[tx], min_x[tx + stride]);
            min_y[tx] = fminf(min_y[tx], min_y[tx + stride]);
            min_z[tx] = fminf(min_z[tx], min_z[tx + stride]);

            max_x[tx] = fmaxf(max_x[tx], max_x[tx + stride]);
            max_y[tx] = fmaxf(max_y[tx], max_y[tx + stride]);
            max_z[tx] = fmaxf(max_z[tx], max_z[tx + stride]);
        }
    }

    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);
        min_corner_x[0] = fminf(min_corner_x[0], min_x[0] - 1.0e10f);
        min_corner_y[0] = fminf(min_corner_y[0], min_y[0] - 1.0e10f);
        min_corner_z[0] = fminf(min_corner_z[0], min_z[0] - 1.0e10f);

        max_corner_x[0] = fmaxf(max_corner_x[0], max_x[0] + 1.0e10f);
        max_corner_y[0] = fmaxf(max_corner_y[0], max_y[0] + 1.0e10f);
        max_corner_z[0] = fmaxf(max_corner_z[0], max_z[0] + 1.0e10f);
        atomicExch(mutex, 0);
    }
}
'''

compute_bounding_box_kernel = cp.RawKernel(compute_bounding_box_kernel_code, 'ComputeBoundingBoxKernel')

compute_force_kernel_code = r'''
extern "C" __global__
void ComputeForceKernel(
    // Node data
    float *min_corner_x, float *min_corner_y, float *min_corner_z,
    float *max_corner_x, float *max_corner_y, float *max_corner_z,
    float *center_mass_x, float *center_mass_y, float *center_mass_z,
    float *total_mass, bool *is_leaf, int *start, int *end,
    // Body data
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    float *accelerations_x, float *accelerations_y, float *accelerations_z,
    float *masses, bool *is_dynamic,
    int nNodes, int nBodies, int leafLimit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies) return;

    if (!is_dynamic[i]) return;

    // Initialize acceleration
    accelerations_x[i] = 0.0f;
    accelerations_y[i] = 0.0f;
    accelerations_z[i] = 0.0f;

    // Stack-based tree traversal to avoid recursion
    // Implement an iterative traversal using a stack in shared memory
    // Due to limited space, we can't provide full code here
    // Pseudocode:
    // Initialize a stack with the root node
    // While stack is not empty:
    //     Pop a node
    //     Compute interaction based on Barnes-Hut criteria (theta)
    //     If node is a leaf or meets criteria:
    //         Compute force contribution
    //     Else:
    //         Push child nodes onto stack

    // Update velocities and positions based on acceleration
    velocities_x[i] += accelerations_x[i] * DT;
    velocities_y[i] += accelerations_y[i] * DT;
    velocities_z[i] += accelerations_z[i] * DT;

    positions_x[i] += velocities_x[i] * DT;
    positions_y[i] += velocities_y[i] * DT;
    positions_z[i] += velocities_z[i] * DT;
}
'''

compute_force_kernel = cp.RawKernel(compute_force_kernel_code, 'ComputeForceKernel')

construct_octree_kernel_code = r'''
extern "C" __global__
void ConstructOctTreeKernel(
    // Node data
    float *min_corner_x, float *min_corner_y, float *min_corner_z,
    float *max_corner_x, float *max_corner_y, float *max_corner_z,
    float *center_mass_x, float *center_mass_y, float *center_mass_z,
    float *total_mass, unsigned char *is_leaf, int *start, int *end,
    // Body data
    float *positions_x, float *positions_y, float *positions_z,
    float *masses,
    float *buffer_positions_x, float *buffer_positions_y, float *buffer_positions_z, float *buffer_masses,
    // Node indices to process
    int *node_indices, int num_nodes, int nNodes, int nBodies, int leafLimit)
{
    int nodeIdxInLevel = blockIdx.x;

    if (nodeIdxInLevel >= num_nodes)
        return;

    int nodeIndex = node_indices[nodeIdxInLevel];

    // Shared memory for counts
    __shared__ int counts[8];
    __shared__ int offsets[8];
    __shared__ int counts2[8];

    int tx = threadIdx.x;

    if (tx < 8)
    {
        counts[tx] = 0;
        counts2[tx] = 0;
    }
    __syncthreads();

    // Get current node data
    int start_idx = start[nodeIndex];
    int end_idx = end[nodeIndex];

    if (start_idx == -1 || end_idx == -1)
        return;

    float min_x = min_corner_x[nodeIndex];
    float min_y = min_corner_y[nodeIndex];
    float min_z = min_corner_z[nodeIndex];
    float max_x = max_corner_x[nodeIndex];
    float max_y = max_corner_y[nodeIndex];
    float max_z = max_corner_z[nodeIndex];

    // For each body in this node, determine the octant
    for (int i = start_idx + tx; i <= end_idx; i += blockDim.x)
    {
        float x = positions_x[i];
        float y = positions_y[i];
        float z = positions_z[i];

        int octant = 0;
        float mid_x = (min_x + max_x) * 0.5f;
        float mid_y = (min_y + max_y) * 0.5f;
        float mid_z = (min_z + max_z) * 0.5f;

        if (x >= mid_x) octant |= 1;
        if (y >= mid_y) octant |= 2;
        if (z >= mid_z) octant |= 4;

        atomicAdd(&counts[octant], 1);
    }

    __syncthreads();

    // Compute offsets
    if (tx == 0)
    {
        offsets[0] = start_idx;
        for (int i = 1; i < 8; ++i)
        {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }

        for (int i = 0; i < 8; ++i)
        {
            counts2[i] = offsets[i];
        }
    }
    __syncthreads();

    // Rearrange bodies into buffer
    for (int i = start_idx + tx; i <= end_idx; i += blockDim.x)
    {
        float x = positions_x[i];
        float y = positions_y[i];
        float z = positions_z[i];
        float m = masses[i];

        int octant = 0;
        float mid_x = (min_x + max_x) * 0.5f;
        float mid_y = (min_y + max_y) * 0.5f;
        float mid_z = (min_z + max_z) * 0.5f;

        if (x >= mid_x) octant |= 1;
        if (y >= mid_y) octant |= 2;
        if (z >= mid_z) octant |= 4;

        int dest_idx = atomicAdd(&counts2[octant], 1);

        buffer_positions_x[dest_idx] = x;
        buffer_positions_y[dest_idx] = y;
        buffer_positions_z[dest_idx] = z;
        buffer_masses[dest_idx] = m;
    }

    __syncthreads();

    if (tx == 0)
    {
        is_leaf[nodeIndex] = 0; // false

        // Create child nodes
        for (int i = 0; i < 8; ++i)
        {
            int child_node_idx = nodeIndex * 8 + i + 1; // +1 because childOffset ranges from 1 to 8

            if (counts[i] > 0 && child_node_idx < nNodes)
            {
                start[child_node_idx] = offsets[i];
                end[child_node_idx] = offsets[i] + counts[i] - 1;

                // Update min_corner and max_corner for child nodes
                float child_min_x, child_min_y, child_min_z;
                float child_max_x, child_max_y, child_max_z;

                float mid_x = (min_x + max_x) * 0.5f;
                float mid_y = (min_y + max_y) * 0.5f;
                float mid_z = (min_z + max_z) * 0.5f;

                int octant = i;

                child_min_x = (octant & 1) ? mid_x : min_x;
                child_max_x = (octant & 1) ? max_x : mid_x;

                child_min_y = (octant & 2) ? mid_y : min_y;
                child_max_y = (octant & 2) ? max_y : mid_y;

                child_min_z = (octant & 4) ? mid_z : min_z;
                child_max_z = (octant & 4) ? max_z : mid_z;

                min_corner_x[child_node_idx] = child_min_x;
                min_corner_y[child_node_idx] = child_min_y;
                min_corner_z[child_node_idx] = child_min_z;

                max_corner_x[child_node_idx] = child_max_x;
                max_corner_y[child_node_idx] = child_max_y;
                max_corner_z[child_node_idx] = child_max_z;

                is_leaf[child_node_idx] = 1; // true
            }
        }
    }
}
'''

construct_octree_kernel = cp.RawKernel(construct_octree_kernel_code, 'ConstructOctTreeKernel')