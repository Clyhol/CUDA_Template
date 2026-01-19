#include <stdio.h>
#include "kernels.cuh"

/**
 * @brief Simple CUDA kernel that prints information from each thread
 * @param n Size of the data
 */
__global__ void simpleKernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Thread %d in block %d (local idx: %d)\n", threadIdx.x, blockIdx.x, idx);
    }
}

/**
 * @brief Wrapper function to safely launch the simple kernel
 */
void launchSimpleKernel() {
    const int n = 256;  // Total number of threads
    const int blockSize = 32;  // Threads per block
    const int gridSize = (n + blockSize - 1) / blockSize;  // Number of blocks
    
    printf("Launching kernel with %d blocks and %d threads per block\n", gridSize, blockSize);
    simpleKernel<<<gridSize, blockSize>>>(n);
}
