#pragma once

#include <cuda_runtime.h>

/**
 * @brief Simple CUDA kernel that demonstrates basic kernel launch
 */
__global__ void simpleKernel(int n);

/**
 * @brief Wrapper function to launch the simple kernel
 */
void launchSimpleKernel();
