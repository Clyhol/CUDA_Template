#include <iostream>
#include <cuda_runtime.h>
#include "kernels.cuh"

void checkCudaError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in " << context << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    std::cout << "=== CUDA Practice Program ===" << std::endl;

    // Print CUDA device information
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    }
    
    // Set device
    checkCudaError(cudaSetDevice(0), "cudaSetDevice");
    
    // Call a simple kernel example
    std::cout << "\n=== Running CUDA Kernel ===" << std::endl;
    launchSimpleKernel();
    
    // Synchronize to ensure all kernel operations are complete
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    
    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}