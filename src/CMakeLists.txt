# ===========================
# CUDA Kernels Library
# ===========================
add_library(cuda_kernels STATIC
    kernels.cu
)

target_include_directories(cuda_kernels 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cuda_kernels 
    PUBLIC 
        CUDA::cudart
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ===========================
# Main Executable
# ===========================
add_executable(main main.cpp)

target_link_libraries(main 
    PRIVATE 
        cuda_kernels
        CUDA::cudart
)

target_include_directories(main 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
