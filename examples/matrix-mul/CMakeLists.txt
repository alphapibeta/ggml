cmake_minimum_required(VERSION 3.12)
project(matrix-mul)

# Set the target name
set(TARGET_NAME matrix-mul)

# Add the executable with the source files
add_executable(${TARGET_NAME}
    src/main.cpp
    src/MatrixMulKernels.cu
)

# Include the headers
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set C++ standard
set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Detect and configure OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENMP)
else()
    message(WARNING "OpenMP not found. CPU parallelization will be disabled.")
endif()

# Detect and configure CUDA
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    
    # Set architecture-specific CUDA settings
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDAToolkit_LIBRARY_DIR})
        if (NVTX_LIBRARY)
            target_link_libraries(${TARGET_NAME} PRIVATE ${NVTX_LIBRARY})
        endif()
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDAToolkit_LIBRARY_DIR})
        if (NVTX_LIBRARY)
            target_link_libraries(${TARGET_NAME} PRIVATE ${NVTX_LIBRARY})
        endif()
    endif()

    target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)

    # Enable separable compilation for CUDA
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(FATAL_ERROR "CUDA toolkit not found. The project requires CUDA support.")
endif()
