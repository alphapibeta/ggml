cmake_minimum_required(VERSION 3.12)

project(HeatEquationSolver)

# Set the target name
set(TEST_TARGET heat_equation_solver)

# Add the executable with the source files
add_executable(${TEST_TARGET}
    main.cpp
    src/HeatEquationSolver.cpp
    src/FiniteDifference.cpp

)

# Include the headers
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# Set C++ standard
set_target_properties(${TEST_TARGET} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

# Detect and configure OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${TEST_TARGET} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${TEST_TARGET} PRIVATE USE_OPENMP)
else()
    message(WARNING "OpenMP not found. CPU parallelization will be disabled.")
endif()

# Detect and configure CUDA and related libraries
if (GGML_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Set architecture-specific CUDA and NVTX settings
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        find_library(NVML_LIBRARIES NAMES nvidia-ml PATHS ${CUDAToolkit_LIBRARY_DIR})
        find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDAToolkit_LIBRARY_DIR})
        if (NVML_LIBRARIES)
            target_link_libraries(${TEST_TARGET} PRIVATE ${NVML_LIBRARIES})
            target_compile_definitions(${TEST_TARGET} PRIVATE GGML_USE_NVML)
        endif()
        if (NVTX_LIBRARY)
            target_link_libraries(${TEST_TARGET} PRIVATE ${NVTX_LIBRARY})
        endif()
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDAToolkit_LIBRARY_DIR})
        if (NVTX_LIBRARY)
            target_link_libraries(${TEST_TARGET} PRIVATE ${NVTX_LIBRARY})
        endif()
    endif()

    target_include_directories(${TEST_TARGET} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_sources(${TEST_TARGET} PRIVATE src/cuda/HeatEquationKernels.cu)
    target_compile_definitions(${TEST_TARGET} PRIVATE GGML_USE_CUDA)
    
    # Enable separable compilation for CUDA
    set_target_properties(${TEST_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TEST_TARGET} PRIVATE CUDA::cudart)
endif()
