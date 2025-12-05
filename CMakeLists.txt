cmake_minimum_required(VERSION 3.10)
project(CUDAConv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fopenmp")

# include directories
include_directories(${CMAKE_SOURCE_DIR}/external/argparse)

# Find packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(main
    main.cpp
)

# Set main.cpp as a CUDA source file
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
)