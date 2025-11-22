cmake_minimum_required(VERSION 3.10)
project(CUDAConv)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall")

# include directories
## internal
include_directories(image_loader)
include_directories(conv)
## external
include_directories(${CMAKE_SOURCE_DIR}/external/argparse)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main
    main.cpp
    image_loader/image_loader.cpp
    conv/conv.cpp
)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)
