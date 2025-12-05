# Compiler settings
NVCC = nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_75 -Xcompiler -fopenmp
CXX_FLAGS = -std=c++17 -O3 -fopenmp -Wall

# Include directories
INCLUDES = -I./image_loader -I./conv -I./external/argparse

# OpenCV flags (using pkg-config)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# If opencv4 is not found, try opencv
ifeq ($(OPENCV_CFLAGS),)
    OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
    OPENCV_LIBS = $(shell pkg-config --libs opencv)
endif

# OpenMP flags
OPENMP_FLAGS = -fopenmp

# Target executable
TARGET = imreader

# Source files
CUDA_SRC = imreader.cpp
CPP_SRC = image_loader/image_loader.cpp

# Object files
CUDA_OBJ = imreader.o
CPP_OBJ = image_loader.o

# Default target
all: $(TARGET)

# Compile image_loader.cpp
image_loader.o: $(CPP_SRC)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

# Compile imreader.cpp with CUDA (force treat as CUDA source with -x cu)
imreader.o: $(CUDA_SRC)
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

# Link everything together
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(OPENCV_LIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET) $(CUDA_OBJ) $(CPP_OBJ)

# Rebuild everything
rebuild: clean all

.PHONY: all run clean rebuild
