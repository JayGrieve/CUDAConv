# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -fopenmp -Wall

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Target executable
TARGET = image_loader

# Source file
SRC = image_loader.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(SRC) -o $(TARGET) $(OPENCV_LIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

.PHONY: all run clean
