# Makefile for compiling gmonitor

# Compiler
CXX = g++

# Output executable
TARGET = gmonitor

# Source files
SRC = gmonitor.cpp

# Include directories
INCLUDES = -I/usr/local/cuda-12.3/targets/x86_64-linux/include

# Path to the NVML shared library
NVML_LIB = /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# Compiler and linker flags
CXXFLAGS = -O2 -std=c++17

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(NVML_LIB)

# Clean up
clean:
	rm -f $(TARGET)