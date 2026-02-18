# Makefile for mixed CPU (GCC) and CUDA (NVCC) project

# Compiler settings
CXX = g++
NVCC = nvcc
TARGET = math2code_mtt

# Directories
SRC_DIR = src
CUDA_DIR = src_cuda
BUILD_DIR = build
BIN_DIR = "build/bin"

# CUDA architecture (adjust for your GPU, e.g., sm_75 for RTX 2080, sm_86 for RTX 3090)
# Use 'nvcc --list-gpu-code' to see supported architectures
CUDA_ARCH = sm_89

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -g -O0
CUDA_FLAGS = -std=c++20 -arch=$(CUDA_ARCH) -g -G -O0

# CUDA paths (adjust if needed)
CUDA_PATH = /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart

# Include paths
INCLUDES = -I$(SRC_DIR) -I$(CUDA_DIR) $(CUDA_INC)

# Source files (recursively find all .cpp and .cu files)
CPP_SOURCES = $(shell find $(SRC_DIR) -type f -name '*.cpp' 2>/dev/null)
CUDA_SOURCES = $(shell find $(CUDA_DIR) -type f -name '*.cu' 2>/dev/null)
CUDA_HEADERS = $(shell find $(CUDA_DIR) -type f -name '*.cuh' 2>/dev/null)

# Object files (preserve directory structure in build dir)
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/cpp/%.o,$(CPP_SOURCES))
CUDA_OBJECTS = $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/cuda/%.o,$(CUDA_SOURCES))

# All object files
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Default target
all: directories $(BIN_DIR)/$(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Link everything together (use g++ for linking)
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(OBJECTS) -o $@ $(CUDA_LIB)
	@echo "Build complete: $@"

# Compile CPU source files with g++ (create subdirectories as needed)
$(BUILD_DIR)/cpp/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling CPU code: $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files with nvcc (create subdirectories as needed)
$(BUILD_DIR)/cuda/%.o: $(CUDA_DIR)/%.cu $(CUDA_HEADERS)
	@echo "Compiling CUDA code: $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Run the program
run: all
	@echo "Running $(TARGET)..."
	@./$(BIN_DIR)/$(TARGET)

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Clean and rebuild
rebuild: clean all

# Show compiler versions
info:
	@echo "=== Compiler Information ==="
	@$(CXX) --version | head -n 1
	@$(NVCC) --version | grep "release"
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo "=========================="

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build the project"
	@echo "  make run      - Build and run the project"
	@echo "  make clean    - Remove build files"
	@echo "  make rebuild  - Clean and rebuild"
	@echo "  make info     - Show compiler information"
	@echo "  make help     - Show this help message"

.PHONY: all directories run clean rebuild info help
