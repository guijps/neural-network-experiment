# Variables
BUILD_DIR := build

# Default target
all: build

# Create the build directory and compile the project
build:
	@echo "BUILD_DIR is set to: $(BUILD_DIR)"
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make
	@echo "Build Done"

# Clean the build directory
clean:
	@echo "Cleaning build directory: $(BUILD_DIR)"
	rm -rf $(BUILD_DIR)
	@echo "Clean Done"