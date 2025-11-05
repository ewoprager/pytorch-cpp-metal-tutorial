cmake_minimum_required(VERSION 3.18)

# List your sources
set(SOURCES
    cpp_extension.mm
    add_tensors.metal
)

project(pytorch-metal-cpp-tutorial LANGUAGES CXX OBJCXX)
# Enable Objective-C++ as a language
enable_language(OBJCXX)

# Get the PyTorch include directories
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths())"
                OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE RESULT
)

string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if (NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.include_paths(): ${RESULT}")
endif ()
string(REPLACE "[" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "]" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "'" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "," ";" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})

include_directories(${TORCH_INCLUDE_DIRS})

# Define the library or executable
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Use C++17 or newer
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Frameworks for Metal and Foundation
find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)

target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      ${METAL_LIB}
                      ${FOUNDATION_LIB}
)

# Extra compiler flags if needed
target_compile_options(${PROJECT_NAME} PRIVATE
                       -ObjC++       # ensure Objective-C++ mode
                       -fobjc-arc    # use ARC for Objective-C memory management
)
