cmake_minimum_required(VERSION 3.16)
project(tbn-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(TBN_BUILD_TESTS "Build tests" ON)
option(TBN_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TBN_BUILD_EXAMPLES "Build examples" ON)
option(TBN_ONNX_RUNTIME "Enable ONNX Runtime integration" ON)

# Find packages
find_package(Threads REQUIRED)

if(TBN_ONNX_RUNTIME)
    # Will be configured later
    # find_package(onnxruntime REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE TBN_SOURCES
    src/*.cpp
    src/*.cc
)

# Create library
add_library(tbn ${TBN_SOURCES})
target_link_libraries(tbn PUBLIC Threads::Threads)

# Set properties
set_target_properties(tbn PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/tbn/tbn.hpp
)

# Installation
install(TARGETS tbn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/tbn
)

install(DIRECTORY include/tbn
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Subdirectories
if(TBN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(TBN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TBN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()