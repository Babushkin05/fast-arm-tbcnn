cmake_minimum_required(VERSION 3.16)
project(tbn-runtime VERSION 0.1.0 LANGUAGES CXX)

# Set build output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(TBN_BUILD_TESTS "Build tests" ON)
option(TBN_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TBN_BUILD_EXAMPLES "Build examples" ON)
option(TBN_BUILD_PYTHON "Build Python bindings" ON)

# ============================================================================
# Dependencies
# ============================================================================
find_package(Threads REQUIRED)

# pybind11 for Python bindings
if(TBN_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Try to find pybind11, if not found fetch it
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    message(STATUS "Python3: ${Python3_VERSION}")
    message(STATUS "pybind11 found: ${pybind11_FOUND}")
endif()

# Find protobuf
find_package(Protobuf REQUIRED)

# Find ONNX - manually since CMake config may not be available
set(ONNX_INCLUDE_DIRS /opt/homebrew/include)
set(ONNX_LIBRARIES
    /opt/homebrew/lib/libonnx.dylib
    /opt/homebrew/lib/libonnx_proto.dylib
)

message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")
message(STATUS "ONNX found at: ${ONNX_INCLUDE_DIRS}")

# ============================================================================
# TBN Library
# ============================================================================

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${ONNX_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE TBN_SOURCES
    src/*.cpp
    src/*.cc
    ../GeMM/05-final/GeMM.cpp
)

# Remove the simple parser - we always use real ONNX now
list(REMOVE_ITEM TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser_simple.cpp)

# Create library
add_library(tbn ${TBN_SOURCES})

target_link_libraries(tbn PUBLIC
    Threads::Threads
    ${ONNX_LIBRARIES}
    ${Protobuf_LIBRARIES}
)

target_compile_definitions(tbn PUBLIC
    TBN_ONNX_RUNTIME_ENABLED
    ONNX_ML
)

# Include directories for ONNX
target_include_directories(tbn PUBLIC
    ${ONNX_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    /opt/homebrew/include  # For onnx/ headers
)

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

# ============================================================================
# Python Bindings
# ============================================================================
if(TBN_BUILD_PYTHON)
    pybind11_add_module(tbn_python python/bindings.cpp)

    target_link_libraries(tbn_python PRIVATE tbn)

    target_compile_definitions(tbn_python PRIVATE
        TBN_ONNX_RUNTIME_ENABLED
        ONNX_ML
    )

    target_include_directories(tbn_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNX_INCLUDE_DIRS}
    )

    # Set output name to 'tbn' (not 'tbn_python')
    set_target_properties(tbn_python PROPERTIES
        OUTPUT_NAME tbn
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )

    message(STATUS "Python module will be built at: ${CMAKE_BINARY_DIR}/python")
endif()
