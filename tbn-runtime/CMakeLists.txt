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

# Find ONNX - try different locations based on platform
if(APPLE)
    set(ONNX_INCLUDE_DIRS /opt/homebrew/include)
    set(ONNX_LIBRARIES
        /opt/homebrew/lib/libonnx.dylib
        /opt/homebrew/lib/libonnx_proto.dylib
    )
elseif(UNIX)
    # Linux - try pkg-config first
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNX onnx)
        if(ONNX_FOUND)
            message(STATUS "ONNX found via pkg-config")
        endif()
    endif()

    # Fallback to standard paths
    if(NOT ONNX_FOUND)
        find_path(ONNX_INCLUDE_DIRS NAMES onnx/onnx_pb.h
            PATHS /usr/include /usr/local/include)
        find_library(ONNX_LIB NAMES onnx libonnx
            PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
        find_library(ONNX_PROTO_LIB NAMES onnx_proto libonnx_proto
            PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
        if(ONNX_LIB AND ONNX_PROTO_LIB)
            set(ONNX_LIBRARIES ${ONNX_LIB} ${ONNX_PROTO_LIB})
            set(ONNX_FOUND TRUE)
        endif()
    endif()

    if(NOT ONNX_FOUND)
        message(WARNING "ONNX not found - some features may be unavailable")
        set(ONNX_INCLUDE_DIRS "")
        set(ONNX_LIBRARIES "")
    endif()
endif()

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
)

# Set properties
set_target_properties(tbn PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/tbn/tbn.hpp
    POSITION_INDEPENDENT_CODE ON  # Required for Python bindings
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
    # Build C++ extension module as _tbn (internal)
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

    # Create python package directory structure
    set(PYTHON_PKG_DIR ${CMAKE_BINARY_DIR}/python/tbn)
    file(MAKE_DIRECTORY ${PYTHON_PKG_DIR})

    # Set output name to '_tbn' (internal module, imported by __init__.py)
    set_target_properties(tbn_python PROPERTIES
        OUTPUT_NAME _tbn
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_PKG_DIR}
    )

    # Copy Python package files
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tbn/__init__.py
        ${PYTHON_PKG_DIR}/__init__.py
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/python/tbn/quantization.py
        ${PYTHON_PKG_DIR}/quantization.py
        COPYONLY
    )

    message(STATUS "Python package will be built at: ${CMAKE_BINARY_DIR}/python")
endif()
