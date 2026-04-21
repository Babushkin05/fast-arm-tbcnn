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
    # Try to find ONNX library
    find_package(ONNX QUIET)

    if(NOT ONNX_FOUND AND NOT TARGET onnx_proto)
        message(STATUS "ONNX not found, will use dummy implementation for now")
        message(STATUS "To enable real ONNX support, install ONNX or set ONNX_ROOT")

        # For now, we'll use the dummy implementation
        set(TBN_ONNX_RUNTIME FALSE)
    else()
        message(STATUS "ONNX found, real ONNX support enabled")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE TBN_SOURCES
    src/*.cpp
    src/*.cc
)

# Handle ONNX integration source files
if(TBN_ONNX_RUNTIME)
    # Use real ONNX parser if available, otherwise use dummy
    if(TARGET onnx_proto)
        message(STATUS "Using real ONNX parser implementation")
        # Remove the simple parser and keep the real one
        list(REMOVE_ITEM TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser_simple.cpp)
        # Add compile definition for real ONNX support
        add_compile_definitions(TBN_ONNX_RUNTIME_ENABLED)
    else()
        message(STATUS "Using dummy ONNX parser implementation")
        # Remove the real parser and use the simple one
        list(REMOVE_ITEM TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser.cpp)
        list(APPEND TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser_simple.cpp)
    endif()
else()
    # ONNX runtime disabled, use only dummy implementation
    message(STATUS "ONNX runtime disabled, using dummy implementation")
    list(REMOVE_ITEM TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser.cpp)
    list(APPEND TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser_simple.cpp)
endif()

# Create library
add_library(tbn ${TBN_SOURCES})
target_link_libraries(tbn PUBLIC Threads::Threads)

# Link ONNX if enabled
if(TBN_ONNX_RUNTIME AND TARGET onnx_proto)
    target_link_libraries(tbn PUBLIC onnx_proto protobuf::libprotobuf)
    target_compile_definitions(tbn PUBLIC TBN_ONNX_RUNTIME_ENABLED)
endif()

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