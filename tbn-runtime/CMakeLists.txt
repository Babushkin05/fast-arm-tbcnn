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
    # Find ONNX library
    find_package(ONNX REQUIRED)
    if(NOT ONNX_FOUND)
        message(STATUS "ONNX not found, downloading...")

        # Download and build ONNX
        include(FetchContent)
        FetchContent_Declare(
            onnx
            GIT_REPOSITORY https://github.com/onnx/onnx.git
            GIT_TAG v1.14.1
        )

        # Set ONNX options
        set(ONNX_BUILD_TESTS OFF CACHE BOOL "Build ONNX tests" FORCE)
        set(ONNX_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

        FetchContent_MakeAvailable(onnx)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE TBN_SOURCES
    src/*.cpp
    src/*.cc
)

# For now, exclude the real ONNX parser and use the simple one
list(REMOVE_ITEM TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser.cpp)

# Add the simple ONNX parser
list(APPEND TBN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_integration/onnx_parser_simple.cpp)

# Create library
add_library(tbn ${TBN_SOURCES})
target_link_libraries(tbn PUBLIC Threads::Threads)

# Link ONNX if enabled
if(TBN_ONNX_RUNTIME AND TARGET onnx_proto)
    target_link_libraries(tbn PUBLIC onnx_proto)
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