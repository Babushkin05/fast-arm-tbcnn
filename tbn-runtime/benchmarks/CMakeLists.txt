# Benchmarks CMakeLists.txt

# Find Google Benchmark (optional)
find_package(benchmark QUIET)

if(benchmark_FOUND)
    # Create benchmark executable
    add_executable(tbn_benchmarks
        gemm_benchmark.cpp
        conv2d_benchmark.cpp
        quantization_benchmark.cpp
    )

    target_link_libraries(tbn_benchmarks
        PRIVATE
        tbn
        benchmark::benchmark
    )

    # Set properties
    set_target_properties(tbn_benchmarks PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Add custom target
    add_custom_target(benchmarks
        DEPENDS tbn_benchmarks
        COMMENT "Building TBN Runtime benchmarks"
    )

    # Add run target
    add_custom_target(run_benchmarks
        COMMAND tbn_benchmarks
        DEPENDS tbn_benchmarks
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        COMMENT "Running TBN Runtime benchmarks"
    )
else()
    message(STATUS "Google Benchmark not found - benchmarks will not be built")
endif()

# Create a simple performance test (only if the source file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/simple_performance.cpp")
    add_executable(simple_performance_test
        simple_performance.cpp
    )

    target_link_libraries(simple_performance_test
        PRIVATE
        tbn
    )

    set_target_properties(simple_performance_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        EXCLUDE_FROM_ALL TRUE
    )

    # Add performance profiling target (if available)
    find_program(GPROF_EXECUTABLE gprof)
    if(GPROF_EXECUTABLE)
        add_custom_target(profile_performance
            COMMAND ${CMAKE_COMMAND} -E echo "Profiling performance test..."
            COMMAND ./simple_performance_test
            COMMAND ${GPROF_EXECUTABLE} simple_performance_test gmon.out > performance_profile.txt
            DEPENDS simple_performance_test
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            COMMENT "Profiling performance test"
        )
    endif()
endif()

# Add quantized GEMM benchmark
add_executable(benchmark_quantized_gemm
    benchmark_quantized_gemm.cpp
)

target_link_libraries(benchmark_quantized_gemm
    PRIVATE
    tbn
)

set_target_properties(benchmark_quantized_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add simple NEON test
add_executable(simple_neon_test
    simple_neon_test.cpp
)

target_link_libraries(simple_neon_test
    PRIVATE
    tbn
)

set_target_properties(simple_neon_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)