# Benchmark executable
cmake_minimum_required(VERSION 3.16)

# Helper function to create benchmark target for a specific implementation
function(add_benchmark_target IMPL_NAME IMPL_DEFINE IMPL_DIR)
    add_executable(bench_gemm_${IMPL_NAME}
        src/main.cpp
        src/matrix_gen.cpp
        src/runner.cpp
        src/csv_writer.cpp
    )

    target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_definitions(bench_gemm_${IMPL_NAME} PRIVATE ${IMPL_DEFINE})
    target_compile_features(bench_gemm_${IMPL_NAME} PRIVATE cxx_std_20)

    # Link with appropriate GeMM library
    if(IMPL_NAME STREQUAL "01")
        target_link_libraries(bench_gemm_${IMPL_NAME} PRIVATE GeMM_naive)
        target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../01-naive
        )
    elseif(IMPL_NAME STREQUAL "02")
        target_link_libraries(bench_gemm_${IMPL_NAME} PRIVATE GeMM_coded)
        target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../02-coded
        )
    elseif(IMPL_NAME STREQUAL "03")
        target_link_libraries(bench_gemm_${IMPL_NAME} PRIVATE GeMM_blocked)
        target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../03-blocked
        )
    elseif(IMPL_NAME STREQUAL "04")
        target_link_libraries(bench_gemm_${IMPL_NAME} PRIVATE GeMM_neon)
        target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../04-neon
        )
    else()
        target_link_libraries(bench_gemm_${IMPL_NAME} PRIVATE GeMM_final)
        target_include_directories(bench_gemm_${IMPL_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../05-final
        )
    endif()
endfunction()

# Create benchmark targets for all implementations
add_benchmark_target(01 USE_IMPL_01 01-naive)
add_benchmark_target(02 USE_IMPL_02 02-coded)
add_benchmark_target(03 USE_IMPL_03 03-blocked)
add_benchmark_target(04 USE_IMPL_04 04-neon)
add_benchmark_target(05 USE_IMPL_05 05-final)

# Default target (05-final)
add_custom_target(bench_gemm ALL DEPENDS bench_gemm_05)
