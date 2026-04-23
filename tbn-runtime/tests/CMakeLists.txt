# Tests CMakeLists.txt

# Find Catch2 (optional for now)
find_package(Catch2 3 QUIET)

if(Catch2_FOUND)
    # Create test executable
    add_executable(tbn_tests
        test_basic.cpp
    )

    target_link_libraries(tbn_tests
        PRIVATE
        tbn
        Catch2::Catch2WithMain
    )

    # Add tests to CTest
    include(CTest)
    include(Catch)
    catch_discover_tests(tbn_tests)
else()
    message(STATUS "Catch2 not found - tests will not be built")
endif()

# Create a simple test executable without Catch2
add_executable(tbn_simple_test
    test_basic.cpp
)

# Create separate test for quantized operators
add_executable(test_quantized_operators
    test_quantized_operators.cpp
)

# Create test for optimized Conv2D
add_executable(test_conv2d_optimized
    test_conv2d_optimized.cpp
)

# Create test for ONNX Conv speed
add_executable(test_onnx_conv_speed
    test_onnx_conv_speed.cpp
)

# Create test for pooling
add_executable(test_pooling
    test_pooling.cpp
)

target_link_libraries(tbn_simple_test
    PRIVATE
    tbn
)

target_link_libraries(test_quantized_operators
    PRIVATE
    tbn
)

target_link_libraries(test_conv2d_optimized
    PRIVATE
    tbn
)

target_link_libraries(test_onnx_conv_speed
    PRIVATE
    tbn
)

target_link_libraries(test_pooling
    PRIVATE
    tbn
)

# Set test properties
set_target_properties(tbn_simple_test PROPERTIES
    EXCLUDE_FROM_ALL TRUE
)

set_target_properties(test_quantized_operators PROPERTIES
    EXCLUDE_FROM_ALL TRUE
)

set_target_properties(test_conv2d_optimized PROPERTIES
    EXCLUDE_FROM_ALL TRUE
)

# Add custom target for running tests
add_custom_target(run_tests
    COMMAND tbn_simple_test
    COMMAND test_quantized_operators
    DEPENDS tbn_simple_test test_quantized_operators
    COMMENT "Running TBN Runtime tests"
)

# Add test for memory leaks (if valgrind is available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(test_memory
        COMMAND ${VALGRIND_EXECUTABLE} --leak-check=full --show-leak-kinds=all ./tbn_simple_test
        DEPENDS tbn_simple_test
        COMMENT "Running memory leak tests"
    )
endif()