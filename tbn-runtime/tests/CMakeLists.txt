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

target_link_libraries(tbn_simple_test
    PRIVATE
    tbn
)

# Set test properties
set_target_properties(tbn_simple_test PROPERTIES
    EXCLUDE_FROM_ALL TRUE
)

# Add custom target for running tests
add_custom_target(run_tests
    COMMAND tbn_simple_test
    DEPENDS tbn_simple_test
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