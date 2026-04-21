# Examples CMakeLists.txt

add_executable(simple_example
    simple_example.cpp
)

target_link_libraries(simple_example
    PRIVATE
    tbn
)

# Set properties
set_target_properties(simple_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add more examples as they are created
# add_executable(advanced_example
#     advanced_example.cpp
# )

# target_link_libraries(advanced_example
#     PRIVATE
#     tbn
# )

# Add custom target to build all examples
add_custom_target(examples
    DEPENDS simple_example
    COMMENT "Building TBN Runtime examples"
)

# Add install target for examples
install(TARGETS simple_example
    RUNTIME DESTINATION bin
    COMPONENT examples
)

# Add a target to run the example
add_custom_target(run_example
    COMMAND simple_example
    DEPENDS simple_example
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    COMMENT "Running simple example"
)