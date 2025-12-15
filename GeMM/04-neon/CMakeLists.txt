add_library(GeMM_neon GeMM.cpp)
target_include_directories(GeMM_neon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_neon test_neon.cpp)
target_link_libraries(test_neon PRIVATE GeMM_neon Catch2::Catch2WithMain)

include(CTest)
add_test(NAME GeMM_Neon_Test COMMAND test_neon)
