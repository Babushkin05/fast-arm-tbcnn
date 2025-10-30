add_library(GeMM_coded GeMM.cpp)
target_include_directories(GeMM_coded PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_coded test_coded.cpp)
target_link_libraries(test_coded PRIVATE GeMM_coded Catch2::Catch2WithMain)

include(CTest)
add_test(NAME GeMM_Naive_Test COMMAND test_coded)
