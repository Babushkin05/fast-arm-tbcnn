add_library(GeMM_naive GeMM.cpp)
target_include_directories(GeMM_naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_naive test_naive.cpp)
target_link_libraries(test_naive PRIVATE GeMM_naive Catch2::Catch2WithMain)

include(CTest)
add_test(NAME GeMM_Naive_Test COMMAND test_naive)
