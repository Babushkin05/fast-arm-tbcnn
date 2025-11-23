add_library(GeMM_blocked GeMM.cpp)
target_include_directories(GeMM_blocked PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_blocked test_blocked.cpp)
target_link_libraries(test_blocked PRIVATE GeMM_blocked Catch2::Catch2WithMain)

include(CTest)
add_test(NAME GeMM_Blocked_Test COMMAND test_blocked)
