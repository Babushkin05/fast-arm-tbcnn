add_library(GeMM_final GeMM.cpp)
target_include_directories(GeMM_final PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(GeMM_final PUBLIC cxx_std_20)

add_executable(test_final test_final.cpp)
target_link_libraries(test_final PRIVATE GeMM_final Catch2::Catch2WithMain)
target_compile_features(test_final PRIVATE cxx_std_20)

include(CTest)
add_test(NAME GeMM_Final_Test COMMAND test_final)
