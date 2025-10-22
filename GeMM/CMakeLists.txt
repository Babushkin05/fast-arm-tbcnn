cmake_minimum_required(VERSION 3.16)
project(MatrixMultiplication CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(01-naive)
# add_subdirectory(algorithms/blocked)
