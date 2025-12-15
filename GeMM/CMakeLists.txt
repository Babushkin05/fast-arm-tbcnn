cmake_minimum_required(VERSION 3.16)
project(MatrixMultiplication CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(01-naive)
add_subdirectory(02-coded)
add_subdirectory(03-blocked)
add_subdirectory(04-neon)
