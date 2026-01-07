cmake_minimum_required(VERSION 3.16)
project(microexec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pybind11 (header-only + CMake helpers)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Python extension module
pybind11_add_module(lob_cpp
  src/bindings.cpp
  src/engine.cpp
  src/book.cpp
  src/flow.cpp
)

# Reasonable warnings
target_compile_options(lob_cpp PRIVATE -Wall -Wextra -Wpedantic)
