cmake_minimum_required(VERSION 3.11)
project(int4_kernel LANGUAGES CXX)

set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
# add_subdirectory(/home/wyj24/infra/cutlass)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})

include_directories("${CMAKE_SOURCE_DIR}")
include_directories(/home/yujin-wa20/projects/cutlass/tools/util/include)
include_directories(/home/yujin-wa20/projects/cutlass/include)
include_directories(int4_kernel/include)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
