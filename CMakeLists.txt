cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(TF_VERSION 1.12.0)

include_directories("${PROJECT_SOURCE_DIR}/versions/${TF_VERSION}/include"
        "${PROJECT_SOURCE_DIR}/third-party/GSL/include")

link_directories("${PROJECT_SOURCE_DIR}/versions/${TF_VERSION}/lib")


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main dl Threads::Threads)

if (WIN32)
  set(TF_LIBNAME tensorflow.dll)
endif (WIN32)

if (MSVC)
  set(TF_LIBNAME tensorflow.dll)
endif (MSVC)

if (UNIX)
  set(TF_LIBNAME libtensorflow.so)
endif (UNIX)
