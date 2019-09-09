cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_CXX_STANDARD 11)
set(TF_VERSION 1.6.0)

include_directories("${PROJECT_SOURCE_DIR}/versions/${TF_VERSION}/include"
        "${PROJECT_SOURCE_DIR}/third-party/GSL/include")

link_directories("${PROJECT_SOURCE_DIR}/versions/${TF_VERSION}/lib")

add_executable(main main.cpp)
target_link_libraries(main tensorflow)

if (WIN32)
  set(TF_LIBNAME tensorflow.dll)
endif (WIN32)

if (MSVC)
  set(TF_LIBNAME tensorflow.dll)
endif (MSVC)

if (UNIX)
  set(TF_LIBNAME libtensorflow.so)
endif (UNIX)

add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/versions/${TF_VERSION}/lib/${TF_LIBNAME}"
        $<TARGET_FILE_DIR:main>/libtensorflow.so)
