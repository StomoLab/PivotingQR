cmake_minimum_required(VERSION 3.28)
project(Pivoting_QR VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(CTest)

find_package(MKL CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(test)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()



#add_executable(Pivoting_QR src/main.cpp)
#
#target_compile_options(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
#target_include_directories(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#target_link_libraries(Pivoting_QR PUBLIC $<LINK_ONLY:MKL::MKL>)

