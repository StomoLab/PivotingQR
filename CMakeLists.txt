cmake_minimum_required(VERSION 3.28)
enable_testing()

project(Pivoting_QR)

set(CMAKE_CXX_STANDARD 20)

find_package(MKL CONFIG REQUIRED)

add_executable(Pivoting_QR main.cpp)

target_compile_options(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(Pivoting_QR PUBLIC $<LINK_ONLY:MKL::MKL>)

