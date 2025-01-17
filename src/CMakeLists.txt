cmake_minimum_required(VERSION 3.28)

add_executable(Pivoting_QR main.cpp)

target_compile_options(Pivoting_QR PRIVATE -Wall -Wextra)

target_compile_options(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(Pivoting_QR PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(Pivoting_QR PUBLIC $<LINK_ONLY:MKL::MKL>)