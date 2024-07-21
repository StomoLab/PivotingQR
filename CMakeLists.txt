cmake_minimum_required(VERSION 3.28)
project(Pivoting_QR)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH "/opt/OpenBLAS")

find_library(OpenBLAS_LIBRARY
        NAMES openblas_seq
        PATHS /opt/OpenBLAS
        NO_SYSTEM_ENVIRONMENT_PATH)

add_executable(Pivoting_QR main.cpp)

target_link_libraries(Pivoting_QR "${OpenBLAS_LIBRARY}")

include_directories("/opt/OpenBLAS/include")



