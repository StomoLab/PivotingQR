cmake_minimum_required(VERSION 3.28)

add_executable(QR_test test.cpp)
target_compile_options(QR_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(QR_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(QR_test PUBLIC $<LINK_ONLY:MKL::MKL>)
target_link_libraries(QR_test PUBLIC gtest_main)

include(FetchContent)
FetchContent_Declare(
        "googletest"
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        GIT_TAG "v1.13.0"
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_test(NAME test COMMAND QR_test)
enable_testing()
