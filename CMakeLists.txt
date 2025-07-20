cmake_minimum_required(VERSION 3.5)

project(Simple_matrix_operations
        VERSION 0.5.0
        LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "C:/Users/Dell/.vcpkg-clion/vcpkg/installed/x64-mingw-dynamic")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(TESTS "Enable tests" ON)

find_package(Matplot++ CONFIG REQUIRED)

add_subdirectory(../Simple-matrix-operations/Libs/Matrix ${CMAKE_BINARY_DIR}/Matrix-build)
add_subdirectory(../Simple-matrix-operations/Libs/Matrix_tools ${CMAKE_BINARY_DIR}/Matrix_tools-build)

message(STATUS "Configuring Percepton library")
add_subdirectory(Libs/Percepton)
message(STATUS "Percepton library configured")


if(TESTS)
    find_package(Catch2 3 REQUIRED CONFIG)
    enable_testing()
    add_subdirectory(Libs/Percepton/Tests)
    add_subdirectory(../Simple-matrix-operations/Libs/Matrix/Tests ${CMAKE_BINARY_DIR}/Matrix-tests-build)
endif()

add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_20)
target_link_libraries(my_app PRIVATE Matrix)
target_link_libraries(my_app PRIVATE Percepton)

