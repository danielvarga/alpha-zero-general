cmake_minimum_required(VERSION 3.5)
project(AMOBA)

set(CMAKE_CXX_STANDARD 11)

add_custom_target(
    run ALL COMMAND ${CMAKE_COMMAND} -E copy ../model.pb model.pb
)
include_directories(${CMAKE_SOURCE_DIR}/tflib/include)
link_directories(${CMAKE_SOURCE_DIR}/tflib/lib)
add_executable(AMOBA src/main.cpp cppflow/src/Model.cpp cppflow/src/Tensor.cpp)
target_include_directories(AMOBA PRIVATE cppflow/include)
target_link_libraries (AMOBA -ltensorflow)
