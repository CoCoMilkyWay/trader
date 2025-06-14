# sudo apt install pybind11-dev
# pip install pybind11

# set(PYTHON_VERSION 3.13)

cmake_minimum_required(VERSION 3.16)
project(Pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# set(Python3_ROOT_DIR "C:/Users/chuyin.wang/.pyenv/pyenv-win/versions/3.12.9/")
find_package(Python3 ${} REQUIRED COMPONENTS Interpreter Development)

# Collect sources
set(SRC_MAIN src/main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -static-libgcc -static-libstdc++")

# Create target
add_library(Pipeline SHARED 
${SRC_MAIN}
# ${SRC_INTERFACE}
# ${SRC_SAMPLE}
)
# add_executable()

# Include directories for Python and Boost
target_include_directories(Pipeline PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# Link Boost.Interprocess and Python libs
target_link_libraries(Pipeline PRIVATE
    ${Python3_LIBRARIES}
    DbgHelp
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set_target_properties(Pipeline PROPERTIES PREFIX "")

# Enable full optimizations in Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling full optimization flags for best performance")
    
    # Enable LTO (portable CMake flag)
    set_target_properties(Pipeline PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    
    # Set aggressive compiler flags manually
    target_compile_options(Pipeline PRIVATE
        -static-libgcc      # static library
        -static-libstdc++   # static library
        -O3
        -march=native
        -ffast-math
        -funroll-loops
    )

    target_link_options(Pipeline PRIVATE
        -O3
        -march=native
        -flto
    )
endif()
