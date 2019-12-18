cmake_minimum_required(VERSION 3.10)
project(wildcardHElib)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include_directories(/home/benjamin/Code/C++/wildcardHElib/oldlib/include) # point to where NTL and HElib headers are
# link_directories(/home/benjamin/Code/C++/wildcardHElib/oldlib/lib) # point to where libntl.a/libntl.so and fhe.a/libhelib.so are 

include_directories(/home/benjamin/Code/C++/wildcardHElib/newlib/include) # point to where NTL and HElib headers are
link_directories(/home/benjamin/Code/C++/wildcardHElib/newlib/lib) # point to where libntl.a/libntl.so and fhe.a/libhelib.so are 

add_executable(wildcardHElib privateWildcard.cpp)
target_link_libraries(wildcardHElib -lhelib -lntl -lgmp -pthread)