cmake_minimum_required(VERSION 3.12)

project(oetsort LANGUAGES CUDA CXX)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

set(SOURCE_FILES
    oetsort_cuda.cu
    )

add_executable(oetsort_cuda ${SOURCE_FILES})

message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(oetsort_cuda PRIVATE caliper)