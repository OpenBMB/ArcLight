cmake_minimum_required(VERSION 3.15)
project("nnml" C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC_FILES
    src/*.c
    src/*.cpp
    src/ops/*.cpp
    # src/ops/arm/*.cpp
)

if(NEON)
    add_compile_definitions(__ARM_NEON)
    add_compile_definitions(__ARM_FEATURE_FMA)
    add_compile_definitions(__ARM_FEATURE_DOTPROD)
    add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    add_compile_options(-march=armv8.2-a+dotprod+fp16)
    add_compile_options(-ffp-contract=fast)
    # add_compile_options(-mcpu=tsv110+dotprod+fp16)
    # add_compile_options(-fno-fast-math -fno-associative-math -frounding-math)
    file(GLOB NEON_SRC_FILES src/ops/arm/*.cpp)
    list(APPEND SRC_FILES ${NEON_SRC_FILES})
endif()

add_library(nnml SHARED ${SRC_FILES})
target_include_directories(nnml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


option(NNML_USE_NUMA "Enable NUMA memory allocation" ON)
if (NNML_USE_NUMA)
    find_library(LIBNUMA numa)
    if (LIBNUMA)
        message(STATUS "NUMA support enabled: ${LIBNUMA}")
        target_link_libraries(nnml PUBLIC ${LIBNUMA})
        target_compile_definitions(nnml PUBLIC NNML_HAS_NUMA=1)
    else()
        message(WARNING "libnuma not found — fallback to UMA.")
        target_compile_definitions(nnml PUBLIC NNML_NO_NUMA=1)
    endif()
else()
    message(STATUS "NUMA support disabled by user.")
    target_compile_definitions(nnml PUBLIC NNML_NO_NUMA=1)
endif()

option(NNML_USE_OPENMP "Enable OpenMP parallel support in NNML" OFF)
message(STATUS "NNML_USE_OPENMP = ${NNML_USE_OPENMP}")
if(NNML_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(backend PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(backend PUBLIC NNML_USE_OPENMP)
endif()


# add_executable(test-memory ${CMAKE_CURRENT_SOURCE_DIR}/test/test-memory.cpp)
# target_link_libraries(test-memory PRIVATE nnml)

# add_executable(test-thread ${CMAKE_CURRENT_SOURCE_DIR}/test/test-thread.cpp)
# target_link_libraries(test-thread PRIVATE nnml)

# add_executable(test-tensor ${CMAKE_CURRENT_SOURCE_DIR}/test/test-tensor.cpp)
# target_link_libraries(test-tensor PRIVATE nnml)

add_executable(test-cgraph ${CMAKE_CURRENT_SOURCE_DIR}/test/test-cgraph.cpp)
target_link_libraries(test-cgraph PRIVATE nnml)

add_executable(test-gemm ${CMAKE_CURRENT_SOURCE_DIR}/test/test-gemm.cpp)
target_link_libraries(test-gemm PRIVATE nnml)
