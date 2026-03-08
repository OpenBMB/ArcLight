cmake_minimum_required(VERSION 3.15)
project("ArcLight" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

option(NEON "Enable NEON backend" ON)

if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

if(NEON)
    message(STATUS "NEON enabled")
    # add_compile_definitions(__ARM_NEON)
endif()
# add_subdirectory(nnml)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

add_subdirectory(nnml)

file(GLOB NNML_SRC
    *.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp
)

add_library(arclight SHARED ${NNML_SRC})
set_target_properties(arclight PROPERTIES
    OUTPUT_NAME al
)
target_include_directories(arclight PUBLIC include models)
target_link_libraries(arclight PUBLIC nnml)


if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ipython/CMakeLists.txt")
    add_subdirectory(ipython)
endif()

# add_executable(test-gguf ${CMAKE_CURRENT_SOURCE_DIR}/test/test-gguf.cpp)
# target_link_libraries(test-gguf PRIVATE arclight)

# add_executable(test-model ${CMAKE_CURRENT_SOURCE_DIR}/test/test-model.cpp)
# target_link_libraries(test-model PRIVATE arclight)

# add_executable(test-tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/test/test-tokenizer.cpp)
# target_link_libraries(test-tokenizer PRIVATE arclight)

# add_executable(test-onetoken ${CMAKE_CURRENT_SOURCE_DIR}/test/test-onetoken.cpp)
# target_link_libraries(test-onetoken PRIVATE arclight)

# add_executable(test-onetoken-tp ${CMAKE_CURRENT_SOURCE_DIR}/test/test-onetoken-tp.cpp)
# target_link_libraries(test-onetoken-tp PRIVATE arclight)

add_executable(al-gen ${CMAKE_CURRENT_SOURCE_DIR}/apps/al-gen.cpp)
target_link_libraries(al-gen PRIVATE arclight)

add_executable(al-chat ${CMAKE_CURRENT_SOURCE_DIR}/apps/al-chat.cpp)
target_link_libraries(al-chat PRIVATE arclight)

add_executable(al-ppl ${CMAKE_CURRENT_SOURCE_DIR}/apps/al-ppl.cpp)
target_link_libraries(al-ppl PRIVATE arclight)

# enable_testing()

# file(GLOB TEST_SOURCES test/*.cpp)
# foreach(TEST_SRC ${TEST_SOURCES})
#     get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
#     add_executable(${TEST_NAME} ${TEST_SRC})
#     target_link_libraries(${TEST_NAME} PRIVATE arclight)
#     add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
# endforeach()

message(STATUS "ArcLight build configured successfully.")
