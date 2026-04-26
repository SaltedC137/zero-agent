cmake_minimum_required(VERSION 3.14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

add_compile_options(-O3 -march=native -ffast-math)

# for cland complete
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ZATO VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

find_package(llama REQUIRED)
find_package(httplib REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(${PROJECT_NAME} src/gym.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE llama)
target_link_libraries(${PROJECT_NAME} PRIVATE httplib::httplib)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)


target_compile_options(${PROJECT_NAME} PRIVATE -O3)
