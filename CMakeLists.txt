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

option(ZATO_USE_CUDA_BACKEND "Link against llama build with CUDA backend" OFF)
set(ZATO_LLAMA_ROOT "" CACHE PATH "Root path of llama install (e.g. /opt/llama-cuda)")

if(ZATO_LLAMA_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${ZATO_LLAMA_ROOT}")
  list(PREPEND CMAKE_LIBRARY_PATH "${ZATO_LLAMA_ROOT}/lib")
endif()

find_package(llama REQUIRED)
find_package(httplib REQUIRED)
find_package(nlohmann_json REQUIRED)

# cuda
if(ZATO_USE_CUDA_BACKEND)
  find_library(GGML_CUDA_LIB
    NAMES ggml-cuda libggml-cuda
    PATHS ${CMAKE_LIBRARY_PATH} /usr/lib64 /usr/lib /lib64 /lib
  )

  if(NOT GGML_CUDA_LIB)
    message(FATAL_ERROR
      "ZATO_USE_CUDA_BACKEND=ON but ggml-cuda was not found. "
      "Build/install CUDA llama first, then set -DZATO_LLAMA_ROOT=/path/to/llama-cuda.")
  endif()

  message(STATUS "CUDA backend enabled: ${GGML_CUDA_LIB}")
else()
  message(STATUS "CUDA backend disabled (ZATO_USE_CUDA_BACKEND=OFF).")
endif()

add_executable(${PROJECT_NAME}
  src/gym.cpp
  src/model.cpp
  src/agent.cpp
  src/tool.cpp
  src/api_model.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE llama)
target_link_libraries(${PROJECT_NAME} PRIVATE httplib::httplib)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(${PROJECT_NAME} PRIVATE -O3)
