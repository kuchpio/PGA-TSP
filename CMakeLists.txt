cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project("PGA-TSP" LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(pga-tsp main.cu)
target_include_directories(pga-tsp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(pga-tsp PROPERTIES CUDA_ARCHITECTURES "89")
target_compile_options(pga-tsp PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -lineInfo>)