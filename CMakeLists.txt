cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project("PGA-TSP" LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("external")

find_package(MPI REQUIRED)

add_executable(pga-tsp main.cu)
target_link_libraries(pga-tsp MPI::MPI_CXX nlohmann_json::nlohmann_json)
target_include_directories(pga-tsp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} external/args/)
set_target_properties(pga-tsp PROPERTIES CUDA_ARCHITECTURES "80")
target_compile_options(pga-tsp PRIVATE -Wall
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>: -maxrregcount=64>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -lineInfo>)
