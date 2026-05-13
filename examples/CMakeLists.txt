project(BlockFormer LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # clangd

find_package(OpenMP REQUIRED)
find_package(HEaaN REQUIRED)
# find_package(CUDA REQUIRED)

add_executable(pcmm pcmm.cpp)
target_link_libraries(pcmm PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX)
# target_link_libraries(pcmm PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})

add_executable(ccmm ccmm.cpp)
target_link_libraries(ccmm PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX)
# target_link_libraries(ccmm PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})

add_executable(nonlinear nonlinear.cpp)
target_link_libraries(nonlinear PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX)
# target_link_libraries(ex17_bootstrapping PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
