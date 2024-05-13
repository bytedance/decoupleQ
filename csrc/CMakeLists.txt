# ===================== trtllm =========================
cmake_minimum_required(VERSION 3.21)
project(TRTLLM_KERNELS CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_PATH ${DECOUPLEQ_CUDA_HOME})
set(CUDNN_PATH ${DECOUPLEQ_CUDNN_HOME})
set(TORCH_PATH ${DECOUPLEQ_TORCH_HOME})
set(CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/cutlass)
set(TRTLLM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/TensorRT-LLM)
set(TRTLLM_KERNEL_PATH ${TRTLLM_PATH}/cpp/tensorrt_llm/kernels)

set(CMAKE_CUDA_ARCHITECTURES 80-real 89-real)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
if(Python3_FOUND)
  message(f"python3 found")
elseif()
  message(f"python3 not found")
endif()
if(Torch_FOUND)
  message(f"torch found")
elseif()
  message(f"torch not found")
endif()

find_library(TORCH_LIBS torch_python PATHS ${TORCH_PATH})
find_library(CUDA_LIBS cudart PATHS ${CUDA_PATH}/lib64)
find_library(CUDNN_LIBS cudnn PATHS ${CUDNN_PATH})

include_directories(${CUDA_PATH}/include)
include_directories(${CUDNN_PATH}/include)
include_directories(${CUTLASS_PATH}/include)
include_directories(${CUTLASS_PATH}/tools/util/include)
include_directories(${TRTLLM_PATH}/cpp/include)
include_directories(${TORCH_PATH}/include)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include)
include_directories(${Python3_INCLUDE_DIRS})

link_directories(${CUDA_PATH}/lib64)
link_directories(${CUDNN_PATH}/lib64)

add_definitions("-DENABLE_BF16")
add_definitions("-DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(TRTLLM_HEADERS
  ${TRTLLM_KERNEL_PATH}/../..
  ${TRTLLM_KERNEL_PATH}/cutlass_kernels/
  ${TRTLLM_KERNEL_PATH}/../cutlass_extensions/include
  )
include_directories(${TRTLLM_HEADERS})

set(TRTLLM_SRC_CPP
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/cutlass_heuristic.cpp
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/cutlass_preprocessors.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/logger.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/tllmException.cpp
    ${TRTLLM_KERNEL_PATH}/../../tensorrt_llm/common/stringUtils.cpp
   )

list(APPEND TRTLLM_SRC_CU
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/fpA_intB_gemm/fp16_int2_gemm_fg_scalebias.cu
    )
list(APPEND TRTLLM_SRC_CU
    ${TRTLLM_KERNEL_PATH}/cutlass_kernels/fpA_intB_gemm/bf16_int2_gemm_fg_scalebias.cu
    )
add_definitions("-DUSE_W2A16")

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

set(DECOUPLEQ_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/w2a16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_kernel_file_1.generated.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_kernel_file_2.generated.cu
    )

add_library(decoupleQ_kernels SHARED ${TRTLLM_SRC_CPP} ${TRTLLM_SRC_CU} ${DECOUPLEQ_SRC_CU})
target_link_libraries(decoupleQ_kernels ${TORCH_LIBS} ${CUDA_LIBS} ${CUDNN_LIBS})
set_property(TARGET decoupleQ_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET decoupleQ_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
