set -x

cd csrc
mkdir -p build
cd build

export TORCH_CUDA_ARCH_LIST="8.0+PTX" 
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.9/dist-packages/torch \
    -DDECOUPLEQ_TORCH_HOME=/usr/local/lib/python3.9/dist-packages/torch \
    -DCMAKE_BUILD_TYPE=Release \
    -DDECOUPLEQ_CUDA_HOME=/usr/local/cuda  \
    -DDECOUPLEQ_CUDNN_HOME=/usr/local/cuda ..
make -j 
cp libdecoupleQ_kernels.so ../../decoupleQ/decoupleQ_kernels.so

cd ../../



