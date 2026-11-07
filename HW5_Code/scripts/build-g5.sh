#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-12.8/bin:$PATH
make train_gpt2cu PRECISION=BF16 USE_CUDNN=1 GPU_COMPUTE_CAPABILITY=86 \
  NO_MULTI_GPU=1 NO_USE_MPI=1 \
  NVCC_INCLUDES="-I$HOME/cudnn-frontend/include -I/opt/hw5/cudnn/include" \
  NVCC_LDFLAGS="-lcublas -lcublasLt -lnvidia-ml -L/opt/hw5/cudnn/lib -lcudnn -Xlinker -rpath -Xlinker /opt/hw5/cudnn/lib"
