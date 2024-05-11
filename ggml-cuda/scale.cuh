#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_map_custom1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);