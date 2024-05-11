#include "scale.cuh"

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, ggml_nelements(src0), stream);
    CUDA_CHECK(cudaGetLastError());
}

// ggml_map_custom1
struct ggml_map_custom1_op_params {
    ggml_custom1_op_t fun;
    int n_tasks;
    void * userdata;
};

void ggml_cuda_op_map_custom1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    
    const struct ggml_tensor * a = dst->src[0];

    struct ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    size_t size = a->nb[0] * a->ne[0] * a->ne[1] * a->ne[2] * a->ne[3];
    char * data = new char[size];

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)a->data, size, cudaMemcpyDeviceToHost, ctx.stream()));

    p.fun(dst, a, 0, 0, data);
    delete data;

    CUDA_CHECK(cudaGetLastError());
}