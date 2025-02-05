#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// 检查CUDA错误的辅助函数
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

template<int blockSize>
__global__ void reduce_max(float *d_in, float *d_out) {
    __shared__ float smem[blockSize];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    
    // Load
    smem[tid] = d_in[gtid];
    __syncthreads();

    // Reduction in shared memory
    for(int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) {
            smem[tid] = max(smem[tid], smem[tid + index]);
        }
        __syncthreads();
    }

    // Store
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

// CPU版本用于验证结果
void reduce_max_cpu(float *x, float *out, int N) {
    out[0] = x[0];
    for (int i = 1; i < N; i++) {
        out[0] = max(out[0], x[i]);
    }
}

// 初始化数据为随机数
void initialize_data(float *x, int N) {
    srand(time(NULL));  // 设置随机数种子
    for (int i = 0; i < N; i++) {
        x[i] = (float)(rand() % 1000) / 100.0f;  // 生成0到10之间的随机浮点数
    }
}

// 验证结果
bool verify_results(float *gpu_result, float *cpu_result, int N) {
    if (fabs(cpu_result[0] - gpu_result[0]) > 1e-6) {
        printf("Result verification failed!\n");
        return false;
    }
    return true;
}

// 资源清理
void cleanup(float *d_in, float *d_out, float *h_in, float *h_out, float *h_cpu_out) {
    // 释放GPU内存
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    
    // 释放CPU内存
    if (h_in) free(h_in);
    if (h_out) free(h_out);
    if (h_cpu_out) free(h_cpu_out);
}

int main() {
    const int N = 25600000;
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int nbytes = N * sizeof(float);
    int out_nbytes = gridSize * sizeof(float);

    // 声明指针
    float *d_in = NULL, *d_out = NULL;  // GPU
    float *h_in = NULL, *h_out = NULL, *h_cpu_out = NULL;  // CPU

    // 分配GPU内存
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_in, nbytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_out, out_nbytes));

    // 分配CPU内存
    h_in = (float *)malloc(nbytes);
    h_out = (float *)malloc(out_nbytes);
    h_cpu_out = (float *)malloc(sizeof(float));
    
    if (!h_in || !h_out || !h_cpu_out) {
        printf("CPU Memory allocation failed!\n");
        cleanup(d_in, d_out, h_in, h_out, h_cpu_out);
        return -1;
    }

    // 初始化数据
    initialize_data(h_in, N);

    // 数据拷贝到GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice));

    // GPU计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    float gpu_time = 0.0f;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    reduce_max<blockSize><<<gridSize, blockSize>>>(d_in, d_out);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // 检查kernel是否执行成功
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 拷贝结果回CPU
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, out_nbytes, cudaMemcpyDeviceToHost));

    // CPU计算作为对照
    reduce_max_cpu(h_in, h_cpu_out, N);

    // 对GPU结果进行最终规约
    float final_result = h_out[0];
    for (int i = 1; i < gridSize; i++) {
        final_result = max(final_result, h_out[i]);
    }
    h_out[0] = final_result;

    // 验证结果
    if (verify_results(h_out, h_cpu_out, 1)) {
        printf("Results verified successfully!\n");
        printf("GPU Execution time: %f ms\n", gpu_time);
    }
    else {
        printf("Results verification failed!\n");
        printf("CPU result: %f\n", h_cpu_out[0]);
        printf("GPU result: %f\n", h_out[0]);
    }

    // 清理资源
    cleanup(d_in, d_out, h_in, h_out, h_cpu_out);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}