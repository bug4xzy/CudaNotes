#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

typedef float FLOAT;

// 检查CUDA错误的辅助函数
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 2D grid */
    // int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    /* 1D grid */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++) z[i] = y[i] + x[i];
}

// 初始化数据为随机数
void initialize_data(FLOAT *x, FLOAT *y, int N) {
    srand(time(NULL));  // 设置随机数种子
    for (int i = 0; i < N; i++) {
        x[i] = (FLOAT)(rand() % 1000) / 100.0f;  // 生成0到10之间的随机浮点数
        y[i] = (FLOAT)(rand() % 1000) / 100.0f;
    }
}

// 验证结果
bool verify_results(FLOAT *gpu_result, FLOAT *cpu_result, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-6) {
            printf("Result verification failed at element %d!\n", i);
            return false;
        }
    }
    return true;
}

// 资源清理
void cleanup(FLOAT *dx, FLOAT *dy, FLOAT *dz, FLOAT *hx, FLOAT *hy, FLOAT *hz, FLOAT *hz_cpu) {
    // 释放GPU内存
    if (dx) cudaFree(dx);
    if (dy) cudaFree(dy);
    if (dz) cudaFree(dz);
    
    // 释放CPU内存
    if (hx) free(hx);
    if (hy) free(hy);
    if (hz) free(hz);
    if (hz_cpu) free(hz_cpu);
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);
    int bs = 256;  // block size

    int s = ceil((N + bs - 1.) / bs);
    dim3 grid(s);

    // 声明指针
    FLOAT *dx = NULL, *dy = NULL, *dz = NULL;  // GPU
    FLOAT *hx = NULL, *hy = NULL, *hz = NULL, *hz_cpu_res = NULL;  // CPU

    // 分配GPU内存
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dx, nbytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dy, nbytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dz, nbytes));

    // 分配CPU内存
    hx = (FLOAT *)malloc(nbytes);
    hy = (FLOAT *)malloc(nbytes);
    hz = (FLOAT *)malloc(nbytes);
    hz_cpu_res = (FLOAT *)malloc(nbytes);
    
    if (!hx || !hy || !hz || !hz_cpu_res) {
        printf("CPU Memory allocation failed!\n");
        cleanup(dx, dy, dz, hx, hy, hz, hz_cpu_res);
        return -1;
    }

    // 初始化数据
    initialize_data(hx, hy, N);

    // 数据拷贝到GPU
    CHECK_CUDA_ERROR(cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice));

    // GPU计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    float gpu_time = 0.0f;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vec_add<<<grid, bs>>>(dx, dy, dz, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // 检查kernel是否执行成功
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 拷贝结果回CPU
    CHECK_CUDA_ERROR(cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost));

    // CPU计算作为对照
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    // 验证结果
    if (verify_results(hz, hz_cpu_res, N)) {
        printf("Results verified successfully!\n");
        printf("GPU Execution time: %f ms\n", gpu_time);
    }
    else {
        printf("Results verification failed!\n");
        printf("cpu result: %f\n", hz_cpu_res[0]);
        printf("gpu result: %f\n", hz[0]);
    }

    // 清理资源
    cleanup(dx, dy, dz, hx, hy, hz, hz_cpu_res);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}