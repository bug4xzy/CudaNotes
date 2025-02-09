#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <limits>

// CUDA错误检查宏（保持不变）
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// 假设输入数组中有大量数据
// 使用多个块，每个块128个线程(blockSize=128)
#define WarpSize 32

template<typename T, int blockSize>
__device__ T WarpShuffle(T sum) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

template<typename T, int blockSize>
__global__ void reduce_warplevel(T *d_in, T *d_out, int N) {
    T sum = 0;
    
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    int total_threads = blockSize * gridDim.x;
    
    // 优化1：每个线程一次处理4个连续元素
    #pragma unroll 4
    for (int i = gtid; i < N/4; i += total_threads) {
        if (i*4 + 3 < N) {
            T v1 = d_in[i*4];
            T v2 = d_in[i*4 + 1];
            T v3 = d_in[i*4 + 2];
            T v4 = d_in[i*4 + 3];
            sum += v1 + v2 + v3 + v4;
        }
    }
    
    // 处理剩余元素
    for (int i = gtid*4 + (N/4)*4; i < N; i += total_threads) {
        sum += d_in[i];
    }
    
    __shared__ T warpSums[blockSize / WarpSize];
    const int laneId = tid % WarpSize;
    const int warpId = tid / WarpSize;
    
    // warp内规约
    sum = WarpShuffle<T, blockSize>(sum);
    
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // 最终规约
    if (tid < blockSize/WarpSize) {
        sum = warpSums[tid];
        if (warpId == 0) {
            sum = WarpShuffle<T, blockSize/WarpSize>(sum);
        }
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}


// 模板化的CPU规约函数
template<typename T>
void reduce_sum_cpu_kahan(T *x, T *out, int N) {
    T sum = T(0);
    T c = T(0);
    
    for (int i = 0; i < N; i++) {
        T y = x[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    out[0] = sum;
}

// 模板化的最终规约函数
template<typename T>
T final_reduction_kahan(T *partial_sums, int size) {
    T sum = T(0);
    T c = T(0);
    
    for (int i = 0; i < size; i++) {
        T y = partial_sums[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

// 模板化的数据初始化函数
template<typename T>
void initialize_data(T *x, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<T>(rand() % 1000) / static_cast<T>(100);
    }
}

// 模板化的结果验证函数
template<typename T>
bool verify_results(T *gpu_result, T *cpu_result, int N) {
    T relative_error = std::abs(cpu_result[0] - gpu_result[0]) / cpu_result[0];
    if (relative_error > std::numeric_limits<T>::epsilon() * 100) {
        printf("Result verification failed!\n");
        printf("Relative Error: %e\n", static_cast<double>(relative_error));
        return false;
    }
    return true;
}

// 模板化的资源清理函数
template<typename T>
void cleanup(T *d_in, T *d_out, T *h_in, T *h_out, T *h_cpu_out) {
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (h_in) free(h_in);
    if (h_out) free(h_out);
    if (h_cpu_out) free(h_cpu_out);
}

// 模板化的主计算函数
template<typename T>
void run_reduction(int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize / 4 ;
    size_t nbytes = N * sizeof(T);
    size_t out_nbytes = gridSize * sizeof(T);

    // 声明指针
    T *d_in = NULL, *d_out = NULL;
    T *h_in = NULL, *h_out = NULL, *h_cpu_out = NULL;

    // 分配内存
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_in, nbytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_out, out_nbytes));

    h_in = (T *)malloc(nbytes);
    h_out = (T *)malloc(out_nbytes);
    h_cpu_out = (T *)malloc(sizeof(T));
    
    if (!h_in || !h_out || !h_cpu_out) {
        printf("CPU Memory allocation failed!\n");
        cleanup(d_in, d_out, h_in, h_out, h_cpu_out);
        return;
    }

    // 初始化数据
    initialize_data(h_in, N);

    // 复制数据到GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice));

    // GPU计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    float gpu_time = 0.0f;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // 调用kernel
    reduce_warplevel<T, blockSize><<<gridSize, blockSize>>>(d_in, d_out, N);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // 检查kernel执行
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 复制结果回CPU
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, out_nbytes, cudaMemcpyDeviceToHost));

    // CPU计算
    reduce_sum_cpu_kahan(h_in, h_cpu_out, N);

    // 最终规约
    T final_result = final_reduction_kahan(h_out, gridSize);
    h_out[0] = final_result;

    // 验证结果
    if (verify_results(h_out, h_cpu_out, 1)) {
        printf("Results verified successfully!\n");
        printf("GPU Execution time: %f ms\n", gpu_time);
        printf("Final sum: %e\n", static_cast<double>(h_out[0]));
    }
    else {
        printf("Results verification failed!\n");
        printf("CPU result: %e\n", static_cast<double>(h_cpu_out[0]));
        printf("GPU result: %e\n", static_cast<double>(h_out[0]));
    }

    // 清理资源
    cleanup(d_in, d_out, h_in, h_out, h_cpu_out);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

int main() {
    const int N = 25600000;
    
    printf("\nRunning with float:\n");
    run_reduction<float>(N);
    
    printf("\nRunning with double:\n");
    run_reduction<double>(N);
    
    printf("\nRunning with int:\n");
    run_reduction<int>(N);
    
    return 0;
}