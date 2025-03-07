#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <limits>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// GPU Kernel
template<typename T, int blockSize>
__global__ void reduce_base(T *d_in, T *d_out, int N) {
    __shared__ T sdata[blockSize];
    int tid = threadIdx.x;
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 使用Kahan求和算法的局部累加
    T sum = T(0);
    T c = T(0);
    
    // 每个线程处理多个元素
    for(int i = globalIdx; i < N; i += blockDim.x * gridDim.x) {
        T y = d_in[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    // 加载局部和到共享内存
    sdata[tid] = sum;
    __syncthreads();
    
    // 在共享内存中进行规约
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            // 对共享内存中的数据进行Kahan求和
            T y = sdata[tid + s] - c;
            T t = sdata[tid] + y;
            c = (t - sdata[tid]) - y;
            sdata[tid] = t;
        }
        __syncthreads();
    }
    
    // 将每个block的结果写入全局内存
    if(tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// CPU规约函数
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

// 最终规约函数
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

// 数据初始化函数
template<typename T>
void initialize_data(T *x, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<T>(rand() % 1000) / static_cast<T>(100);
    }
}

// 结果验证函数
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

// 资源清理函数
template<typename T>
void cleanup(T *d_in, T *d_out, T *h_in, T *h_out, T *h_cpu_out) {
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (h_in) free(h_in);
    if (h_out) free(h_out);
    if (h_cpu_out) free(h_cpu_out);
}

// 主计算函数
template<typename T>
void run_reduction(int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
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

    // 复制数据到GPU并初始化输出数组
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_out, 0, out_nbytes));

    // GPU计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    float gpu_time = 0.0f;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // 调用kernel
    reduce_base<T, blockSize><<<gridSize, blockSize>>>(d_in, d_out, N);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // 检查kernel执行
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 复制结果回CPU
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, out_nbytes, cudaMemcpyDeviceToHost));

    // CPU计算参考结果
    reduce_sum_cpu_kahan(h_in, h_cpu_out, N);

    // 使用Kahan求和算法对block结果进行最终规约
    T final_sum = T(0);
    T c = T(0);
    for(int i = 0; i < gridSize; i++) {
        T y = h_out[i] - c;
        T t = final_sum + y;
        c = (t - final_sum) - y;
        final_sum = t;
    }
    h_out[0] = final_sum;

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