#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void computeKernel(double* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double temp = data[tid];
        temp = temp * temp + 1.0;
        data[tid] = temp;
    }
}

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    double duration = 10.0;
    if (argc >= 2) duration = atof(argv[1]);

    int N = 1 << 24;  // 16M doubles
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemset(d_data, 0, N * sizeof(double));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    computeKernel<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();

    long count = 0;
    double start = now_sec();
    while (now_sec() - start < duration) {
        computeKernel<<<blocks, threads>>>(d_data, N);
        cudaDeviceSynchronize();
        count++;
    }

    printf("%ld\n", count);
    cudaFree(d_data);
    return 0;
}
