/*
 * Workload that reproduces the burst-debit bug in the time-based throttle.
 *
 * Submits KERNELS_PER_SET kernel launches before synchronising, creating a
 * large GPU burst.  For low limits (≤ ~30%) this burst exceeds one watcher-tick
 * grant, driving the token bucket deeply negative and causing the pod to stall
 * for multiple ticks — i.e. more throttling than configured.
 *
 * Each kernel reads and writes 128 MB of doubles (memory-bandwidth-bound).
 * Burst duration is ~64ms on an A100 and ~215ms on a T4 (500 kernels).
 *
 * Usage: test_throttle_gemm [duration_seconds]
 * Prints: number of sets completed to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define KERNELS_PER_SET 500

__global__ void burstKernel(double* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double v = data[tid];
        v = v * v + 1.0;
        data[tid] = v;
    }
}

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    double duration = 30.0;
    if (argc >= 2) duration = atof(argv[1]);

    int N = 1 << 24;  // 16M doubles = 128 MB
    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemset(d_data, 0, N * sizeof(double));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    for (int i = 0; i < 4; i++) {
        burstKernel<<<blocks, threads>>>(d_data, N);
    }
    cudaDeviceSynchronize();

    long count = 0;
    double start = now_sec();
    while (now_sec() - start < duration) {
        for (int i = 0; i < KERNELS_PER_SET; i++) {
            burstKernel<<<blocks, threads>>>(d_data, N);
        }
        cudaDeviceSynchronize();
        count++;
    }

    printf("%ld\n", count);
    cudaFree(d_data);
    return 0;
}
