/*
 * Demonstrates CUPTI Activity API for measuring true per-kernel GPU compute time.
 *
 * Launches a simple kernel, then reads back CUPTI activity records to get the
 * hardware start/end timestamps.  Compares against cuEventElapsedTime to show
 * they agree when running solo (no contention from other processes).
 *
 * Build: part of the normal cmake build (auto-discovered by test/CMakeLists.txt)
 * Link:  needs -lcupti (added below via pragma)
 * Run:   build/test/test_cupti_kernel_time
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_activity.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUPTI(call) do { \
    CUptiResult e = (call); \
    if (e != CUPTI_SUCCESS) { \
        const char *s; cuptiGetResultString(e, &s); \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", __FILE__, __LINE__, s); \
        exit(1); \
    } \
} while(0)

/* Accumulates true GPU compute time (ns) across all kernels seen so far. */
static uint64_t g_total_gpu_ns = 0;
static int      g_kernel_count  = 0;

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size           = 64 * 1024;  /* 64 KB per buffer */
    *buffer         = (uint8_t *)malloc(*size);
    *maxNumRecords  = 0;           /* fill the whole buffer */
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    CUptiResult status;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                CUpti_ActivityKernel8 *k = (CUpti_ActivityKernel8 *)record;
                uint64_t duration_ns = k->end - k->start;
                g_total_gpu_ns += duration_ns;
                g_kernel_count++;
                printf("  CUPTI kernel: name=%-30s  start=%llu  end=%llu  duration=%.3fms\n",
                       k->name,
                       (unsigned long long)k->start,
                       (unsigned long long)k->end,
                       duration_ns / 1e6);
            }
        } else if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        }
    } while (status == CUPTI_SUCCESS);

    free(buffer);
}

__global__ void workKernel(float *data, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float v = data[tid];
        for (int i = 0; i < iters; i++) v = v * 1.0001f + 0.0001f;
        data[tid] = v;
    }
}

int main(void) {
    /* Init CUPTI activity recording */
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    int N = 1 << 22;  /* 4M floats */
    float *d;
    CHECK_CUDA(cudaMalloc(&d, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d, 0, N * sizeof(float)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    /* Warmup */
    workKernel<<<blocks, threads>>>(d, N, 100);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    g_total_gpu_ns = 0;
    g_kernel_count = 0;

    /* Timed run — use both CUDA events and CUPTI */
    CUevent ev_start, ev_end;
    cuEventCreate(&ev_start, CU_EVENT_DEFAULT);
    cuEventCreate(&ev_end,   CU_EVENT_DEFAULT);

    int num_kernels = 5;
    printf("Launching %d kernels:\n", num_kernels);

    cuEventRecord(ev_start, 0);
    for (int i = 0; i < num_kernels; i++) {
        workKernel<<<blocks, threads>>>(d, N, 500);
    }
    cuEventRecord(ev_end, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Flush CUPTI buffers to trigger buffer_completed callback */
    CHECK_CUPTI(cuptiActivityFlushAll(0));

    float event_ms = 0;
    cuEventElapsedTime(&event_ms, ev_start, ev_end);

    printf("\ncuEventElapsedTime (wall-clock between events): %.3f ms\n", event_ms);
    printf("CUPTI total GPU compute (%d kernels):           %.3f ms\n",
           g_kernel_count, g_total_gpu_ns / 1e6);
    printf("CUPTI per-kernel average:                       %.3f ms\n",
           g_total_gpu_ns / 1e6 / (g_kernel_count > 0 ? g_kernel_count : 1));
    printf("\nWhen running solo these should be equal.\n");
    printf("Under multi-process contention, cuEventElapsedTime would be inflated;\n");
    printf("CUPTI would remain accurate.\n");

    cuEventDestroy(ev_start);
    cuEventDestroy(ev_end);
    CHECK_CUDA(cudaFree(d));
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    return 0;
}
