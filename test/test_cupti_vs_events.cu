/*
 * Proves whether cuEventElapsedTime is inflated in the multi-kernel batch scenario
 * and whether CUPTI avoids that inflation.
 *
 * Three modes:
 *   solo_events   — 1 kernel per sync, events measurement (baseline, no contention)
 *   batch_events  — N kernels before sync, per-kernel start/end events (may inflate)
 *   batch_cupti   — N kernels before sync, CUPTI measures each kernel (should not inflate)
 *
 * Run 4 concurrent processes. If cuEventElapsedTime is inflated in the batch case,
 * batch_events per-kernel time will be >> solo_events per-kernel time.
 * If CUPTI is accurate, batch_cupti per-kernel time will match solo_events.
 *
 * Usage: test_cupti_vs_events <solo_events|batch_events|batch_cupti> <duration_s> [kernels_per_batch]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
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

static uint64_t g_cupti_total_ns  = 0;
static int64_t  g_cupti_count     = 0;

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size          = 512 * 1024;
    *buffer        = (uint8_t *)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel8 *k = (CUpti_ActivityKernel8 *)record;
            g_cupti_total_ns += k->end - k->start;
            g_cupti_count++;
        }
    }
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

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <solo_events|batch_events|batch_cupti> <duration_s> [kernels_per_batch]\n", argv[0]);
        return 1;
    }
    const char *mode     = argv[1];
    double      duration = atof(argv[2]);
    int         batch    = (argc >= 4) ? atoi(argv[3]) : 500;

    int solo   = strcmp(mode, "solo_events")  == 0;
    int bevts  = strcmp(mode, "batch_events") == 0;
    int bcupti = strcmp(mode, "batch_cupti")  == 0;

    if (!solo && !bevts && !bcupti) {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    if (bcupti) {
        CHECK_CUPTI(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
        CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    int N = 1 << 22;
    float *d;
    CHECK_CUDA(cudaMalloc(&d, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d, 0, N * sizeof(float)));
    int threads = 256, blocks = (N + threads - 1) / threads;

    /* Warmup */
    workKernel<<<blocks, threads>>>(d, N, 100);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (bcupti) { cuptiActivityFlushAll(0); g_cupti_total_ns = 0; g_cupti_count = 0; }

    double   total_events_ms = 0.0;
    int64_t  total_kernels   = 0;
    int      sets            = 0;

    /* Per-kernel event storage for batch_events mode */
    CUevent *ev_starts = NULL, *ev_ends = NULL;
    if (bevts) {
        ev_starts = (CUevent *)malloc(batch * sizeof(CUevent));
        ev_ends   = (CUevent *)malloc(batch * sizeof(CUevent));
        for (int i = 0; i < batch; i++) {
            cuEventCreate(&ev_starts[i], CU_EVENT_DEFAULT);
            cuEventCreate(&ev_ends[i],   CU_EVENT_DEFAULT);
        }
    }

    double start_wall = now_sec();
    while (now_sec() - start_wall < duration) {
        if (solo) {
            /* Baseline: 1 kernel per sync, events around each kernel */
            CUevent ev_s, ev_e;
            cuEventCreate(&ev_s, CU_EVENT_DEFAULT);
            cuEventCreate(&ev_e, CU_EVENT_DEFAULT);
            cuEventRecord(ev_s, 0);
            workKernel<<<blocks, threads>>>(d, N, 500);
            cuEventRecord(ev_e, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            float ms = 0;
            cuEventElapsedTime(&ms, ev_s, ev_e);
            total_events_ms += ms;
            total_kernels++;
            cuEventDestroy(ev_s);
            cuEventDestroy(ev_e);

        } else if (bevts) {
            /* Batch: N kernels before sync, per-kernel events recorded inline */
            for (int i = 0; i < batch; i++) {
                cuEventRecord(ev_starts[i], 0);
                workKernel<<<blocks, threads>>>(d, N, 500);
                cuEventRecord(ev_ends[i], 0);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            for (int i = 0; i < batch; i++) {
                float ms = 0;
                cuEventElapsedTime(&ms, ev_starts[i], ev_ends[i]);
                total_events_ms += ms;
                total_kernels++;
            }

        } else { /* batch_cupti */
            /* Batch: N kernels before sync, CUPTI measures execution time */
            for (int i = 0; i < batch; i++) {
                workKernel<<<blocks, threads>>>(d, N, 500);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            cuptiActivityFlushAll(0);
            total_kernels += batch;
        }
        sets++;
    }
    double wall_s = now_sec() - start_wall;

    if (bcupti) {
        total_events_ms = g_cupti_total_ns / 1e6;
        total_kernels   = g_cupti_count;
    }

    double per_kernel_ms = total_kernels > 0 ? total_events_ms / total_kernels : 0;
    double wall_ms       = wall_s * 1000.0;

    printf("mode=%-15s  sets=%d  kernels=%ld  wall=%.0fms  "
           "measured_total=%.1fms  per_kernel=%.4fms\n",
           mode, sets, (long)total_kernels, wall_ms, total_events_ms, per_kernel_ms);

    if (ev_starts) {
        for (int i = 0; i < batch; i++) { cuEventDestroy(ev_starts[i]); cuEventDestroy(ev_ends[i]); }
        free(ev_starts); free(ev_ends);
    }
    cudaFree(d);
    return 0;
}
