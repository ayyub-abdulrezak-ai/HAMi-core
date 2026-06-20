#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "multiprocess/multiprocess_memory_limit.h"
#include "multiprocess/multiprocess_utilization_watcher.h"
#include "include/log_utils.h"
#include "include/nvml_override.h"


static int g_sm_num[CUDA_DEVICE_MAX_COUNT];
static int g_max_thread_per_sm[CUDA_DEVICE_MAX_COUNT];
static volatile int64_t g_cur_cuda_cores[CUDA_DEVICE_MAX_COUNT] = {0};
static volatile int64_t g_total_cuda_cores[CUDA_DEVICE_MAX_COUNT] = {0};
extern int pidfound;
int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

/* Cached at init — these values do not change at runtime */
static int cached_sm_limit[CUDA_DEVICE_MAX_COUNT] = {0};
static int cached_util_switch = 0;

/* Time-based throttle state — forward declared here, defined below */
typedef struct { CUevent start; CUevent end; } tbt_pair_t;
static tbt_pair_t *tbt_pairs[CUDA_DEVICE_MAX_COUNT];
static int         tbt_count[CUDA_DEVICE_MAX_COUNT];
static int         tbt_cap[CUDA_DEVICE_MAX_COUNT];
static tbt_pair_t  tbt_pending[CUDA_DEVICE_MAX_COUNT];
static int         tbt_dev_ready[CUDA_DEVICE_MAX_COUNT];
static int64_t     tbt_stall_count[CUDA_DEVICE_MAX_COUNT];
static int64_t     tbt_debited_ns[CUDA_DEVICE_MAX_COUNT];

void rate_limiter(int grids, int blocks) {
  CUdevice current_device;
  CUresult res = cuCtxGetDevice(&current_device);
  int device_id = (res == CUDA_SUCCESS) ? (int)current_device : 0;

  int64_t before_cuda_cores = 0;
  int64_t after_cuda_cores = 0;
  int64_t kernel_size = grids;

  /* Fast exit using cached values — no shared memory access needed */
  if (cached_sm_limit[device_id] >= 100 || cached_sm_limit[device_id] == 0) {
      return;
  }
  if (cached_util_switch == 0) {
      return;
  }

  while (get_recent_kernel()<0) {
    sleep(1);
  }
  set_recent_kernel(2);

  do {
CHECK:
      before_cuda_cores = g_cur_cuda_cores[device_id];
      if (before_cuda_cores < 0) {
        nanosleep(&g_cycle, NULL);
        goto CHECK;
      }
      after_cuda_cores = before_cuda_cores - kernel_size;
  } while (!CAS(&g_cur_cuda_cores[device_id], before_cuda_cores, after_cuda_cores));
}

static void change_token(int64_t delta, int device_id) {
  int64_t cuda_cores_before = 0, cuda_cores_after = 0;

  LOG_DEBUG("device %d: delta: %ld, curr: %ld", device_id, delta, g_cur_cuda_cores[device_id]);
  do {
    cuda_cores_before = g_cur_cuda_cores[device_id];
    cuda_cores_after = cuda_cores_before + delta;

    if (cuda_cores_after > g_total_cuda_cores[device_id]) {
      cuda_cores_after = g_total_cuda_cores[device_id];
    }
  } while (!CAS(&g_cur_cuda_cores[device_id], cuda_cores_before, cuda_cores_after));
}

static int64_t delta(int up_limit, int user_current, int64_t share, int device_id) {
  int utilization_diff =
      abs(up_limit - user_current) < 5 ? 5 : abs(up_limit - user_current);
  int64_t increment =
      (int64_t)g_sm_num[device_id] * (int64_t)g_sm_num[device_id] *
      (int64_t)g_max_thread_per_sm[device_id] * (int64_t)utilization_diff / 2560;

  /* Accelerate cuda cores allocation when utilization vary widely */
  if (utilization_diff > up_limit / 2) {
    increment = increment * utilization_diff * 2 / (up_limit + 1);
  }

  if (user_current <= up_limit) {
    share = (share + increment) > g_total_cuda_cores[device_id]
            ? g_total_cuda_cores[device_id]
            : (share + increment);
  } else {
    share = (share - increment) < 0 ? 0 : (share - increment);
  }

  return share;
}

unsigned int nvml_to_cuda_map(unsigned int nvmldev){
    unsigned int devcount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&devcount));
    int i=0;
    for (i=0;i<devcount;i++){
        if (cuda_to_nvml_map(i)==nvmldev)
          return i;
    }
    return -1;
}

unsigned int cuda_to_nvml_map(unsigned int cudadev){
    return cuda_to_nvml_map_array[cudadev];
}

int setspec() {
    unsigned int device_count;

    CHECK_NVML_API(nvmlInit());
    CHECK_NVML_API(nvmlDeviceGetCount(&device_count));

    for (unsigned int dev = 0; dev < device_count && dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        CUdevice cu_dev;
        CHECK_CU_RESULT(cuDeviceGet(&cu_dev, dev));
        CHECK_CU_RESULT(cuDeviceGetAttribute(&g_sm_num[dev],
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_dev));
        CHECK_CU_RESULT(cuDeviceGetAttribute(&g_max_thread_per_sm[dev],
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cu_dev));
        g_total_cuda_cores[dev] = g_max_thread_per_sm[dev] * g_sm_num[dev] * FACTOR;
        LOG_INFO("setspec: device %d sm_num=%d max_threads_per_sm=%d total_cores=%ld FACTOR=%d",
                 dev, g_sm_num[dev], g_max_thread_per_sm[dev], g_total_cuda_cores[dev], FACTOR);
    }
    return 0;
}

int get_used_gpu_utilization(int *userutil,int *sysprocnum) {
    struct timeval cur;
    size_t microsec;

    int i;
    unsigned int infcount;
    nvmlProcessInfo_v1_t infos[SHARED_REGION_MAX_PROCESS_NUM];

    unsigned int nvmlCounts;
    CHECK_NVML_API(nvmlDeviceGetCount(&nvmlCounts));

    int devi,cudadev;
    for (devi=0;devi<nvmlCounts;devi++){
      uint64_t sum=0;
      infcount = SHARED_REGION_MAX_PROCESS_NUM;
      shrreg_proc_slot_t *proc;
      cudadev = nvml_to_cuda_map((unsigned int)(devi));
      if (cudadev<0)
        continue;
      userutil[cudadev] = 0;
      nvmlDevice_t device;
      CHECK_NVML_API(nvmlDeviceGetHandleByIndex(cudadev, &device));

      // OPTIMIZATION: Do slow NVML queries WITHOUT holding lock
      // This prevents blocking memory allocation operations

      //Get Memory for container
      nvmlReturn_t res = nvmlDeviceGetComputeRunningProcesses(device,&infcount,infos);

      // Get SM util for container
      gettimeofday(&cur, NULL);
      microsec = (cur.tv_sec - 1) * 1000UL * 1000UL + cur.tv_usec;
      nvmlProcessUtilizationSample_t processes_sample[SHARED_REGION_MAX_PROCESS_NUM];
      unsigned int processes_num = SHARED_REGION_MAX_PROCESS_NUM;
      nvmlReturn_t res2 = nvmlDeviceGetProcessUtilization(device, processes_sample, &processes_num, microsec);

      // Now acquire lock only for the brief period needed to update shared memory
      lock_shrreg();

      if (res == NVML_SUCCESS) {
        for (i=0; i<infcount; i++){
          proc = find_proc_by_hostpid(infos[i].pid);
          if (proc != NULL){
              proc->monitorused[cudadev] = infos[i].usedGpuMemory;
          }
        }
      }

      if (res2 == NVML_SUCCESS) {
        for (i=0; i<processes_num; i++){
          proc = find_proc_by_hostpid(processes_sample[i].pid);
          if (proc != NULL){
              sum += processes_sample[i].smUtil;
              proc->device_util[cudadev].sm_util = processes_sample[i].smUtil;
          }
        }
      }

      unlock_shrreg();

      if (sum < 0)
        sum = 0;
      userutil[cudadev] = sum;
    }
    return 0;
}

void* utilization_watcher() {
    nvmlInit();
    int userutil[CUDA_DEVICE_MAX_COUNT];
    int sysprocnum;

    unsigned int device_count;
    if (nvmlDeviceGetCount(&device_count) != NVML_SUCCESS) {
        return NULL;
    }

    int64_t share[CUDA_DEVICE_MAX_COUNT] = {0};

    ensure_initialized();

    while (1){
        nanosleep(&g_wait, NULL);
        if (pidfound==0) {
          update_host_pid();
          if (pidfound==0)
            continue;
        }
        cached_util_switch = get_utilization_switch();
        LOG_INFO("init_utilization_watcher: util_switch=%d", cached_util_switch);
        init_gpu_device_utilization();
        get_used_gpu_utilization(userutil,&sysprocnum);

        // Calculate independently for each device
        for (unsigned int dev = 0; dev < device_count && dev < CUDA_DEVICE_MAX_COUNT; dev++) {
            if (cached_sm_limit[dev] <= 0 || cached_sm_limit[dev] >= 100) {
                continue;
            }

            if (get_time_based_throttle()) {
              int64_t tick_ns = g_wait.tv_nsec;
              int64_t grant = (int64_t)(cached_sm_limit[dev] / 100.0 * tick_ns);
              change_token(grant, dev);
              int64_t debited = tbt_debited_ns[dev];
              int64_t stalls  = tbt_stall_count[dev];
              tbt_debited_ns[dev] = 0;
              tbt_stall_count[dev] = 0;
              int util_pct = grant > 0 ? (int)(debited * 100 / grant) : 0;
              LOG_INFO("device %d: [time-based] limit=%d%% grant=%ldns debited=%ldns util=%d%% bucket=%ldns pending=%d stalls=%ld\n",
                       dev, cached_sm_limit[dev], grant, debited, util_pct,
                       g_cur_cuda_cores[dev], tbt_count[dev], stalls);
            } else {
              if ((share[dev] == g_total_cuda_cores[dev]) && (g_cur_cuda_cores[dev] < 0)) {
                g_total_cuda_cores[dev] *= 2;
                share[dev] = g_total_cuda_cores[dev];
              }

              if ((userutil[dev] <= 100) && (userutil[dev] >= 0)) {
                share[dev] = delta(cached_sm_limit[dev], userutil[dev], share[dev], dev);
                change_token(share[dev], dev);
              }
            }

            LOG_INFO("device %d: userutil=%d currentcores=%ld total=%ld limit=%d share=%ld\n",
                     dev, userutil[dev], g_cur_cuda_cores[dev], g_total_cuda_cores[dev],
                     cached_sm_limit[dev], share[dev]);
        }
    }
}

void init_utilization_watcher() {
    unsigned int device_count;
    if (nvmlDeviceGetCount(&device_count) != NVML_SUCCESS) {
        LOG_WARN("nvmlDeviceGetCount failed");
        return;
    }

    setspec();

    // Initialize cached_sm_limit for each device
    int has_limit = 0;
    for (unsigned int dev = 0; dev < device_count && dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        cached_sm_limit[dev] = get_current_device_sm_limit(dev);
        LOG_INFO("device %d: core utilization limit = %d", dev, cached_sm_limit[dev]);
        if (cached_sm_limit[dev] > 0 && cached_sm_limit[dev] <= 100) {
            has_limit = 1;
        }
    }

    LOG_MSG("throttle mode: %s", get_time_based_throttle() ? "time-based (experimental)" : "NVML-feedback");

    pthread_t tid;
    if (has_limit) {
        pthread_create(&tid, NULL, utilization_watcher, NULL);
    }
    return;
}


static void tbt_init_device(int device_id) {
    if (tbt_dev_ready[device_id]) return;
    // Skip if no limit — watcher won't grant tokens at 100%, which would deadlock the stall loop
    if (cached_sm_limit[device_id] <= 0 || cached_sm_limit[device_id] >= 100) {
        tbt_dev_ready[device_id] = 1;
        return;
    }
    // Cap bucket at one tick period so idle processes don't accumulate burst credit
    g_total_cuda_cores[device_id] = g_wait.tv_nsec;
    g_cur_cuda_cores[device_id] = 0;
    tbt_pairs[device_id] = NULL;
    tbt_count[device_id] = 0;
    tbt_cap[device_id] = 0;
    tbt_dev_ready[device_id] = 1;
}

static void tbt_drain(int device_id) {
    int drained = 0;
    while (drained < tbt_count[device_id]) {
        tbt_pair_t *pair = &tbt_pairs[device_id][drained];
        if (cuEventQuery(pair->end) != CUDA_SUCCESS)
            break;
        float ms = 0.0f;
        cuEventElapsedTime(&ms, pair->start, pair->end);
        int64_t ns = (int64_t)(ms * 1e6f);
        tbt_debited_ns[device_id] += ns;
        int64_t before, after;
        do {
            before = g_cur_cuda_cores[device_id];
            after = before - ns;
        } while (!CAS(&g_cur_cuda_cores[device_id], before, after));
        cuEventDestroy(pair->start);
        cuEventDestroy(pair->end);
        drained++;
    }
    if (drained > 0) {
        tbt_count[device_id] -= drained;
        memmove(tbt_pairs[device_id], tbt_pairs[device_id] + drained,
                tbt_count[device_id] * sizeof(tbt_pair_t));
    }
}

static void tbt_push(int device_id, tbt_pair_t pair) {
    if (tbt_count[device_id] == tbt_cap[device_id]) {
        int new_cap = tbt_cap[device_id] == 0 ? 8 : tbt_cap[device_id] * 2;
        tbt_pairs[device_id] = (tbt_pair_t *)realloc(tbt_pairs[device_id],
                                                      new_cap * sizeof(tbt_pair_t));
        tbt_cap[device_id] = new_cap;
    }
    tbt_pairs[device_id][tbt_count[device_id]++] = pair;
}

void time_throttle_pre_launch(CUstream hStream, int device_id) {
    tbt_init_device(device_id);
    // Drain completed events and stall until budget is non-negative
    do {
        tbt_drain(device_id);
        if (g_cur_cuda_cores[device_id] >= 0) break;
        tbt_stall_count[device_id]++;
        nanosleep(&g_cycle, NULL);
    } while (1);
    // Create events and record start for this kernel
    cuEventCreate(&tbt_pending[device_id].start, CU_EVENT_DEFAULT);
    cuEventCreate(&tbt_pending[device_id].end, CU_EVENT_DEFAULT);
    cuEventRecord(tbt_pending[device_id].start, hStream);
}

void time_throttle_post_launch(CUstream hStream, int device_id) {
    cuEventRecord(tbt_pending[device_id].end, hStream);
    tbt_push(device_id, tbt_pending[device_id]);
}

