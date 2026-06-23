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
#include <pthread.h>
#include <math.h>

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

/* Time-based throttle state.
 *
 * Algorithm: integral controller with forgetting factor (EMA).
 *
 * At each sync (cudaDeviceSynchronize / cuStreamSynchronize):
 *   burst_ns   = wall-clock time since last stall ended
 *   active_frac = burst_ns / (burst_ns + last_stall_ns)   [CPU-side, no GPU measurement]
 *   avg_gpu_frac = EMA(active_frac, gamma=0.95)
 *   computed_stall = burst_ns * (avg_gpu_frac - limit) / limit   if avg_gpu_frac > limit
 *
 */

static int             tbt_dev_ready[CUDA_DEVICE_MAX_COUNT];
static int             tbt_stall_count[CUDA_DEVICE_MAX_COUNT];
static volatile int    tbt_process_count[CUDA_DEVICE_MAX_COUNT];
static pthread_mutex_t tbt_mutex[CUDA_DEVICE_MAX_COUNT];        // guards sum_active_fracs + stall_count
static double          tbt_sum_active_fracs[CUDA_DEVICE_MAX_COUNT]; // written by watcher, read by sync hook
static int64_t tbt_chunk_ns  = 2000000LL; // default 2ms, overridden by TBT_CHUNK_MS
static int     tbt_jitter_pct = 10;         // default ±10%, overridden by TBT_JITTER_PCT

static int             tbt_in_burst[CUDA_DEVICE_MAX_COUNT];
static struct timespec tbt_burst_start[CUDA_DEVICE_MAX_COUNT];
static unsigned int    tbt_rand_seed[CUDA_DEVICE_MAX_COUNT];
static int64_t         tbt_stall_debt_ns[CUDA_DEVICE_MAX_COUNT];  // remaining stall to drip across launches

/* Shared stalling-state mmap: per-process atomic flag visible to all processes
 * on the same device. Used for work-conserving skip: if all others are stalling,
 * skip this chunk with probability = limit. */
#define TBT_STATE_MAX_PROCS 64
#define TBT_STATE_PATH_FMT  "/tmp/hami_tbt_state_%d.dat"

typedef struct {
    _Atomic int32_t pid;
    _Atomic int32_t stalling;
} tbt_proc_slot_t;

typedef struct {
    _Atomic int32_t    n_procs;
    tbt_proc_slot_t    procs[TBT_STATE_MAX_PROCS];
} tbt_shared_state_t;

static tbt_shared_state_t *tbt_state[CUDA_DEVICE_MAX_COUNT];
static int                 tbt_my_slot[CUDA_DEVICE_MAX_COUNT];

static void tbt_state_init(int dev) {
    char path[64];
    snprintf(path, sizeof(path), TBT_STATE_PATH_FMT, dev);
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) return;
    if (ftruncate(fd, sizeof(tbt_shared_state_t)) != 0) { close(fd); return; }
    tbt_state[dev] = (tbt_shared_state_t *)mmap(NULL, sizeof(tbt_shared_state_t),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (tbt_state[dev] == MAP_FAILED) { tbt_state[dev] = NULL; return; }

    pid_t my_pid = getpid();
    int n = atomic_load(&tbt_state[dev]->n_procs);
    for (int i = 0; i < n && i < TBT_STATE_MAX_PROCS; i++) {
        if (atomic_load(&tbt_state[dev]->procs[i].pid) == (int32_t)my_pid) {
            tbt_my_slot[dev] = i;
            return;
        }
    }
    int slot = atomic_fetch_add(&tbt_state[dev]->n_procs, 1);
    if (slot >= TBT_STATE_MAX_PROCS) { slot = TBT_STATE_MAX_PROCS - 1; }
    atomic_store(&tbt_state[dev]->procs[slot].pid,      (int32_t)my_pid);
    atomic_store(&tbt_state[dev]->procs[slot].stalling, 0);
    tbt_my_slot[dev] = slot;
}

static void tbt_set_stalling(int dev, int val) {
    if (!tbt_state[dev]) return;
    atomic_store(&tbt_state[dev]->procs[tbt_my_slot[dev]].stalling, val);
}

static int tbt_all_others_stalling(int dev) {
    if (!tbt_state[dev]) return 0;
    int n = atomic_load(&tbt_state[dev]->n_procs);
    if (n <= 1) return 0;
    int my_slot = tbt_my_slot[dev];
    for (int i = 0; i < n && i < TBT_STATE_MAX_PROCS; i++) {
        if (i == my_slot) continue;
        if (atomic_load(&tbt_state[dev]->procs[i].pid) == 0) continue;
        if (!atomic_load(&tbt_state[dev]->procs[i].stalling)) return 0;
    }
    return 1;
}

void tbt_shared_state_cleanup(void) {
    pid_t my_pid = getpid();
    for (int dev = 0; dev < CUDA_DEVICE_MAX_COUNT; dev++) {
        if (!tbt_state[dev]) continue;
        int slot = tbt_my_slot[dev];
        atomic_store(&tbt_state[dev]->procs[slot].stalling, 0);
        atomic_store(&tbt_state[dev]->procs[slot].pid,      0);
        atomic_fetch_sub(&tbt_state[dev]->n_procs, 1);
        (void)my_pid;
        munmap(tbt_state[dev], sizeof(tbt_shared_state_t));
        tbt_state[dev] = NULL;
    }
}
static int64_t         tbt_burst_sleep_ns[CUDA_DEVICE_MAX_COUNT]; // sleep already applied during current burst

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
      userutil[cudadev] = -1;  // sentinel: no fresh NVML sample yet this tick
      nvmlDevice_t device;
      CHECK_NVML_API(nvmlDeviceGetHandleByIndex(cudadev, &device));

      // OPTIMIZATION: Do slow NVML queries WITHOUT holding lock
      // This prevents blocking memory allocation operations

      //Get Memory for container
      nvmlReturn_t res = nvmlDeviceGetComputeRunningProcesses(device,&infcount,infos);

      // Get SM util for container — look back one tick interval so each watcher
      // tick gets a fresh sample rather than a stale 1-second aggregate.
      gettimeofday(&cur, NULL);
      microsec = (cur.tv_sec * 1000UL * 1000UL + cur.tv_usec) - g_wait.tv_nsec / 1000UL;
      nvmlProcessUtilizationSample_t processes_sample[SHARED_REGION_MAX_PROCESS_NUM];
      unsigned int processes_num = SHARED_REGION_MAX_PROCESS_NUM;
      nvmlReturn_t res2 = nvmlDeviceGetProcessUtilization(device, processes_sample, &processes_num, microsec);

      // Now acquire lock only for the brief period needed to update shared memory
      lock_shrreg();

      if (res == NVML_SUCCESS) {
        pthread_mutex_lock(&tbt_mutex[cudadev]);
        tbt_process_count[cudadev] = (int)infcount;
        pthread_mutex_unlock(&tbt_mutex[cudadev]);
        for (i=0; i<infcount; i++){
          proc = find_proc_by_hostpid(infos[i].pid);
          if (proc != NULL){
              proc->monitorused[cudadev] = infos[i].usedGpuMemory;
          }
        }
      }

      // Use this process's own smUtil rather than the sum across all processes.
      // The sum is deflated by 1/N under hardware time-slicing, making delta()
      // compare a device-wide aggregate against a per-process limit — wrong units.
      // Each process's watcher only needs to know its own utilization vs its own limit.
      int my_smutil = -1;
      pid_t my_pid = getpid();

      LOG_INFO("device %d: nvml_sample ts=%zuus n_procs=%u n_samples=%u my_pid=%d\n",
               cudadev, microsec, infcount, processes_num, my_pid);
      if (res2 == NVML_SUCCESS) {
        for (i=0; i<processes_num; i++){
          proc = find_proc_by_hostpid(processes_sample[i].pid);
          LOG_INFO("device %d: pid=%u smUtil=%u%% matched=%d\n",
                   cudadev, processes_sample[i].pid, processes_sample[i].smUtil, proc != NULL);
          if (proc != NULL){
              proc->device_util[cudadev].sm_util = processes_sample[i].smUtil;
              if ((pid_t)processes_sample[i].pid == my_pid) {
                  my_smutil = (int)processes_sample[i].smUtil;
              }
          }
        }
      }

      unlock_shrreg();

      // Only update userutil when NVML actually reported our process this tick.
      // NVML samples at ~100ms intervals; with a 30ms watcher tick most ticks
      // won't have a fresh sample.  Passing 0 to delta() when we have no data
      // causes it to spuriously increase share, so we skip delta() those ticks
      // by leaving userutil at -1 as a sentinel.
      userutil[cudadev] = my_smutil;  // -1 if not reported this tick
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

    // Allow tick interval to be overridden at runtime for tuning.
    struct timespec tick = g_wait;
    const char *tick_env = getenv("HAMI_WATCHER_TICK_MS");
    if (tick_env != NULL) {
        long ms = atol(tick_env);
        if (ms > 0 && ms <= 1000) {
            tick.tv_nsec = ms * MILLISEC;
            LOG_MSG("watcher tick overridden to %ldms via HAMI_WATCHER_TICK_MS", ms);
        }
    }

    struct timespec tick_start, tick_end;
    while (1){
        clock_gettime(CLOCK_MONOTONIC, &tick_start);
        nanosleep(&tick, NULL);
        clock_gettime(CLOCK_MONOTONIC, &tick_end);
        int64_t tick_ms = (tick_end.tv_sec - tick_start.tv_sec) * 1000
                        + (tick_end.tv_nsec - tick_start.tv_nsec) / 1000000;
        LOG_INFO("watcher tick: actual_interval=%ldms\n", tick_ms);
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
              // EMA is driven exclusively from sync boundaries (time_throttle_sync).
              // The watcher sums active_fracs from shared memory and updates tbt_sum_active_fracs.
              double sum = tbt_get_sum_active_fracs(dev);
              pthread_mutex_lock(&tbt_mutex[dev]);
              tbt_sum_active_fracs[dev] = sum;
              int stalls = tbt_stall_count[dev];
              tbt_stall_count[dev] = 0;
              pthread_mutex_unlock(&tbt_mutex[dev]);
              LOG_INFO("device %d: [time-based] limit=%d%% sum_active=%.2f target_active=%.1f%% syncs=%d n_procs=%d\n",
                       dev, cached_sm_limit[dev], sum,
                       cached_sm_limit[dev] / 100.0 * sum * 100.0,
                       stalls, tbt_process_count[dev]);
            } else {
              if ((share[dev] == g_total_cuda_cores[dev]) && (g_cur_cuda_cores[dev] < 0)) {
                g_total_cuda_cores[dev] *= 2;
                share[dev] = g_total_cuda_cores[dev];
              }

              // Skip delta() when NVML has no sample this tick (userutil=-1), UNLESS
              // the bucket is negative — a stalled process stops launching kernels so
              // NVML won't report it, but we need delta() to see 0 and increase share.
              int effective_util = userutil[dev];
              if (effective_util < 0) {
                  effective_util = (g_cur_cuda_cores[dev] < 0) ? 0 : -1;
              }
              if ((effective_util <= 100) && (effective_util >= 0)) {
                share[dev] = delta(cached_sm_limit[dev], effective_util, share[dev], dev);
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

    const char *chunk_env = getenv("TBT_CHUNK_MS");
    if (chunk_env) tbt_chunk_ns = (int64_t)(atof(chunk_env) * 1e6);
    const char *jitter_env = getenv("TBT_JITTER_PCT");
    if (jitter_env) tbt_jitter_pct = atoi(jitter_env);

    LOG_MSG("throttle mode: %s", get_time_based_throttle() ? "time-based (experimental)" : "NVML-feedback");
    LOG_MSG("tbt chunk=%.0fms jitter=±%d%%", tbt_chunk_ns / 1e6, tbt_jitter_pct);

    pthread_t tid;
    if (has_limit) {
        pthread_create(&tid, NULL, utilization_watcher, NULL);
    }
    return;
}


static void tbt_init_device(int device_id) {
    if (tbt_dev_ready[device_id]) return;
    pthread_mutex_init(&tbt_mutex[device_id], NULL);
    if (cached_sm_limit[device_id] <= 0 || cached_sm_limit[device_id] >= 100) {
        tbt_dev_ready[device_id] = 1;
        return;
    }
    // tbt_sum_active_fracs is intentionally not reset here — the watcher may have
    // already written a valid sum before the first kernel launch triggers init.
    tbt_stall_count[device_id]    = 0;
    tbt_process_count[device_id]  = 1;
    tbt_in_burst[device_id]       = 0;
    tbt_stall_debt_ns[device_id]  = 0;
    tbt_burst_sleep_ns[device_id] = 0;
    const char *seed_env = getenv("TBT_RAND_SEED");
    unsigned int base_seed = seed_env ? (unsigned int)atoi(seed_env)
                                      : (unsigned int)(getpid() ^ (unsigned int)time(NULL));
    tbt_rand_seed[device_id] = base_seed ^ (device_id * 2654435761U);
    tbt_state_init(device_id);
    tbt_set_my_active_frac(device_id, 1.0);
    tbt_dev_ready[device_id] = 1;
}

void time_throttle_pre_launch(CUstream hStream, int device_id) {
    tbt_init_device(device_id);
    if (cached_sm_limit[device_id] <= 0 || cached_sm_limit[device_id] >= 100) return;

    if (!tbt_in_burst[device_id]) {
        clock_gettime(CLOCK_MONOTONIC, &tbt_burst_start[device_id]);
        tbt_in_burst[device_id] = 1;
    }

    // Drip stall debt in small chunks with small jitter.
    // Scale chunk by (1-limit)/limit so low-limit processes drain debt at the same
    // rate as high-limit processes relative to their burst size.
    if (tbt_stall_debt_ns[device_id] <= 0) return;

    double limit_frac = cached_sm_limit[device_id] / 100.0;
    int64_t scaled_chunk = (int64_t)(tbt_chunk_ns * log(1.0 / limit_frac));
    if (scaled_chunk < tbt_chunk_ns) scaled_chunk = tbt_chunk_ns;
    int64_t chunk = tbt_stall_debt_ns[device_id] < scaled_chunk
                  ? tbt_stall_debt_ns[device_id] : scaled_chunk;
    // If all other processes are already stalling (GPU would be idle without us),
    // skip this chunk with probability = limit. Higher-limit processes fill the
    // idle GPU first, maintaining proportionality while keeping the GPU busy.
    if (tbt_all_others_stalling(device_id)) {
        double r = (double)rand_r(&tbt_rand_seed[device_id]) / RAND_MAX;
        if (r < limit_frac) {
            // Back-calculation: running instead of stalling counts as spending quota.
            tbt_stall_debt_ns[device_id] -= chunk;
            if (tbt_stall_debt_ns[device_id] < 0) tbt_stall_debt_ns[device_id] = 0;
            return;
        }
    }

    int64_t sleep_ns = chunk;

    tbt_set_stalling(device_id, 1);
    struct timespec ts = {
        .tv_sec  = sleep_ns / 1000000000LL,
        .tv_nsec = sleep_ns % 1000000000LL,
    };
    nanosleep(&ts, NULL);
    tbt_set_stalling(device_id, 0);
    tbt_stall_debt_ns[device_id]  -= sleep_ns;
    if (tbt_stall_debt_ns[device_id] < 0) tbt_stall_debt_ns[device_id] = 0;
    tbt_burst_sleep_ns[device_id] += sleep_ns;

    pthread_mutex_lock(&tbt_mutex[device_id]);
    tbt_stall_count[device_id]++;
    pthread_mutex_unlock(&tbt_mutex[device_id]);
}

void time_throttle_post_launch(CUstream hStream, int device_id) {
    // No-op.
}

void time_throttle_sync(int device_id) {
    if (!tbt_dev_ready[device_id]) return;
    if (cached_sm_limit[device_id] <= 0 || cached_sm_limit[device_id] >= 100) return;
    if (!tbt_in_burst[device_id]) return;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t burst_ns = (now.tv_sec  - tbt_burst_start[device_id].tv_sec)  * 1000000000LL
                     + (now.tv_nsec - tbt_burst_start[device_id].tv_nsec);

    double limit_frac = cached_sm_limit[device_id] / 100.0;

    pthread_mutex_lock(&tbt_mutex[device_id]);
    double sum = tbt_sum_active_fracs[device_id];
    pthread_mutex_unlock(&tbt_mutex[device_id]);
    if (sum < 1.0) sum = 1.0;

    double target = limit_frac * sum;
    int64_t burst_sleep = tbt_burst_sleep_ns[device_id];
    if (target < 1.0 && burst_ns > 0) {
        int64_t net_burst_ns = burst_ns - burst_sleep;
        if (net_burst_ns > 0)
            tbt_stall_debt_ns[device_id] += (int64_t)(net_burst_ns * (1.0 - target) / target);
    }

    double active_frac = target < 1.0 ? target : 1.0;
    tbt_set_my_active_frac(device_id, active_frac);
    tbt_in_burst[device_id]       = 0;
    tbt_burst_sleep_ns[device_id] = 0;

    LOG_INFO("device %d: [tbt sync] burst=%ldns sleep=%ldns debt=%ldns target=%.1f%% limit=%d%% sum=%.2f\n",
             device_id, burst_ns, burst_sleep,
             tbt_stall_debt_ns[device_id], target * 100.0, cached_sm_limit[device_id], sum);
}

