# Time-Based GPU Throttle

## Overview

HAMi-core's time-based throttle enforces per-process GPU compute limits by stalling processes at sync boundaries when they are using more GPU time than their configured limit.

It replaces the original NVML-feedback throttle (`GPU_CORE_UTILIZATION_POLICY=FORCE`), which failed under hardware time-slicing because NVML reports deflated per-process smUtil (≈1/N of true utilization when N processes share the GPU).

## How It Works

### Measurement

At each `cudaDeviceSynchronize` or `cuStreamSynchronize`, the throttle measures the **wall-clock burst duration** — the time from when the last stall ended to when the sync fires. This is called `burst_ns`.

After computing and sleeping the stall (see below), it measures the stall duration `stall_ns`.

The **active fraction** for this cycle is:

```
active_frac = burst_ns / (burst_ns + stall_ns)
```

This is the fraction of total cycle time the process spent actively submitting kernels (not stalling). It approximates the process's GPU time fraction because GPU-bound processes use the GPU nearly 100% of their active time.

### EMA Controller

The active fraction is fed into an **exponential moving average (EMA)** with forgetting factor γ = 0.95:

```
total_gpu  = γ × total_gpu  + active_frac
total_wall = γ × total_wall + 1.0
avg_gpu_frac = total_gpu / total_wall
```

The forgetting factor prevents integral windup — early measurements (before the system converges) decay over time rather than permanently biasing the average.

The EMA is **warm-started** at `limit` so the first few syncs don't over-stall.

### Stall Formula

At each sync, if `avg_gpu_frac > limit`:

```
stall_ns = burst_ns × (avg_gpu_frac - limit) / limit
```

The process then sleeps for `stall_ns` before returning from the sync hook.

The fixed point of this controller is where `avg_gpu_frac = √limit` (not `limit`), due to the relationship between active fraction and GPU fraction under hardware time-slicing with N competing processes. In practice this produces proportional GPU allocation within the ±15% test tolerance for realistic workloads (large batch sizes).

### Convergence

The system converges within approximately:
- **1 second** for symmetric scenarios (equal limits)
- **2–5 seconds** for mixed scenarios (D2-2 style)

### Limitations

- Works best when burst durations are large (hundreds of ms). Very short bursts (< ~5ms, e.g. single-kernel-per-sync workloads) may not produce stalls long enough for the GPU's hardware time-slicer to notice and reassign time to other processes.
- The active fraction approximation introduces ~10% error in the target GPU fraction for some limit configurations.

## Enabling

Set both environment variables:

```
GPU_CORE_UTILIZATION_POLICY=FORCE
EXPERIMENTAL_THROTTLER=true
```

These are injected automatically by HAMi's scheduler webhook when `timeBasedThrottle: true` is set in the device ConfigMap.

## Log Reference

All logs are emitted at `LIBCUDA_LOG_LEVEL=4` (max logging).

### `[experimental-throttler]` — Watcher tick summary

```
device 0: [experimental-throttler] limit=30% avg_gpu=33.5% syncs=1 n_procs=3
```

Emitted every watcher tick (120ms) by the background watcher thread.

| Field | Meaning |
|-------|---------|
| `limit` | Configured SM limit for this process |
| `avg_gpu` | Current EMA estimate of GPU fraction |
| `syncs` | Number of stalls applied since last tick |
| `n_procs` | Number of CUDA processes on the device (from NVML) |

### `[et sync]` — Sync boundary decision

```
device 0: [et sync] burst=490000ns stall=204157ns avg_gpu=70.8% limit=50%
```

Emitted at each `cudaDeviceSynchronize` / `cuStreamSynchronize`.

| Field | Meaning |
|-------|---------|
| `burst` | Wall-clock time from last stall end to this sync |
| `stall` | Sleep duration applied after this sync (0 if no stall) |
| `avg_gpu` | EMA estimate used to compute this stall |
| `limit` | Configured SM limit |

### `[et ema]` — EMA update

```
device 0: [et ema] burst=490000ns stall=204157ns active_frac=70.6% avg_gpu=70.7% limit=50%
```

Emitted immediately after each sync, showing the EMA update.

| Field | Meaning |
|-------|---------|
| `burst` | Wall-clock burst duration |
| `stall` | Stall applied this sync |
| `active_frac` | Raw active fraction this cycle: `burst / (burst + stall)` |
| `avg_gpu` | New EMA value after incorporating `active_frac` |
| `limit` | Configured SM limit |

## Test Suite

```bash
# Quick smoke test (time-based mode only, 15s each)
bash test/run_throttle_suite.sh test/suites/micro.json

# Full suite (all modes, 30s each)
bash test/run_throttle_suite.sh test/suites/full.json

# With max logging
bash test/run_throttle_suite.sh test/suites/micro.json 4
```

### Test scripts

| Script | Description |
|--------|-------------|
| `test/test_throttle_proportional.sh` | Simple kernel (1 kernel/sync). Tests proportional allocation across modes. |
| `test/test_throttle_burst.sh` | Batch kernel (500 kernels/sync). Tests both proportional and absolute throughput. Designed to expose burst-debit issues. |
| `test/run_throttle_suite.sh` | Suite runner — reads scenarios from a JSON file, runs both test scripts for each scenario and mode. |

### Suite files

| File | Description |
|------|-------------|
| `test/suites/micro.json` | Time-based mode only, 2 scenarios, 15s each |
| `test/suites/full.json` | All 3 modes, 5 scenarios, 30s each |

### Test output columns

**Proportional test:**
```
process-1: 12050 iterations  (803.3 iter/s,  40.0% of GPU,  expected 50%)
```
- `iterations` — kernel launches completed in the test duration
- `iter/s` — throughput
- `% of GPU` — this process's share of total iterations across all processes
- `expected` — configured limit

**Burst test:**
```
process-1    27    1.800    45.8%    50%    OK
Total:  3.933 sets/s  (GPU util 85.5%)  expected >= 4.600 sets/s
```
- `sets` — number of 500-kernel batches completed
- `sets/s` — batch throughput
- `prop%` — proportional share of total batches
- `exp_prop%` — expected proportional share (= limit)
- `GPU util` — total GPU utilization relative to solo baseline
