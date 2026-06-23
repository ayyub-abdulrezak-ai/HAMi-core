# Experimental Time-Based GPU Throttler

## Problem

The original HAMi-core throttler (`GPU_CORE_UTILIZATION_POLICY=FORCE`) uses NVML `smUtil` as a feedback signal. Under hardware GPU time-slicing, NVML deflates per-process `smUtil` by approximately 1/N when N processes share the GPU — a process using the GPU 50% of the time appears to use only ~17% when three processes are running. This makes the feedback signal unreliable and causes the throttle to fail.

## Design

The experimental throttler replaces NVML feedback with direct timing of GPU work at kernel launch and synchronize boundaries.

### Core algorithm

At each `cuStreamSynchronize` or `cuCtxSynchronize`, the process measures the wall-clock burst duration since its last stall ended and computes the stall debt it owes:

```
stall_debt += burst_ns * (1 - limit) / limit
```

This is the open-loop formula: if a process runs for `burst_ns` at the configured `limit`, it should stall for `burst_ns * (1-limit)/limit` to achieve the correct long-run active fraction. No feedback loop, no EMA bias.

### Debt drip

Rather than applying the full stall at the sync boundary (which causes all processes to stall simultaneously, leaving the GPU idle), the debt is dripped across subsequent kernel launches. Before each `cuLaunchKernel`, a small chunk of the outstanding debt is slept off:

```
chunk = min(debt, base_chunk * log(1/limit))
sleep(chunk ± jitter)
debt -= chunk_slept
```

The chunk is scaled by `log(1/limit)` so low-limit processes (with larger debts per burst) drain at a proportionally faster rate. Default base chunk: 2ms.

### Work-conserving skip

To prevent the GPU from idling when all processes happen to stall simultaneously (thundering herd), each process sets an atomic flag in a shared mmap'd file before sleeping. Before each chunk sleep, the process checks whether all other processes are already stalling. If so, it skips the sleep with probability equal to its limit — higher-limit processes are more likely to fill the idle GPU, maintaining proportional allocation while keeping the GPU busy.

The shared state lives at `/tmp/hami_et_state_<dev>.dat` — a small mmap'd struct with one atomic stalling bit per process. Reads and writes are pointer dereferences with no syscall overhead.

### Back-calculation anti-windup

When the work-conserving skip fires, the process runs instead of stalling. To prevent integral windup (debt accumulating while the process runs freely), the debt is reduced by the chunk amount that was skipped:

```
if skip:
    debt -= chunk   // back-calculation: running counts as spending quota
```

This keeps the debt bounded near the steady-state value and prevents over-stalling when other processes become active again.

### Cross-process active_frac

Each process writes its current active fraction to `/tmp/hami_et_<pid>_<dev>.dat`. The watcher thread sums these every 120ms and makes the sum available to the sync hook. This supports correct proportional allocation when limits don't sum to 100%.

## Enabling

```bash
export EXPERIMENTAL_THROTTLER=true
export LD_PRELOAD=/path/to/libvgpu.so
```

`EXPERIMENTAL_THROTTLER=true` is sufficient. `GPU_CORE_UTILIZATION_POLICY=FORCE` is only needed if you also want the old NVML-feedback path active — it has no effect on the time-based throttler.

## Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ET_CHUNK_MS` | 2 | Base chunk size in ms |
| `ET_JITTER_PCT` | 10 | Jitter range as ±% of chunk |
| `ET_RAND_SEED` | random | Fixed seed for reproducible tests |

## Test results (30s, ±10% proportional, ≥80% GPU util)

| Scenario | Proportional accuracy | Burst GPU util |
|----------|----------------------|----------------|
| 50/30/20% | 56.0/27.6/16.5% | 92.0% |
| 10/10/10/10% | 10.0/10.0/10.0/10.0% | 53.6% |
| 90/5/5% | 86.3/6.8/6.8% | 94.9% |
| 70/25/5% | 76.1/16.3/7.5% | 93.5% |
| 30/15/5% | 30.4/12.3/7.3% | 68.1% |

All scenarios pass at both 10s and 30s test durations.

## Running the tests

```bash
# Build first
make build-in-docker

# Proportional allocation test (1 kernel/sync)
bash test/experimental-throttler/test_throttle_proportional.sh time-based 30 0 50 30 20

# Burst throughput test (500 kernels/sync)
bash test/experimental-throttler/test_throttle_burst.sh time-based 30 0 50 30 20

# Full suite (all 5 scenarios, 30s each)
bash test/experimental-throttler/run_throttle_suite.sh test/experimental-throttler/suites/et_30s.json 0

# With max logging
bash test/experimental-throttler/run_throttle_suite.sh test/experimental-throttler/suites/et_30s.json 4
```

## Implementation

All throttle logic lives in `src/multiprocess/multiprocess_utilization_watcher.c`. The hook points are:

- `et_pre_launch()` — called before each `cuLaunchKernel`: records burst start on first launch, drips debt on subsequent launches
- `et_sync()` — called after each sync: measures burst wall time, computes and accumulates debt
