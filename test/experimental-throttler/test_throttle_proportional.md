# Proportional GPU Throttle Test

## What this tests

HAMi enforces per-process GPU compute limits via `CUDA_DEVICE_SM_LIMIT`. When multiple processes share a GPU simultaneously, the throttle should allocate GPU time proportionally to their configured limits — a process at 70% should complete roughly 14× more work than one at 5%.

The old algorithm (`GPU_CORE_UTILIZATION_POLICY=FORCE`) uses NVML's `smUtil` as a feedback signal. Under hardware GPU time-slicing, NVML reports `smUtil=0` for a process whenever another process holds the GPU, causing spurious token bucket refills. This prevents the throttle from draining properly, breaking proportional allocation.

The new algorithm (`EXPERIMENTAL_THROTTLER=true`) replaces NVML feedback with direct CUDA event timing. Each process measures its own actual GPU execution time and debits its token budget accordingly. Hardware time-slicing is invisible to this measurement — the throttle enforces limits correctly regardless of how many processes are competing.

## Running the test

```bash
# Single mode, single configuration
bash test/test_throttle_proportional.sh <none|force|time-based> [duration_s] [limit1] [limit2] ... [limitN]

# Examples
bash test/test_throttle_proportional.sh time-based 30 75 25
bash test/test_throttle_proportional.sh force 20 70 25 5
bash test/test_throttle_proportional.sh none 20 50 30 20
```

Requires `build/libvgpu.so` and `build/test/test_throttle_proportional` — run `make build-in-docker` first.

Pass/fail: each process must land within ±15% of its expected allocation fraction. Only `time-based` is evaluated for pass/fail; `force` reports whether the bug is confirmed; `none` is informational.

## Results (20s, 3 processes)

### Allocation accuracy (% of total iterations)

**50 / 30 / 20:**

| Mode | Process-1 (expected 50%) | Process-2 (expected 30%) | Process-3 (expected 20%) |
|------|--------------------------|--------------------------|--------------------------|
| none | 33.3% | 33.3% | 33.3% |
| force | 55.5% | 40.4% | 4.1% |
| time-based | 46.3% | 32.2% | 21.5% |

**70 / 25 / 5:**

| Mode | Process-1 (expected 70%) | Process-2 (expected 25%) | Process-3 (expected 5%) |
|------|--------------------------|--------------------------|-------------------------|
| none | 33.3% | 33.3% | 33.4% |
| force | 99.4% | 0.5% | 0.1% |
| time-based | 68.6% | 26.1% | 5.2% |

**90 / 5 / 5:**

| Mode | Process-1 (expected 90%) | Process-2 (expected 5%) | Process-3 (expected 5%) |
|------|--------------------------|-------------------------|-------------------------|
| none | 33.4% | 33.3% | 33.3% |
| force | 95.7% | 2.1% | 2.2% |
| time-based | 89.9% | 5.0% | 5.1% |

`none` gives equal shares regardless of limits. `force` fails to enforce proportional allocation — the highest-limit process dominates while lower-limit processes are starved. `time-based` tracks the target allocation closely across all configurations.

### Combined throughput (iter/s)

| Configuration | none | force | time-based |
|---------------|------|-------|------------|
| 50/30/20 | 2,044 | 1,940 | 2,055 |
| 70/25/5  | 2,045 | 2,098 | 2,116 |
| 90/5/5   | 2,046 | 2,122 | 2,179 |

The time-based path introduces no measurable throughput overhead relative to the unthrottled baseline. The slight upside in some configurations is due to reduced GPU contention when lower-priority processes are correctly throttled.
