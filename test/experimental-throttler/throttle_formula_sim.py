"""
Simulates the wall-clock rate limiter throttle formula and verifies GPU fractions.

Each process independently cycles: submit kernels (active for burst_ms), then stall.
GPU is time-sliced round-robin among all active processes.
We verify that each process achieves its configured limit% of GPU time.

Formula:
  At end of each burst, process observes its own GPU fraction during that burst:
    my_gpu_frac = (GPU time received during burst) / burst_ms

  This is exactly what NVML smUtil reports for our process (our own utilization).
  No need to observe others.

  stall = burst × (my_gpu_frac - limit%) / limit%   when my_gpu_frac > limit%
  stall = 0                                          when my_gpu_frac <= limit%

Derivation: we want my_gpu_frac_overall = limit%.
  my_gpu = my_gpu_frac × burst_ms
  total_cycle = burst_ms + stall
  my_gpu / total_cycle = limit%
  → stall = burst × (my_gpu_frac - limit%) / limit%
"""

BURST_MS = 100.0   # wall-clock duration of one batch (fixed)
SIM_MS   = 100_000 # total simulation duration

def simulate(limits, burst_ms=BURST_MS, sim_ms=SIM_MS):
    N = len(limits)
    gpu_time      = [0.0] * N  # total GPU time (for result reporting)
    next_active   = [0.0] * N
    burst_gpu     = [0.0] * N  # GPU time in current burst
    total_gpu     = [0.0] * N  # cumulative GPU time (for formula)
    total_wall    = [0.0] * N  # cumulative wall-clock time (for formula)

    import heapq

    # Event-driven: no dt, exact event times, GPU sharing computed analytically
    # between consecutive events.
    # State per process: 'active' or 'stalling', plus time of next event.
    state      = ['active'] * N
    next_event = [burst_ms] * N   # first event = end of first burst for each process

    events = [(burst_ms, i) for i in range(N)]
    heapq.heapify(events)

    current_time = 0.0

    while events:
        t, i = heapq.heappop(events)
        t = min(t, sim_ms)

        # Advance time from current_time to t: GPU shared among all active processes
        duration = t - current_time
        if duration > 0:
            active = [j for j in range(N) if state[j] == 'active']
            if active:
                share = duration / len(active)
                for j in active:
                    gpu_time[j]  += share
                    burst_gpu[j] += share

        current_time = t
        if current_time >= sim_ms:
            break

        # Process event for process i
        if state[i] == 'active':
            # Burst just ended: compute stall
            gamma = 0.95
            total_gpu[i]  = gamma * total_gpu[i]  + burst_gpu[i]
            total_wall[i] = gamma * total_wall[i] + burst_ms
            burst_gpu[i]  = 0.0

            avg_gpu_frac = total_gpu[i] / total_wall[i]
            lim   = limits[i]
            stall = burst_ms * (avg_gpu_frac - lim) / lim if avg_gpu_frac > lim else 0.0

            state[i]      = 'stalling'
            next_event[i] = current_time + stall
        else:
            # Stall just ended: start next burst
            state[i]      = 'active'
            next_event[i] = current_time + burst_ms

        heapq.heappush(events, (next_event[i], i))

    fracs     = [g / sim_ms for g in gpu_time]
    total_gpu = sum(gpu_time)
    return fracs, total_gpu / sim_ms

def run_test(name, limits, tolerance=0.02):
    fracs, util = simulate(limits)
    N = len(limits)
    passed = True
    errors = []
    for i, (lim, frac) in enumerate(zip(limits, fracs)):
        if abs(frac - lim) > tolerance:
            passed = False
            errors.append(f"P{i+1}: got {frac:.3f} expected {lim:.3f}")
    status = "PASS" if passed else "FAIL"
    fracs_str = " ".join(f"P{i+1}={f:.3f}" for i, f in enumerate(fracs))
    limits_str = "/".join(f"{l:.0%}" for l in limits)
    print(f"[{status}] {name:35s} limits={limits_str:15s} got={fracs_str}  util={util:.3f}")
    if errors:
        for e in errors:
            print(f"         ERROR: {e}")
    return passed

cases = [
    # Equal splits — no process wants less than 1/N, so no stalling needed
    ("Equal 2 processes 50/50",          [0.50, 0.50]),
    ("Equal 3 processes 33/33/33",       [1/3,  1/3,  1/3]),
    ("Equal 4 processes 25/25/25/25",    [0.25, 0.25, 0.25, 0.25]),

    # All processes below equal share — throttling kicks in
    ("4 procs at 10% each (40% total)",  [0.10, 0.10, 0.10, 0.10]),
    ("3 procs at 20% each (60% total)",  [0.20, 0.20, 0.20]),
    ("2 procs at 30% each (60% total)",  [0.30, 0.30]),
    ("4 procs at 5% each (20% total)",   [0.05, 0.05, 0.05, 0.05]),

    # Mixed: some above, some below equal share
    ("D2-2: 50/30/20%",                  [0.50, 0.30, 0.20]),
    ("60/25/15%",                         [0.60, 0.25, 0.15]),
    ("70/20/10%",                         [0.70, 0.20, 0.10]),
    ("80/15/5%",                          [0.80, 0.15, 0.05]),
    ("50/30/10/10%",                      [0.50, 0.30, 0.10, 0.10]),
    ("40/30/20/10%",                      [0.40, 0.30, 0.20, 0.10]),

    # Limits summing to 100%
    ("2 procs 70/30%",                   [0.70, 0.30]),
    ("2 procs 90/10%",                   [0.90, 0.10]),
    ("3 procs 50/30/20%",                [0.50, 0.30, 0.20]),
    ("5 procs 40/20/20/10/10%",          [0.40, 0.20, 0.20, 0.10, 0.10]),

    # Limits summing to less than 100% (GPU will be underutilized)
    ("3 procs at 10/10/10% (30% total)", [0.10, 0.10, 0.10]),
    ("2 procs at 15/10% (25% total)",    [0.15, 0.10]),
    ("4 procs at 15/10/10/5% (40%)",     [0.15, 0.10, 0.10, 0.05]),
]

print(f"Simulating {len(cases)} test cases (burst={BURST_MS}ms, sim={SIM_MS}ms, tolerance=±2%)\n")
results = [run_test(name, limits) for name, limits in cases]
print(f"\n{sum(results)}/{len(results)} passed")
