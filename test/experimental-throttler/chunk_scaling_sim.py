"""
Compare chunk scaling formulas for the time-based throttle debt drip.

For a process with SM limit L, the debt per burst = burst * (1-L)/L.
With N launches per burst, the chunk needs to satisfy:
    chunk * N >= debt  =>  chunk >= burst * (1-L) / (L * N)

We want chunk(L) that:
  - At L=0.50: ~5ms (small, drains quickly)
  - At L=0.05: ~12ms (larger, but not so large it serializes launches)
  - Smooth and monotonically decreasing with L

We also track how many launches are needed to drain the debt
(bursts_to_drain), and the max sleep per launch (chunk itself).
"""

import math

BASE_MS    = 5.0
BURST_MS   = 220.0   # approximate GPU burst time under 3-way sharing
N_LAUNCHES = 500     # kernels per burst (burst test)

limits = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

candidates = {
    "fixed":          lambda L: BASE_MS,
    "sqrt(debt_rate)":lambda L: BASE_MS * math.sqrt((1 - L) / L),
    "1/L^(1/3)":      lambda L: BASE_MS / (L ** (1/3)),
    "1/L^(1/4)":      lambda L: BASE_MS / (L ** (1/4)),
    "log(1/L)":       lambda L: BASE_MS * math.log(1/L),
}

print(f"{'limit':>6}  {'debt_ms':>8}  ", end="")
for name in candidates:
    print(f"{name:>16}", end="")
print()
print("-" * (6 + 2 + 8 + 2 + 18 * len(candidates)))

for L in limits:
    debt_ms = BURST_MS * (1 - L) / L
    print(f"{L:>6.0%}  {debt_ms:>8.1f}  ", end="")
    for name, fn in candidates.items():
        chunk = min(fn(L), debt_ms)  # cap at debt (no point sleeping more than owed)
        launches_to_drain = math.ceil(debt_ms / chunk) if chunk > 0 else float('inf')
        bursts_to_drain   = launches_to_drain / N_LAUNCHES
        print(f"  {chunk:5.1f}ms/{bursts_to_drain:4.1f}b", end="")
    print()

MAX_CHUNK_MS = 15.0  # above this, per-launch delays start serializing GPU pipeline

print()
print(f"Ranking (lower = better): max(bursts_to_drain) penalized if chunk > {MAX_CHUNK_MS}ms")
print(f"{'formula':>20}  {'max_chunk':>10}  {'max_bursts':>12}  {'score':>8}  {'verdict':>10}")
print("-" * 70)

for name, fn in candidates.items():
    chunks   = [min(fn(L), BURST_MS * (1-L)/L) for L in limits]
    debts    = [BURST_MS * (1-L)/L for L in limits]
    drains   = [math.ceil(d/c)/N_LAUNCHES for c, d in zip(chunks, debts)]
    max_chunk  = max(chunks)
    max_bursts = max(drains)
    penalty    = max(0, max_chunk - MAX_CHUNK_MS) * 0.5  # 0.5 bursts penalty per ms over cap
    score      = max_bursts + penalty
    verdict    = "OK" if max_chunk <= MAX_CHUNK_MS else f"OVER by {max_chunk - MAX_CHUNK_MS:.1f}ms"
    print(f"{name:>20}  {max_chunk:>9.1f}ms  {max_bursts:>11.2f}b  {score:>8.2f}  {verdict:>10}")
