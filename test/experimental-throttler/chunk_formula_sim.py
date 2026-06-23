"""
Compare adaptive chunk size formulae for the time-based throttle debt drip.

We want chunk = f(debt) with no external parameters that:
  1. Drains debt quickly (fewer launches needed)
  2. Keeps individual chunks small (avoids serializing GPU pipeline)
  3. Works for 1-launch bursts (proportional test) and 500-launch bursts (burst test)

Simulation: for a given (limit, N_launches_per_burst), compute the steady-state
active_frac achieved vs the target active_frac = limit.
"""

import math

BURST_NS   = 220_000_000  # 220ms wall time per burst (3-way sharing of 73ms GPU time)
LIMITS     = [0.50, 0.30, 0.20, 0.10, 0.05]
N_LAUNCHES = [1, 10, 50, 500]

def simulate(chunk_fn, limit, n_launches, bursts=200):
    """Run N bursts and return steady-state active_frac."""
    debt = 0
    total_active = 0
    total_wall   = 0

    for _ in range(bursts):
        burst_sleep = 0
        for _ in range(n_launches):
            if debt <= 0:
                break
            chunk     = chunk_fn(debt, limit)
            chunk     = min(chunk, debt)
            burst_sleep += chunk
            debt      -= chunk

        # burst wall time = BURST_NS + burst_sleep (sleep extends wall time)
        burst_wall = BURST_NS + burst_sleep
        net_burst  = burst_wall - burst_sleep  # = BURST_NS always

        target     = limit  # target active_frac
        new_debt   = int(net_burst * (1 - target) / target)
        debt      += new_debt

        total_active += BURST_NS
        total_wall   += burst_wall

    return total_active / total_wall if total_wall > 0 else 0

# --- Candidate formulae: chunk = f(debt, limit) ---

candidates = {
    "debt*limit":          lambda d, L: d * L,
    "sqrt(debt*0.01ms)":   lambda d, L: math.sqrt(d * 10_000),
    "sqrt(debt*0.1ms)":    lambda d, L: math.sqrt(d * 100_000),
    "debt*0.002":          lambda d, L: d * 0.002,
    "debt*0.005":          lambda d, L: d * 0.005,
    "debt^0.6/scale":      lambda d, L: (d ** 0.6) / 500,
    "5ms*log(1/limit)":    lambda d, L: 3_000_000 * math.log(1/L),  # current (with base 3ms)
}

# Score: mean absolute error of achieved active_frac vs limit, across all (limit, N) pairs
# weighted more heavily for common cases

print(f"\n{'formula':>22}", end="")
for L in LIMITS:
    print(f"  L={L:.0%}", end="")
print("   MAE   max_err")
print("-" * (22 + 8 * len(LIMITS) + 18))

scores = {}
for name, fn in candidates.items():
    print(f"{name:>22}", end="")
    errors = []
    for L in LIMITS:
        # average across N_launches values (weighted toward common cases)
        achieved_list = [simulate(fn, L, N) for N in N_LAUNCHES]
        achieved_avg  = sum(achieved_list) / len(achieved_list)
        err = abs(achieved_avg - L)
        errors.append(err)
        print(f"  {achieved_avg:.0%}", end="")
    mae     = sum(errors) / len(errors)
    max_err = max(errors)
    scores[name] = mae
    print(f"  {mae:.3f}  {max_err:.3f}")

print()
winner = min(scores, key=scores.get)
print(f"Winner (lowest MAE): {winner}")
