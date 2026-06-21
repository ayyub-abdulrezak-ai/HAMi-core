#!/bin/bash
# Proves whether cuEventElapsedTime is inflated in the multi-kernel batch scenario
# and whether CUPTI avoids that inflation.
#
# Usage: test_cupti_vs_events.sh [duration_s] [kernels_per_batch]

set -e

DURATION="${1:-10}"
BATCH="${2:-500}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/test/test_cupti_vs_events"

[ -f "$BINARY" ] || { echo "ERROR: $BINARY not found." >&2; exit 1; }

run_mode() {
    local mode="$1"
    local nprocs="$2"
    local outs=() pids=()
    for i in $(seq 1 $nprocs); do
        outs+=("/tmp/cupti_test_${mode}_${i}_$$.out")
        "$BINARY" "$mode" "$DURATION" "$BATCH" > "${outs[$((i-1))]}" 2>/dev/null &
        pids+=($!)
    done
    wait "${pids[@]}"
    echo "--- $mode ($nprocs processes, ${DURATION}s, batch=$BATCH) ---"
    for i in $(seq 1 $nprocs); do
        echo "  proc-$i: $(cat ${outs[$((i-1))]})"
    done
    # Average per-kernel time
    avg=$(awk '{ for(i=1;i<=NF;i++) if($i~/per_kernel=/) { split($i,a,"="); sum+=a[2]; n++ } } END { printf "%.4f", sum/n }' "${outs[@]}")
    echo "  avg per_kernel: ${avg}ms"
    echo ""
    rm -f "${outs[@]}"
}

echo "=== Solo baseline (1 process, 1 kernel/sync) ==="
"$BINARY" solo_events "$DURATION" 1 2>/dev/null
echo ""

echo "=== 4 concurrent processes ==="
run_mode "solo_events"  4
run_mode "batch_events" 4
run_mode "batch_cupti"  4

echo "Expected:"
echo "  solo_events:  per_kernel ≈ solo kernel time (baseline)"
echo "  batch_events: per_kernel >> solo (inflated by queuing) IF inflation exists"
echo "  batch_cupti:  per_kernel ≈ solo (true hardware execution time)"
