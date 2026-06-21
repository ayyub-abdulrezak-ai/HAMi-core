#!/bin/bash
# Demonstrates proportional GPU compute allocation under different throttle modes.
#
# Usage: test_throttle_proportional.sh <mode> [duration] [log_level] [limit1] [limit2] ... [limitN]
#
#   mode:      none        - no throttling (baseline)
#              force       - old NVML-feedback algorithm (expected to be incorrect)
#              time-based  - new CUDA-event algorithm (expected to be correct)
#   duration:  seconds each process runs (default: 30)
#   log_level: LIBCUDA_LOG_LEVEL passed to each process (default: 0, silent)
#   limits:    SM limit in percent for each process (default: 75 25)
#              number of limits determines number of concurrent processes
#
# Examples:
#   ./test_throttle_proportional.sh time-based              # 2 processes, 75/25, 30s
#   ./test_throttle_proportional.sh time-based 30 0 75 25   # same, explicit
#   ./test_throttle_proportional.sh time-based 20 3 50 30   # 2 processes with logging

set -e

MODE="${1:-}"
DURATION="${2:-30}"
LOG_LEVEL="${3:-0}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true
LIMITS=("$@")
if [ ${#LIMITS[@]} -eq 0 ]; then
    LIMITS=(75 25)
fi
N=${#LIMITS[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/test/test_throttle_proportional"
LIBVGPU="${LIBVGPU:-${SCRIPT_DIR}/../build/libvgpu.so}"

usage() {
    echo "Usage: $0 <none|force|time-based> [duration] [limit1] [limit2] ... [limitN]" >&2
    exit 1
}

[ -z "$MODE" ] && usage
case "$MODE" in none|force|time-based) ;; *) usage ;; esac

# Validate limits
LIMIT_SUM=0
for i in $(seq 0 $((N-1))); do
    lim=${LIMITS[$i]}
    if ! [[ "$lim" =~ ^[0-9]+$ ]] || [ "$lim" -lt 0 ]; then
        echo "ERROR: limit $((i+1)) must be a non-negative integer (got '$lim')" >&2
        exit 1
    fi
    LIMIT_SUM=$((LIMIT_SUM + lim))
done
if [ "$LIMIT_SUM" -gt 100 ]; then
    echo "ERROR: sum of limits is $LIMIT_SUM, must be <= 100" >&2
    exit 1
fi

[ -f "$BINARY" ] || { echo "ERROR: $BINARY not found. Run make build-in-docker first." >&2; exit 1; }
[ -f "$LIBVGPU" ] || { echo "ERROR: $LIBVGPU not found." >&2; exit 1; }

# Temp files per process — all share one cache so processes register in the same
# shared region and the watcher can match NVML PIDs to registered processes.
SHARED_CACHE="/tmp/hami_test_$$.cache"
OUTS=(); LOGS=(); PIDS=()
for i in $(seq 1 $N); do
    OUTS+=("/tmp/hami_test_${i}_$$.out")
    LOGS+=("/tmp/hami_test_${i}_$$.log")
done

cleanup() { rm -f "${OUTS[@]}" "${LOGS[@]}" "$SHARED_CACHE"; }
trap cleanup EXIT

# Build extra env vars for each mode
case "$MODE" in
    none)        EXTRA="" ;;
    force)       EXTRA="GPU_CORE_UTILIZATION_POLICY=FORCE" ;;
    time-based)  EXTRA="GPU_CORE_UTILIZATION_POLICY=FORCE TIME_BASED_THROTTLE=true" ;;
esac

# Print header
echo "Mode:      $MODE"
echo "Duration:  ${DURATION}s"
echo "Processes: $N"
printf "Limits:    "
for i in $(seq 0 $((N-1))); do printf "process-%d=%d%%" $((i+1)) ${LIMITS[$i]}; [ $i -lt $((N-1)) ] && printf "  "; done
echo ""
echo ""

# Launch all processes
for i in $(seq 0 $((N-1))); do
    SM=${LIMITS[$i]}
    [ "$MODE" = "none" ] && SM=100
    env LD_PRELOAD="$LIBVGPU" \
        CUDA_DEVICE_SM_LIMIT=$SM \
        CUDA_DEVICE_MEMORY_SHARED_CACHE="$SHARED_CACHE" \
        LIBCUDA_LOG_LEVEL=$LOG_LEVEL \
        $EXTRA "$BINARY" "$DURATION" > "${OUTS[$i]}" 2>"${LOGS[$i]}" &
    PIDS+=($!)
done

wait "${PIDS[@]}"

# Collect counts
COUNTS=()
TOTAL=0
for i in $(seq 0 $((N-1))); do
    C=$(cat "${OUTS[$i]}")
    if [ -z "$C" ]; then
        echo "ERROR: process-$((i+1)) produced no output"
        exit 1
    fi
    COUNTS+=($C)
    TOTAL=$((TOTAL + C))
done

# Print results
for i in $(seq 0 $((N-1))); do
    printf "process-%d: %d iterations  (%.1f iter/s,  %.1f%% of GPU,  expected %d%%)\n" \
        $((i+1)) ${COUNTS[$i]} \
        $(awk "BEGIN { printf \"%.1f\", ${COUNTS[$i]} / $DURATION }") \
        $(awk "BEGIN { printf \"%.1f\", ${COUNTS[$i]} / $TOTAL * $LIMIT_SUM }") \
        ${LIMITS[$i]}
done
TOTAL_IPS=$(awk "BEGIN { printf \"%.1f\", $TOTAL / $DURATION }")
echo "Total:     $TOTAL iterations  (${TOTAL_IPS} iter/s combined)"
echo ""

print_logs() {
    if [ "$LOG_LEVEL" != "0" ]; then
        echo ""
        for i in $(seq 0 $((N-1))); do
            echo "=== process-$((i+1)) log ==="
            cat "${LOGS[$i]}"
            echo ""
        done
    fi
}

# Pass/fail: each process's actual throughput fraction must be within 15% of expected
TOLERANCE=0.15
case "$MODE" in
    none)
        echo "Result: informational (no pass/fail for unthrottled baseline)"
        print_logs
        ;;
    force|time-based)
        ALL_PASS=true
        for i in $(seq 0 $((N-1))); do
            EXPECTED_PCT=$(awk "BEGIN { printf \"%.4f\", ${LIMITS[$i]} }")
            ACTUAL_PCT=$(awk "BEGIN { printf \"%.4f\", ${COUNTS[$i]} / $TOTAL * $LIMIT_SUM }")
            OK=$(awk "BEGIN {
                diff = $ACTUAL_PCT - $EXPECTED_PCT
                if (diff < 0) diff = -diff
                print (diff <= $TOLERANCE * 100) ? \"yes\" : \"no\"
            }")
            if [ "$OK" = "no" ]; then ALL_PASS=false; fi
        done

        if [ "$MODE" = "time-based" ]; then
            if [ "$ALL_PASS" = "true" ]; then
                echo "Result: PASS — all processes within $(awk "BEGIN { printf \"%d\", $TOLERANCE * 100 }")% of expected allocation"
                print_logs
                exit 0
            else
                echo "Result: FAIL — one or more processes outside expected allocation (±$(awk "BEGIN { printf \"%d\", $TOLERANCE * 100 }")%)"
                print_logs
                exit 1
            fi
        else  # force
            if [ "$ALL_PASS" = "false" ]; then
                echo "Result: BUG CONFIRMED — old NVML-feedback path fails to enforce proportional limits"
            else
                echo "Result: UNEXPECTED PASS — old path produced correct allocation"
            fi
            print_logs
        fi
        ;;
esac
