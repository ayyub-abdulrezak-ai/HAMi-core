#!/bin/bash
# Reproduces the burst-debit over-throttling bug with a large-burst workload.
#
# Each process submits 500 kernel launches before synchronising, creating a
# large GPU burst.  For low limits this burst far exceeds one watcher-tick
# grant, driving the token bucket deeply negative and causing 2–7× over-
# throttling.
#
# The test uses two checks:
#
#   Proportional check: each process's share of total throughput matches its
#   limit fraction.  This passes even when all processes are equally over-
#   throttled, so it cannot catch the burst-debit bug on its own.
#
#   Absolute check: each process's throughput is compared against a calibrated
#   solo baseline.  A throttled process at limit L% should achieve L% of the
#   solo rate.  This is the check that catches the burst-debit bug.
#
# Usage: test_throttle_burst.sh <mode> [duration] [log_level] [limit1] [limit2] ... [limitN]
#
#   mode:      none        - no throttling (baseline)
#              force       - old NVML-feedback algorithm
#              time-based  - new CUDA-event algorithm
#   duration:  seconds each process runs (default: 30)
#   log_level: LIBCUDA_LOG_LEVEL passed to each process (default: 0, silent)
#   limits:    SM limit in percent for each process
#              (default: runs both D2-2 and C4-2 reference scenarios)
#
# Reference scenarios from context doc (T4, hami_applied_fork 0.0.4):
#   D2-2: 3 processes at 50/30/20% — expected 50/30/20, observed 32.1/30.1/9.4
#   C4-2: 4 processes at 10% each  — expected 10% each, observed ~2.8% each

set -e

MODE="${1:-}"
DURATION="${2:-30}"
LOG_LEVEL="${3:-0}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true
LIMITS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../../build/test/experimental-throttler/test_throttle_gemm"
LIBVGPU="${LIBVGPU:-${SCRIPT_DIR}/../../build/libvgpu.so}"

# Absolute throughput tolerance: actual must be >= (1 - ABS_TOLERANCE) * expected.
# The burst-debit bug produces ~13% of expected, so any reasonable tolerance catches it.
ABS_TOLERANCE=0.20

# Proportional tolerance: share of total output vs share of total limit.
PROP_TOLERANCE=0.15

usage() {
    echo "Usage: $0 <none|force|time-based> [duration] [log_level] [limit1] [limit2] ..." >&2
    exit 1
}

[ -z "$MODE" ] && usage
case "$MODE" in none|force|time-based) ;; *) usage ;; esac

[ -f "$BINARY" ] || { echo "ERROR: $BINARY not found. Run make build-in-docker first." >&2; exit 1; }
[ -f "$LIBVGPU" ] || { echo "ERROR: $LIBVGPU not found." >&2; exit 1; }

# Run a solo unthrottled calibration to get the baseline sets/s.
# This is the reference for the absolute throughput check.
calibrate() {
    local cal_dur
    cal_dur=$(awk "BEGIN { d = int($DURATION / 3); print (d < 5) ? 5 : d }")
    local cal_out="/tmp/hami_burst_cal_$$.out"
    local cal_cache="/tmp/hami_burst_cal_$$.cache"
    trap 'rm -f "$cal_out" "$cal_cache"' RETURN

    echo "Calibrating solo baseline (${cal_dur}s, no throttle)..." >&2
    "$BINARY" "$cal_dur" > "$cal_out" 2>/dev/null
    local sets
    sets=$(cat "$cal_out")
    awk "BEGIN { printf \"%.4f\", $sets / $cal_dur }"
}

run_scenario() {
    local label="$1"
    local solo_rate="$2"
    shift 2
    local -a lims=("$@")
    local n=${#lims[@]}

    local limit_sum=0
    for lim in "${lims[@]}"; do limit_sum=$((limit_sum + lim)); done

    local outs=() logs=() caches=() pids=()
    for i in $(seq 1 $n); do
        outs+=("/tmp/hami_burst_${i}_$$.out")
        logs+=("/tmp/hami_burst_${i}_$$.log")
        caches+=("/tmp/hami_burst_${i}_$$.cache")
    done

    cleanup_files() { rm -f "${outs[@]}" "${logs[@]}" "${caches[@]}"; }
    trap cleanup_files RETURN

    case "$MODE" in
        none)       extra="" ;;
        force)      extra="GPU_CORE_UTILIZATION_POLICY=FORCE" ;;
        time-based) extra="GPU_CORE_UTILIZATION_POLICY=FORCE EXPERIMENTAL_THROTTLER=true" ;;
    esac

    echo ""
    echo "--- $label (mode=$MODE, ${DURATION}s) ---"
    printf "Limits: "
    for i in $(seq 0 $((n-1))); do
        printf "process-%d=%d%%" $((i+1)) ${lims[$i]}
        [ $i -lt $((n-1)) ] && printf "  "
    done
    echo ""

    for i in $(seq 0 $((n-1))); do
        local sm=${lims[$i]}
        [ "$MODE" = "none" ] && sm=100
        env LD_PRELOAD="$LIBVGPU" \
            CUDA_DEVICE_SM_LIMIT=$sm \
            CUDA_DEVICE_MEMORY_SHARED_CACHE="${caches[$i]}" \
            LIBCUDA_LOG_LEVEL=$LOG_LEVEL \
            $extra "$BINARY" "$DURATION" > "${outs[$i]}" 2>"${logs[$i]}" &
        pids+=($!)
    done

    wait "${pids[@]}"

    local counts=() total=0
    for i in $(seq 0 $((n-1))); do
        local c
        c=$(cat "${outs[$i]}")
        [ -z "$c" ] && { echo "ERROR: process-$((i+1)) produced no output"; return 1; }
        counts+=($c)
        total=$((total + c))
    done

    local prop_pass=true abs_pass=true

    # Absolute check: total throughput across all processes should be at least
    # limit_sum% of solo_rate.  The burst-debit bug causes the GPU to sit idle
    # during bucket recovery, pulling total throughput well below this floor.
    local total_rate expected_total_rate abs_util abs_ok
    total_rate=$(awk "BEGIN { printf \"%.3f\", $total / $DURATION }")
    expected_total_rate=$(awk "BEGIN { printf \"%.3f\", $limit_sum / 100.0 * $solo_rate }")
    abs_util=$(awk "BEGIN { printf \"%.1f\", $total_rate / $solo_rate * 100 }")
    if [ "$MODE" = "none" ]; then
        abs_ok="skip"
    else
        abs_ok=$(awk "BEGIN {
            print ($total_rate >= (1 - $ABS_TOLERANCE) * $expected_total_rate) ? \"yes\" : \"no\"
        }")
    fi
    [ "$abs_ok" = "no" ] && abs_pass=false

    echo ""
    printf "  %-12s  %8s  %10s  %10s  %10s  %s\n" \
        "process" "sets" "sets/s" "prop%" "exp_prop%" "verdict"
    echo "  ------------  --------  ----------  ----------  ----------  -------"

    for i in $(seq 0 $((n-1))); do
        local rate prop_actual prop_ok verdict
        rate=$(awk "BEGIN { printf \"%.3f\", ${counts[$i]} / $DURATION }")
        prop_actual=$(awk "BEGIN { printf \"%.1f\", ${counts[$i]} / $total * $limit_sum }")

        prop_ok=$(awk "BEGIN {
            diff = $prop_actual - ${lims[$i]}
            if (diff < 0) diff = -diff
            print (diff <= $PROP_TOLERANCE * 100) ? \"yes\" : \"no\"
        }")

        if [ "$prop_ok" = "no" ]; then prop_pass=false; verdict="FAIL"; else verdict="OK"; fi

        printf "  %-12s  %8d  %10s  %9s%%  %9d%%  %s\n" \
            "process-$((i+1))" "${counts[$i]}" "$rate" \
            "$prop_actual" "${lims[$i]}" \
            "$verdict"
    done

    echo ""
    printf "  Total:  %.3f sets/s  (GPU util %.1f%%)  expected >= %.3f sets/s (%d%% of solo)  %s\n" \
        "$total_rate" "$abs_util" "$expected_total_rate" "$limit_sum" \
        "$([ "$abs_ok" = "yes" ] && echo OK || ([ "$abs_ok" = "skip" ] && echo informational || echo FAIL))"

    echo ""

    if [ "$LOG_LEVEL" != "0" ]; then
        for i in $(seq 0 $((n-1))); do
            echo "=== process-$((i+1)) log ==="
            cat "${logs[$i]}"
            echo ""
        done
    fi

    if [ "$MODE" = "none" ]; then
        echo "Result ($label): informational (no pass/fail for unthrottled baseline)"
        return 0
    fi

    local failed=false
    [ "$abs_pass" = "false" ] && {
        echo "Result ($label): FAIL (absolute) — processes running far below their configured limit"
        echo "  This indicates the burst-debit bug: bucket goes deeply negative, GPU sits idle"
        failed=true
    }
    [ "$prop_pass" = "false" ] && {
        echo "Result ($label): FAIL (proportional) — processes not getting their fair relative share"
        failed=true
    }
    if [ "$failed" = "false" ]; then
        echo "Result ($label): PASS"
        return 0
    fi
    return 1
}

if [ ${#LIMITS[@]} -gt 0 ]; then
    limit_sum=0
    for lim in "${LIMITS[@]}"; do
        if ! [[ "$lim" =~ ^[0-9]+$ ]] || [ "$lim" -lt 0 ]; then
            echo "ERROR: limit must be a non-negative integer (got '$lim')" >&2; exit 1
        fi
        limit_sum=$((limit_sum + lim))
    done
    [ "$limit_sum" -gt 100 ] && { echo "ERROR: sum of limits is $limit_sum, must be <= 100" >&2; exit 1; }

    SOLO_RATE=$(calibrate)
    echo "Solo baseline: ${SOLO_RATE} sets/s"
    run_scenario "custom" "$SOLO_RATE" "${LIMITS[@]}"
else
    SOLO_RATE=$(calibrate)
    echo "Solo baseline: ${SOLO_RATE} sets/s"
    overall=0
    run_scenario "D2-2 (50/30/20%)" "$SOLO_RATE" 50 30 20 || overall=1
    run_scenario "C4-2 (4×10%)"     "$SOLO_RATE" 10 10 10 10 || overall=1
    exit $overall
fi
