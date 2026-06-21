#!/bin/bash
# Throttle test suite runner.
#
# Usage: run_throttle_suite.sh <suite.json> [log_level]
#
# Examples:
#   bash test/run_throttle_suite.sh test/suites/micro.json
#   bash test/run_throttle_suite.sh test/suites/full.json 4

set -e

SUITE="${1:-}"
LOG_LEVEL="${2:-0}"

[ -z "$SUITE" ] && { echo "Usage: $0 <suite.json> [duration] [log_level]" >&2; exit 1; }
[ -f "$SUITE" ] || { echo "ERROR: $SUITE not found" >&2; exit 1; }
command -v jq &>/dev/null || { echo "ERROR: jq not installed" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROPORTIONAL="${SCRIPT_DIR}/test_throttle_proportional.sh"
BURST="${SCRIPT_DIR}/test_throttle_burst.sh"

overall=0

run() {
    local script="$1" mode="$2" duration="$3" label="$4"; shift 4
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [$mode] $label  (${duration}s)"
    echo "════════════════════════════════════════════════════════"
    bash "$script" "$mode" "$duration" "$LOG_LEVEL" "$@" || overall=1
}

n=$(jq 'length' "$SUITE")

for (( idx=0; idx<n; idx++ )); do
    label=$(jq -r ".[$idx].label" "$SUITE")
    duration=$(jq -r ".[$idx].duration" "$SUITE")
    limits=($(jq -r ".[$idx].limits[]" "$SUITE"))
    modes=($(jq -r ".[$idx].modes[]" "$SUITE"))
    for mode in "${modes[@]}"; do
        run "$PROPORTIONAL" "$mode" "$duration" "Proportional $label" "${limits[@]}"
        run "$BURST"        "$mode" "$duration" "Burst        $label" "${limits[@]}"
    done
done

echo ""
echo "════════════════════════════════════════════════════════"
if [ "$overall" -eq 0 ]; then
    echo "  SUITE RESULT: ALL PASSED  ($SUITE)"
else
    echo "  SUITE RESULT: FAILURES DETECTED  ($SUITE)"
fi
echo "════════════════════════════════════════════════════════"
exit $overall
