#!/bin/bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STATE_DIR="$ROOT/.local-jobs"
LOG_DIR="$ROOT/local-job-logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

usage() {
    echo "Usage: bash launch_batch.sh frontier|frontier-scratch|compute|compute-scratch|status|stop JOB" >&2
    exit 2
}

status_jobs() {
    local found=0 name pid_file pid
    for name in frontier frontier-scratch compute compute-scratch; do
        pid_file="$STATE_DIR/$name.pid"
        if [[ -f "$pid_file" ]]; then
            pid="$(<"$pid_file")"
            if kill -0 "$pid" 2>/dev/null; then
                echo "$name is running with PID $pid"
                found=1
            else
                rm -f "$pid_file"
            fi
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "No local RMT job is running."
    fi
}

job="${1:-}"
if [[ "$job" == "stop" ]]; then
    target="${2:-}"
    case "$target" in
        frontier|frontier-scratch|compute|compute-scratch) ;;
        *) usage ;;
    esac
    pid_file="$STATE_DIR/$target.pid"
    if [[ ! -f "$pid_file" ]]; then
        echo "$target is not running."
        exit 0
    fi
    pid="$(<"$pid_file")"
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo "$target is not running."
        exit 0
    fi
    kill -TERM "$pid"
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pid_file"
            echo "$target stopped."
            exit 0
        fi
        sleep 1
    done
    echo "$target did not stop within 60 seconds. Do not use kill -9 unless its final snapshot is expendable." >&2
    exit 1
fi

case "$job" in
    status)
        status_jobs
        exit 0
        ;;
    frontier)
        script="$ROOT/remote-sk-random-matrix-ml-esd-fixed/run_frontier_calibration.slurm"
        ;;
    frontier-scratch)
        script="$ROOT/remote-sk-random-matrix-ml-esd-fixed/run_frontier_scratch.slurm"
        ;;
    compute)
        script="$ROOT/compute_optimal_analysis/run_compute_calibration.slurm"
        ;;
    compute-scratch)
        script="$ROOT/compute_optimal_analysis/run_compute_scratch.slurm"
        ;;
    *)
        usage
        ;;
esac

if command -v sbatch >/dev/null 2>&1; then
    project="$(dirname -- "$script")"
    mkdir -p "$project/logs"
    exec sbatch --chdir="$project" "$script"
fi

for other_pid_file in "$STATE_DIR"/*.pid; do
    [[ -e "$other_pid_file" ]] || continue
    other_pid="$(<"$other_pid_file")"
    if kill -0 "$other_pid" 2>/dev/null; then
        echo "Another local RMT job is running with PID $other_pid." >&2
        echo "Wait for it, or select another free GPU and launch the script manually." >&2
        exit 1
    fi
    rm -f "$other_pid_file"
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$LOG_DIR/${job}_$stamp.log"
nohup bash "$script" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$STATE_DIR/$job.pid"
printf '%s\n' "$log" > "$STATE_DIR/$job.log"
disown "$pid" 2>/dev/null || true

echo "$job started with PID $pid"
echo "Log: $log"
