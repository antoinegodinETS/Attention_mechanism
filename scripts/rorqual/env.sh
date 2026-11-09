# Shared Rorqual settings for the SYS819 job scripts.
#
# This file is the ONE place the allocation account is set (SBATCH_ACCOUNT, read by sbatch).
# Source it on a login node before submitting, then use plain sbatch:
#     source ~/SYS819/scripts/rorqual/env.sh
#     sbatch "$SYS819_REPO/scripts/rorqual/calibrate.sbatch"
# Jobs inherit these variables (sbatch exports the environment) and source this file again
# for the helper functions. Every value can be overridden by exporting it first.

export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-def-chdesa}"
export SALLOC_ACCOUNT="${SALLOC_ACCOUNT:-$SBATCH_ACCOUNT}"

if [ -z "${SYS819_REPO:-}" ]; then
    SYS819_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export SYS819_REPO
export PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"            # must match setup_env.sh
export VENV_DIR="${VENV_DIR:-$HOME/venvs/sys819-attention}"
# Project space persists; scratch (~/links/scratch) is purged after 60 days.
export SYS819_PROJECT="${SYS819_PROJECT:-$HOME/links/projects/$SBATCH_ACCOUNT/$USER}"
export SYS819_DATA_STORE="${SYS819_DATA_STORE:-$SYS819_PROJECT/sys819-data}"   # datasets, as in README
export SYS819_RUNS="${SYS819_RUNS:-$SYS819_PROJECT/sys819-runs}"              # runs/<run_id>/

# Load the Python module and activate the project virtualenv.
sys819_activate() {
    set +u  # Lmod reads unset variables
    module load StdEnv/2023 "$PYTHON_MODULE"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    set -u
}

# Copy datasets from project space to the node-local disk and point SYS819_DATA there.
#     sys819_stage_data cifar10 [cifar10_hr]
sys819_stage_data() {
    local dest="${SLURM_TMPDIR:?sys819_stage_data runs inside a Slurm job}/sys819-data"
    local name
    mkdir -p "$dest"
    for name in "$@"; do
        if [ ! -d "$SYS819_DATA_STORE/$name" ]; then
            echo "error: $SYS819_DATA_STORE/$name not found (see README, 'Datasets')" >&2
            return 1
        fi
        if [ "$name" = cifar10 ]; then  # the loaders need only the extracted batches
            mkdir -p "$dest/cifar10"
            cp -r "$SYS819_DATA_STORE/cifar10/cifar-10-batches-py" "$dest/cifar10/"
        else
            mkdir -p "$dest/$name"
            find "$SYS819_DATA_STORE/$name" -maxdepth 1 -type f -exec cp {} "$dest/$name/" \;
        fi
    done
    export SYS819_DATA="$dest"
    echo "data staged in $SYS819_DATA: $(du -sh "$dest" | cut -f1)"
}

# Run a queue file; forward Slurm's USR1 warning to it, and when the queue stops on that
# signal (exit 3) resubmit this job script so the next chunk resumes from last.pt.
#     sys819_run_queue <queue file> <this job script> [extra queue args...]
# Needs `#SBATCH --signal=B:USR1@<seconds>` in the job script. SYS819_MAX_CHUNKS caps resubmissions.
sys819_run_queue() {
    local queue="$1" script="$2"
    shift 2
    local chunk="${SYS819_CHUNK:-1}" status pid
    python -m attnvit.queue "$queue" "$@" &
    pid=$!
    trap 'echo "USR1 from Slurm: asking the queue to stop"; kill -USR1 "$pid" 2>/dev/null' USR1
    set +e
    wait "$pid"
    status=$?
    while kill -0 "$pid" 2>/dev/null; do  # wait returns early when the trap fires
        wait "$pid"
        status=$?
    done
    set -e
    trap - USR1
    if [ "$status" -eq 3 ]; then
        if [ "$chunk" -lt "${SYS819_MAX_CHUNKS:-5}" ]; then
            echo "queue interrupted: resubmitting $script as chunk $((chunk + 1))"
            sbatch --export=ALL,SYS819_CHUNK=$((chunk + 1)) "$script"
        else
            echo "queue interrupted and SYS819_MAX_CHUNKS reached: resubmit $script by hand"
        fi
    fi
    return "$status"
}
