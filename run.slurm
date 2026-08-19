#!/bin/bash

#SBATCH --job-name=uncertainty_unet
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=mps:2

#SBATCH --mem=4G
#SBATCH --time=00:30:00

# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_VRAM=4098

PROJECT_DIR=/datastore/phucngt/run_experiment
VENV_DIR=/datastore/phucngt/venv

# =========================================================
# PREPARE ENVIRONMENT
# =========================================================

cd "$PROJECT_DIR" || exit 1

module clear -f
module load shared python312

source "$VENV_DIR/bin/activate"

unset CUDA_VISIBLE_DEVICES

# =========================================================
# FIND A SUITABLE GPU
# =========================================================

CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID")
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT

echo "========================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "GPU          : $BEST_GPU"
echo "Project      : $PROJECT_DIR"
echo "Start time   : $(date)"
echo "========================================================="

# =========================================================
# PRIVATE MPS
# =========================================================

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
rm -rf "$CUDA_MPS_LOG_DIRECTORY"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES="$BEST_GPU"

# =========================================================
# RUN SOURCE CODE
# =========================================================

echo "Starting training..."
echo "Python: $(which python)"
python --version

# TODO:
# Thay dòng bên dưới bằng lệnh chạy thật của project.
# Ví dụ:
# python train.py
# hoặc:
# python run.py --config configs/train.yaml

python train.py

EXIT_CODE=$?

echo "========================================================="
echo "Training finished."
echo "Exit code   : $EXIT_CODE"
echo "End time    : $(date)"
echo "========================================================="

exit "$EXIT_CODE"
