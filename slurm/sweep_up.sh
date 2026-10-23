#!/bin/bash
#SBATCH --job-name=vit-resnet-sweep-up
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm/out_sweep_up_%j.txt
# One step above the main grid, run when a winner sat on the grid edge. 4 short runs, about 25 minutes.
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_OFFLINE=1
source venv/bin/activate
# Food-101 lives as one tarball in scratch; each job unpacks it to the node's local disk
# (about a minute) instead of reading 101,000 small files over NFS.
[ -n "$SLURM_JOB_ID" ] || { echo "not inside a Slurm job"; exit 1; }
FOOD101_ROOT=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/food101_${SLURM_JOB_ID}.XXXXXX")
export FOOD101_ROOT
trap 'rm -rf "$FOOD101_ROOT"' EXIT
tar xzf data/food-101.tar.gz -C "$FOOD101_ROOT"
python comparison.py --lr-sweep --grid AdamW=0.003 SGD=0.3 --tag lr_sweep_up
