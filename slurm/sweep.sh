#!/bin/bash
#SBATCH --job-name=vit-resnet-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm/out_sweep_%j.txt
# Learning-rate sweep: 2 models x 2 optimizers x 3 rates x 3 epochs. About 1.5 hours on an H100.
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
python comparison.py --lr-sweep
