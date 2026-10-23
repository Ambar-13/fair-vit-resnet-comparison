#!/bin/bash
#SBATCH --job-name=vit-resnet-run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --array=0-19%4
#SBATCH --output=slurm/out_run_%A_%a.txt
# One configuration per array task: 2 models x 2 optimizers x 5 seeds = 20 tasks.
# About 25 minutes each on an A100 or H100, up to 2 hours on an older card.
# %4 keeps four tasks running at once, inside the per-user GPU cap.
# Submit after sweep.sh has written fair_comparison_results/chosen_learning_rates.json.
# To pin a card type, change --gpus=1 to for example --gpus=a100-40:1.
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
models=(ViT-Small ResNet-34)
optimizers=(AdamW SGD)
seeds=(42 101 1337 2024 888)
i=$SLURM_ARRAY_TASK_ID
model=${models[$(( i / 10 ))]}
optimizer=${optimizers[$(( (i / 5) % 2 ))]}
seed=${seeds[$(( i % 5 ))]}
python comparison.py --run "$model" "$optimizer" "$seed"
