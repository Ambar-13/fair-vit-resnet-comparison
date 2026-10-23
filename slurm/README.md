# Running on the SoC compute cluster

Queues: `gpu` allows 3 hours, which is enough for every job here. Per-user cap is 8 GPUs at once (2 x A100 40 GB, 1 x A100 80 GB, 2 x H100 47 GB, 1 x H100 96 GB, 1 x H200). Cluster home is not backed up; put the repo in scratch.

```bash
# from your Mac
~/nus-vpn.sh
ssh xlogin
cd /mnt/scratch/a/ambar
git clone -b cluster-run https://github.com/Ambar-13/fair-vit-resnet-comparison.git
cd fair-vit-resnet-comparison

# once: venv, the 5 GB Food-101 download and the two weight files (login node, about 20 minutes)
bash slurm/setup.sh

# check a GPU job works
srun --partition=gpu --gpus=1 --time=00:05:00 nvidia-smi

# step 1: learning-rate sweep, one job, about 1.5 hours
sbatch slurm/sweep.sh

# if a winner sits on the top edge of the grid, go one step up (edit the --grid values) and repeat
sbatch slurm/sweep_up.sh

# pick the final rates across all sweep files; refuses while a winner is still on the edge
python choose_lr.py

# step 2: the twenty runs
sbatch slurm/runs.sh

# watch
squeue -u ambar
tail -f slurm/out_run_*.txt

# when all 20 runs/*.json exist, merge and print the summary
python report.py
```

Then copy `fair_comparison_results/all_results.csv`, `all_results.json`, `lr_sweep.csv` and `chosen_learning_rates.json` back to the Mac and commit them. Leave the checkpoints (85 MB each) in scratch.

Budget against the caps: the array keeps 4 GPUs busy out of your 8, uses 8 CPUs per task out of 256, queues 20 jobs out of 32, and every job finishes well inside the 3-hour `gpu` limit (sweep about 1.5 hours on an H100; a run about 25 minutes on an A100 or H100, under an hour on an older card). Batch size 128 fits a 12 GB card. Disk in scratch: about 18 GB (data 10, venv 6, checkpoints 2).

If an array task dies (nodes can reboot), resubmit just that task: `sbatch --array=7 slurm/runs.sh`. Each task writes its own file, so nothing else is affected.

Close the login session when done: `ssh -O exit xlogin`.
