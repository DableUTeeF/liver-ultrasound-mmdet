#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		        # Specify number of tasks per node
#SBATCH --gpus=4		                # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200203                     # Specify project name
#SBATCH -J rl_cap_train                      # Specify job name
#SBATCH --error=slurm_outputs/%j.out
#SBATCH --output=slurm_outputs/%j.out

module purge
module load Mamba/23.11.0-0
conda activate /project/lt200203-aimedi/palm/conda_envs/.conda/envs/palm_mmdet

export _TYPER_STANDARD_TRACEBACK=1
export TYPER_STANDARD_TRACEBACK=1
export ACCELERATE_DISABLE_RICH=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo $@

srun python train.py $1 
