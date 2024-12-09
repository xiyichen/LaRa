#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=24:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1

# NCCL configuration
export NCCL_DEBUG=INFO

## run
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_lightning.py