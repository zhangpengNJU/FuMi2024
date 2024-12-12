#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=0-12:00:00
#SBATCH --qos=normal
#SBATCH -J RUN_ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=peng.zhang@uni.lu
#SBATCH --gres=gpu:1


conda activate fumi_ori
cd /home/users/pzhang/0528/FuMi/llama
(time python t.py ) 2>&1 | tee /home/users/pzhang/0528/FuMiOutput/log/llama_fa_vs_math_sdp.log



