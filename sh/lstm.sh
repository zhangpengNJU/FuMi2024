#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=0-12:00:00
#SBATCH --qos=normal
#SBATCH -J RUN_ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=peng.zhang@uni.lu
#SBATCH --gres=gpu:1


conda activate fumi_ad
cd /home/users/pzhang/0528/FuMi/examples/word_language_model
python /home/users/pzhang/0528/testGPU.py
(time python main.py --cuda --epochs 6 --model LSTM --lr 5) 2>&1 | tee /home/users/pzhang/0528/FuMiOutput/adlog/LSTM.log



