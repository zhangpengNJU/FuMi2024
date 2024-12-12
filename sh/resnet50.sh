#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=0-32:00:00
#SBATCH --qos=normal
#SBATCH -J RUN_ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=peng.zhang@uni.lu
#SBATCH --gres=gpu:1


conda activate fumi_ad
cd /home/users/pzhang/0528/FuMi/examples/imagenet
python /home/users/pzhang/0528/testGPU.py
(time python main.py -a resnet50 --dummy --epoch 1 -b 32) 2>&1 | tee /home/users/pzhang/0528/FuMiOutput/adlog/resnet50output.log

