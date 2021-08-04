#!/bin/bash
#SBATCH --job-name=train_H
#SBATCH --qos=normal
#SBATCH --gres=gpu:1    
#SBATCH --mem=11G
#SBATCH --partition=student

cd ~/dev/PolyGen
source ~/miniconda3/bin/activate
conda activate torch_env
export PYTHONPATH=./
python experiments/main.py -load_weights /home/z1142375/dev/PolyGen/weights/23_07_2021_23_31/epoch_1740.pt
