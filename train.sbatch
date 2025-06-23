#!/bin/bash

#SBATCH --job-name=icvs
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=10:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --exclude=sh00,sh[10-19]

echo "venv setup"

uv venv $TMPDIR/venv --python=3.11
source $TMPDIR/venv/bin/activate
uv pip install scipy numpy tensorflow[and-cuda] opencv-python-headless scikit-image

# Training

python train_sequence.py --model_name UNet --train_path train.json --val_path valid.json
# python train_sequence.py --model_name ConvLSTM --train_path train.json --val_path valid.json
# python train_sequence.py --model_name SegNet --train_path train.json --val_path valid.json
