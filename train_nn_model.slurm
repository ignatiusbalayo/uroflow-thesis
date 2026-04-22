#!/bin/bash
#SBATCH --job-name=uroflow_nn_train
#SBATCH --output=logs/nn_train_%j.out
#SBATCH --error=logs/nn_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wktdcourses@gmail.com

mkdir -p logs

module load python/3.10
module load cuda/11.8

source venv/bin/activate

cd /home/wilfred/sys/analytica/uroflow-thesis

python Model_train/train_nn_model.py

echo "Training complete. Model saved to nn_model_state.pt"
