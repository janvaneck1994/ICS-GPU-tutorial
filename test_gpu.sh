#!/bin/bash
#SBATCH --job-name=pytorch_gpu_test
#SBATCH --partition=gpua16
#SBATCH --time=0:05:00
#SBATCH --output=test_gpu.out
#SBATCH --error=test_gpu.err

# Load necessary modules
module load cuda/12.2
module load python/3.9

# activate python venv
source ics_gpu_tutorial/bin/activate

# Test if GPU is available
python test_gpu.py
