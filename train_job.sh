#!/bin/bash
#SBATCH --job-name=tlsdd_training
#SBATCH --output=/home/gandluriv/tlsdd_output_%j.log
#SBATCH --error=/home/gandluriv/tlsdd_error_%j.log
#SBATCH --partition=long           # Specify GPU partition
#SBATCH --time=6:00:00            # Set max job time
#SBATCH --mem=16G                  # Set memory
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

cd ./

python train.py
