#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S

# load any environment modules you need (such as specific python, or cuda versions)
module load python/3.11

# execute your commands
nvidia-smi

python ~/isws/isws_vulkan/translate_english_to_italian.py
