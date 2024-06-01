#!/bin/bash

#SBATCH --job-name=training_bk_kernels_eft_z0.38
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/rneveux/out/training_bk_kernels_eft_z0.38_%x.%j.out
#SBATCH --error=/home/rneveux/err/training_bk_kernels_eft_z0.38_%x.%j.out
#SBATCH --constraint=avx

python /home/rneveux/bispectrum/BICKER/scripts/bispec/train_components--groups.py --inputX /home/rneveux/bispectrum/theory/cosmologies/forFFcomp/ --inputY /home/rneveux/kernels_EFT/bk/z0.38/Omfid.317/kernels/ --cache /home/rneveux/bicker_cache/z0.38/bispec/ --verbose 1 --ells 202