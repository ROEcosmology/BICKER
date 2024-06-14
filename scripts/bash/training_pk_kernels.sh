#!/bin/bash

#SBATCH --job-name=training_pk_kernels_eft_z0.61
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/rneveux/out/training_pk_kernels_eft_z0.61_%x.%j.out
#SBATCH --error=/home/rneveux/err/training_pk_kernels_eft_z0.61_%x.%j.err
#SBATCH --constraint=avx

python /home/rneveux/bispectrum/BICKER/scripts/powerspec/train_components--poles.py --inputX /home/rneveux/bispectrum/theory/cosmologies/lnAs/ --cosmo_params omega_cdm omega_b h ln10^{10}A_s n_s --inputY /home/rneveux/kernels_EFT/pk/z0.61/Omfid.317/full_cosmo/ --cache /home/rneveux/bicker_cache/z0.61/ --verbose 1