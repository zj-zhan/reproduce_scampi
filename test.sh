#!/bin/bash
ulimit -s unlimited
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BART_TOOLBOX_PARALLEL=0

python cartesian_scampi.py \
--data_root ../../../exp_data \
--output_dir ./results/exp \
--mask_type equispaced1d \
--acc_factor 4.0 \
--center_fraction 0.08 \
--target_size 320 \