#!/bin/bash -l

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate



output_file=/scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/snap_props/v2/halo_3_props-v2
# clear_output_file=--clear_output_file

halo_types=halo
central_types=central
sim_model='Simba'
mgas=5e6 # L1
# 3.5e7 # L0
mgas_units='Msun'




echo
echo 'CALCULATING HALO PROPERTIES'
echo

python calc_extra_halo_properties-hdf5.py --output_file=$output_file.hdf5 --sim_model=$sim_model --halo_types=$halo_types --central_types=$central_types --mgas=$mgas --mgas_units=$mgas_units

echo
echo 'done'

########################################################
