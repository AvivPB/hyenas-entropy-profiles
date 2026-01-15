#!/bin/bash -l
#########################################################

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


snap_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_good-correct_jet
snap_base=snapshot_

caesar_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_good-correct_jet/caesar_snap
# echo $caesar_dir

caesar_base=caesar_
caesar_suffix=''

source_snap_nums=151
# target_snap_nums=$(seq 50 1 272)
source_halo_ids=0

n_most=1
nproc=192

output_file=$caesar_dir/halo_${source_halo_ids}_snap_${source_snap_nums}_progen_info
clear_output_file=--clear_output_file


echo
echo 'Identifying Progenitors'
echo

python identify_halo_progenitors.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --source_snap_nums=$source_snap_nums --target_snap_nums {151..0} --source_halo_ids $source_halo_ids --n_most=$n_most --nproc=$nproc --output_file=$output_file $clear_output_file

echo
echo 'done'

########################################################
