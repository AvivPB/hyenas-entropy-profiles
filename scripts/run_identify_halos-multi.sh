#!/bin/bash -l

# module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate

# dir=/scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration
dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration
echo $dir

halo_dirs=$(ls $dir/ | grep -vE 'slurm_files')

#halo_ids=$(ls -d /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_* | grep -vE '3224|OG')
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/ | grep -vE '3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/ | grep -vE 'aviv_halo_info|3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls $dir/ | grep -vE 'OG|slurm_files' | cut -d "_" -f2)
# halo_ids=3224
halo_ids=$(ls $dir/ | grep -vE 'slurm_files' | cut -d "_" -f2 | cut -d "-" -f1)
echo $halo_ids



snap_nums=151
caesar_suffix=''
caesar_dir_type=caesar_snap

clear_output_file=--clear_output_file

target_property=m200c
domain='inside'
target_value_min=1e12
target_value_max=1e17
target_units=Msun

use_contamination=--use_contamination
return_contamination=--return_contamination
contamination_min=0
contamination_max=0

use_dist=--use_dist
return_dist=--return_dist
pos_units=Mpc

return_pos=--return_pos

use_halo_files=--use_halo_files
try_L1_halo_file=--no-try_L1_halo_file
tolerance=0.5
distance_tolerance=0.5  # in Mpc

echo

halo_ids_array=($halo_ids)
halo_dirs_array=($halo_dirs)

for i in "${!halo_ids_array[@]}"; do
	halo_id="${halo_ids_array[$i]}"
	halo_dir="${halo_dirs_array[$i]}"
	echo "Processing halo: $halo_dir with id: $halo_id"

	caesar_dir=$dir/$halo_dir/$caesar_dir_type
	echo $caesar_dir

	output_file=$caesar_dir/halo_info_snap_${snap_nums}-halo_file
	echo $output_file

	# caesar_base=Caesar_halo_${halo_id}_
	caesar_base=caesar_
	echo $caesar_base

    snap_dir=$dir/$halo_dir
    echo $snap_dir
    
    # snap_base=snap_halo_${halo_id}_
	snap_base=snapshot_
    echo $snap_base

    echo

	python identify_halos.py --caesar_dir=$caesar_dir --snap_nums=$snap_nums --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --output_file=$output_file $clear_output_file --target_property=$target_property --domain=$domain --target_value_min=$target_value_min --target_value_max=$target_value_max --target_units=$target_units $use_contamination $return_contamination --contamination_min=$contamination_min --contamination_max=$contamination_max $use_dist $return_dist --pos_units=$pos_units --snap_dir=$snap_dir --snap_base=$snap_base $use_halo_files $return_pos $try_L1_halo_file --ahf_halo_id=$halo_id --tolerance=$tolerance --distance_tolerance=$distance_tolerance

	echo

done
