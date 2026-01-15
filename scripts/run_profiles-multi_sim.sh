#!/bin/bash -l

umask 007

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate



## Find directories on which to run profiles
# dir=/scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration
top_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration
echo $top_dir
echo

halo_dirs=$(ls $top_dir/ | grep -vE 'slurm_files')
echo "Available halo directories:"
echo $halo_dirs
echo

#halo_ids=$(ls -d /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_* | grep -vE '3224|OG')
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/ | grep -vE '3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/ | grep -vE 'aviv_halo_info|3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls $dir/ | grep -vE 'OG|slurm_files' | cut -d "_" -f2)
# halo_ids=3224
ahf_halo_ids=$(ls $top_dir/ | grep -vE 'slurm_files' | cut -d "_" -f2 | cut -d "-" -f1)
echo $ahf_halo_ids
echo



sim=simbac
code=Simba-C
# redshift=0

xscale=R500
ndim=3
filter=Sphere
profile_type=Profile
weight_by=mass

temp_cut='5e5 K'
nh_cut='0.13 cm**-3'
rho_cut=500  # in units of rho_crit

snap_nums=151
snap_file=snapshot_${snap_nums}
caesar_dir_type=caesar_snap
caesar_file=caesar_${snap_nums}
profiles_dir_type=profiles-new
suffix=all_props

halo_particles=--halo_particles
dm_particles=--dm_particles
bh_particles=--bh_particles
gas_particles=--gas_particles
igrm_particles=--igrm_particles

calc_thermo_props=--calc_thermo_props
calc_metal_props=--calc_metal_props
calc_gradients=--no-calc_gradients
calc_log_props=--calc_log_props

# halo_ids=0
# Read halo_ids from file
# halo_file=$caesar_dir/halo_progen_info_snap_145
# while IFS=$'\t' read -r -a line; do
#     # source_snap_nums+="${line[1]} "
#     halo_ids+="${line[8]} "
# done < $halo_file


echo $snap_file
echo $caesar_file


echo
echo 'CALCULATING PROFILES'
echo


ahf_halo_ids_array=($ahf_halo_ids)
halo_dirs_array=($halo_dirs)

for i in "${!ahf_halo_ids_array[@]}"; do
	ahf_halo_id="${ahf_halo_ids_array[$i]}"
	halo_dir="${halo_dirs_array[$i]}"
	echo "Processing halo: $halo_dir with id: $ahf_halo_id"


    snap_dir=$top_dir/$halo_dir
    caesar_dir=$snap_dir/$caesar_dir_type
    profiles_dir=$caesar_dir/$profiles_dir_type
    save_file="$sim"-"$snap_file"-"$ndim"d_"${filter,,}"_profiles-xscale_"${xscale,,}"-temp_cut"=$temp_cut"-nh_cut"=$nh_cut"-rho_cut"=$rho_cut"rho_crit
    echo $save_file
    echo


    # Read target_snap_nums and target_halo_ids from file
    halo_info_file=$caesar_dir/halo_info_snap_${snap_nums}-halo_file
    while IFS=$'\t' read -r -a line; do
        # target_snap_nums_list+="${line[6]} "
        snap_nums_list+="${line[1]} "
        halo_ids_list+="${line[3]} "
        # target_halo_ids_list+="${line[9]} "
    done < $halo_info_file
    # target_snap_nums=$(seq 53 1 272)
    # target_halo_ids=
    # target_snap_nums=$SLURM_ARRAY_TASK_ID

    ## Get specific halo_id
    # Convert space-separated lists into arrays
    IFS=' ' read -r -a snap_nums_array <<< "$snap_nums_list"
    IFS=' ' read -r -a halo_ids_array <<< "$halo_ids_list"

    # Find index of target_snap_num in halo_ids_array
    for i in "${!snap_nums_array[@]}"; do
    if [[ "${snap_nums_array[$i]}" = "${snap_nums}" ]]; then
        halo_ids="${halo_ids_array[$i]}"
        break
    fi
    done


    echo halo_ids:
    echo $halo_ids
    echo


    python gen_nd_profile_by_halo_id_v2.py --code=$code --snap_file=$snap_dir/$snap_file.hdf5 --caesar_file=$caesar_dir/$caesar_file.hdf5 --halo_ids $halo_ids --save_file=$profiles_dir/"$save_file" --filter=$filter --profile_type=$profile_type --ndim=$ndim --weight_by=$weight_by --xscale=$xscale --temp_cut="$temp_cut" --nh_cut="$nh_cut" --rho_cut=$rho_cut $halo_particles $dm_particles $bh_particles $gas_particles $igrm_particles

    echo
    echo

    echo
    echo 'CALCULATING EXTRA PROPERTIES OF PROFILES'
    echo

    python calc_profile_properties.py --dir=$profiles_dir --name="$save_file" --caesar_file=$caesar_dir/$caesar_file.hdf5 --suffix=$suffix --code=$code --ndim=$ndim $calc_thermo_props $calc_metal_props $calc_gradients $calc_log_props

    echo
    echo
    echo "Finished processing halo: $halo_dir with id: $ahf_halo_id"

done


echo
echo 'done'

########################################################