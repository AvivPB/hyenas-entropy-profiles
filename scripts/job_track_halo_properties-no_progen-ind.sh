#!/bin/bash -l
#########################################################
#SBATCH -J track_halo_properties-no_progen-ind
    
#SBATCH -o /scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/slurm_files/snap_props/v2/slurm-%j.out

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#########################################################
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=0
#########################################################

#SBATCH --array=2-151 # Run a N-job array, all simultaneously

#########################################################


# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "Job task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo ""
# ---------------------------------------------------------------------


module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


snap_dir=/project/rrg-babul-ad/wcui/HYENAS/Level1/halo_3224
# /scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good
snap_base=snap_halo_3224_
# snapshot_
caesar_dir=/project/rrg-babul-ad/wcui/HYENAS/Level1/halo_3224/Groups
# /scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good/caesar_snap/
caesar_base=Caesar_halo_3224_
# caesar_
caesar_suffix=''

# Read target_snap_nums and target_halo_ids from file
progen_file=/scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/halo_3_snap_151_progen_info
while IFS=$'\t' read -r -a line; do
    target_snap_nums_list+="${line[6]} "
    # target_snap_nums_list+="${line[7]} "
    target_halo_ids_list+="${line[8]} "
    # target_halo_ids_list+="${line[9]} "
done < $progen_file
# target_snap_nums=$(seq 53 1 272)
# target_halo_ids=
target_snap_nums=$SLURM_ARRAY_TASK_ID

## Get specific halo_id
# Convert space-separated lists into arrays
IFS=' ' read -r -a snap_nums_array <<< "$target_snap_nums_list"
IFS=' ' read -r -a halo_ids_array <<< "$target_halo_ids_list"

# Find index of SLURM_ARRAY_TASK_ID in snap_nums_array
for i in "${!snap_nums_array[@]}"; do
   if [[ "${snap_nums_array[$i]}" = "${SLURM_ARRAY_TASK_ID}" ]]; then
       target_halo_ids="${halo_ids_array[$i]}"
       break
   fi
done



sim_model='Simba'

nproc=192
output_file=/scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/snap_props/v2/halo_3_props-snap_${SLURM_ARRAY_TASK_ID}-v2
clear_output_file=--clear_output_file


echo target_snap_nums:
echo $target_snap_nums
echo

echo target_halo_ids:
echo $target_halo_ids
echo


echo
echo 'CALCULATING HALO PROPERTIES'
echo

python track_halo_properties-hdf5-no_progen.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --target_snap_nums $target_snap_nums --target_halo_ids $target_halo_ids --sim_model=$sim_model --nproc=$nproc --output_file=$output_file.hdf5 $clear_output_file #> $output_file.out

echo
echo 'done'

########################################################
