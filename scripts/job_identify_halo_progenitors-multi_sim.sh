#!/bin/bash -l
#########################################################
#SBATCH -J identify_halo_progenitors

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/slurm_files/slurm-%j.out
#########################################################
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=0
#########################################################

#SBATCH --array=1,7,8,9,10,11,12,13,27,43,44,45,46,47,48,49,50,51,52,53,54,55,56,64,71,72,73,76,77,78,84,85,86,89,90 # Run a N-job array, all simultaneously
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



## Find directory corresponding to this SLURM_ARRAY_TASK_ID
# dir=/scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration
top_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration
echo $top_dir
echo

# halo_dir=$(ls $top_dir/ | cut -d " " -f ${SLURM_ARRAY_TASK_ID})
halo_dir=$(ls $top_dir/ | awk "NR==${SLURM_ARRAY_TASK_ID}")
echo "Processing halo: $halo_dir"
echo

#halo_ids=$(ls -d /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_* | grep -vE '3224|OG')
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/ | grep -vE '3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls /scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/ | grep -vE 'aviv_halo_info|3224|OG' | cut -d "_" -f2)
# halo_ids=$(ls $dir/ | grep -vE 'OG|slurm_files' | cut -d "_" -f2)
# halo_ids=3224
ahf_halo_id=$(ls $top_dir/ | cut -d " " -f ${SLURM_ARRAY_TASK_ID} | cut -d "_" -f2 | cut -d "-" -f1)
# echo "Processing halo_id: $ahf_halo_id"




snap_dir=$top_dir/$halo_dir
# /scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good
snap_base=snapshot_
# snap_halo_${ahf_halo_id}_
caesar_dir=$top_dir/$halo_dir/caesar_snap
# /scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good/caesar_snap/
caesar_base=caesar_
# Caesar_halo_${ahf_halo_id}_
caesar_suffix=''



source_snap_nums=151
# target_snap_nums=$(seq 50 1 272)
# source_halo_ids=3


# Read target_snap_nums and target_halo_ids from file
halo_info_file=$caesar_dir/halo_info_snap_${source_snap_nums}-halo_file
while IFS=$'\t' read -r -a line; do
    # target_snap_nums_list+="${line[6]} "
    source_snap_nums_list+="${line[1]} "
    source_halo_ids_list+="${line[3]} "
    # target_halo_ids_list+="${line[9]} "
done < $halo_info_file
# target_snap_nums=$(seq 53 1 272)
# target_halo_ids=
# target_snap_nums=$SLURM_ARRAY_TASK_ID

## Get specific halo_id
# Convert space-separated lists into arrays
IFS=' ' read -r -a snap_nums_array <<< "$source_snap_nums_list"
IFS=' ' read -r -a halo_ids_array <<< "$source_halo_ids_list"

# Find index of target_snap_num in halo_ids_array
for i in "${!snap_nums_array[@]}"; do
   if [[ "${snap_nums_array[$i]}" = "${source_snap_nums}" ]]; then
       source_halo_ids="${halo_ids_array[$i]}"
       break
   fi
done



n_most=1
nproc=192

output_file=$caesar_dir/halo_${source_halo_ids}_snap_${source_snap_nums}_progen_info
# $caesar_dir/halo_${source_halo_ids}_snap_${source_snap_nums}_progen_info
clear_output_file=--clear_output_file



echo
echo 'Identifying Progenitors'
echo

python identify_halo_progenitors.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --source_snap_nums=$source_snap_nums --target_snap_nums {151..0} --source_halo_ids $source_halo_ids --n_most=$n_most --nproc=$nproc --output_file=$output_file $clear_output_file

echo
echo 'done'

########################################################
