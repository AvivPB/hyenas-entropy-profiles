#!/bin/bash -l
#########################################################
#SBATCH -J identify_halo_progenitors

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_1390-correct_jet/caesar_snap/slurm_files/slurm-%j.out
#########################################################
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=0
#########################################################

##SBATCH --array=1-10%1 # Run a N-job array, 1 job at a time
#########################################################


# ---------------------------------------------------------------------
#echo "Current working directory: `pwd`"
#echo "Starting run at: `date`"
# ---------------------------------------------------------------------
#echo ""
#echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
#echo "Job task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
#echo ""
# ---------------------------------------------------------------------


module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


snap_dir=/project/rrg-babul-ad/wcui/HYENAS/Level0/halo_3224
# /scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7
snap_base=snap_halo_3224_
# snapshot_

caesar_dir=/project/rrg-babul-ad/wcui/HYENAS/Level0/halo_3224/Groups
# /scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_snap
# echo $caesar_dir

caesar_base=Caesar_halo_3224_
# caesar_
caesar_suffix=''

source_snap_nums=151
# target_snap_nums=$(seq 50 1 272)
source_halo_ids=3

n_most=1
nproc=192

output_file=/scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_3224_weiguang/Groups/halo_${source_halo_ids}_snap_${source_snap_nums}_progen_info
# $caesar_dir/halo_${source_halo_ids}_snap_${source_snap_nums}_progen_info
clear_output_file=--clear_output_file


echo
echo 'Identifying Progenitors'
echo

python identify_halo_progenitors.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --source_snap_nums=$source_snap_nums --target_snap_nums {151..2} --source_halo_ids $source_halo_ids --n_most=$n_most --nproc=$nproc --output_file=$output_file $clear_output_file

echo
echo 'done'

########################################################
