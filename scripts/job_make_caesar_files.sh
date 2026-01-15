#!/bin/bash -l
#########################################################
#SBATCH -J make_caesar_files

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_642-correct_jet-fedd_0.02/caesar_snap/slurm_files/slurm-%j.out
#########################################################
##SBATCH --time=2:50:00
#SBATCH --time=24:00:00
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

snap_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_642-correct_jet-fedd_0.02/
caesar_dir=${snap_dir}caesar_snap/
#caesar_dir=/scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L0_Calibration/halo_3224_good-trillium/caesar_snap/
echo $caesar_dir

python make_caesar_files.py --snap_dir=$snap_dir --snap_nums {151..146} --snap_base=snapshot_ --caesar_dir=$caesar_dir --haloid=snap --blackholes --aperture=30 --half_stellar_radius_property --nproc=192 --lowres=2

########################################################
