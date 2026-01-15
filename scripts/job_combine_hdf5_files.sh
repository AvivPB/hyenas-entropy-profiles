#!/bin/bash -l
#########################################################
#SBATCH -J combine_hdf5_files

#SBATCH -o /scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/slurm_files/snap_props/v2/slurm-%j-combine.out

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#########################################################
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=0
#########################################################

##SBATCH --array=53-272 # Run a N-job array, all simultaneously

#########################################################

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


input_dir="/scratch/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/snap_props/v2"
output_file="${input_dir}/halo_3_props-v2.hdf5"
pattern="halo_3_props-snap_*-v2.hdf5"

echo "Combining HDF5 files..."
python combine_hdf5_files.py "${input_dir}/${pattern}" "${output_file}"

if [ $? -eq 0 ]; then
    echo "Successfully combined files into ${output_file}"
else
    echo "Error combining files"
    exit 1
fi
