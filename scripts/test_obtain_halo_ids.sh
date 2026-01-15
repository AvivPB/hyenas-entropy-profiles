## Find halo id corresponding to this SLURM_ARRAY_TASK_ID
# top_dir=/scratch/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration
top_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration
echo $(ls $top_dir/)
echo
# ahf_halo_ids=$(ls $top_dir/ | grep -vE 'OG|slurm_files' | cut -d "_" -f2)
ahf_halo_ids=$(ls $top_dir/ | cut -d "_" -f2 | cut -d "-" -f1)
echo "Available halo_ids:"
echo $ahf_halo_ids
echo

index=1
ahf_halo_id=$(echo $ahf_halo_ids | cut -d " " -f $((index)))
echo "Processing halo_id: $ahf_halo_id"
echo

halo_dir=$(ls $top_dir/ | awk "NR==${index}")
echo "Processing halo: $halo_dir"
