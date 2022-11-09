#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # 2 compute nodes
#SBATCH --ntasks-per-node=1    # 1 mpi process each node
#SBATCH --mem=2000            # In megabytes
#SBATCH --array=0-29 # Array job indexes
#SBATCH --job-name="explode_count_array"
#SBATCH --cpus-per-task=15 # Number of physical cores
#SBATCH --ntasks-per-core=2 # Hyperthreading -> 2 threads for each physical core
#SBATCH --output=explode_count_array.out
#SBATCH --mail-user=erlenoi@stud.ntnu.no
#SBATCH --mail-type=ALL
    

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "Total of $SLURM_NTASKS cores"

module purge
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
python array_explode_count.py $SLURM_ARRAY_TASK_ID
