#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=32:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1              # 2 compute nodes
#SBATCH --ntasks-per-node=1    # 1 mpi process each node
#SBATCH --mem=48000            # 12GB - in megabytes
#SBATCH --job-name="m-m-lam"
#SBATCH --cpus-per-task=1 # Number of physical cores
#SBATCH --ntasks-per-core=2 # Hyperthreading -> 2 threads for each physical core
#SBATCH --output=abl_mooc_rss_lambda-man.out


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
cd aprec/evaluation
CUDA_VISIBLE_DEVICES=0 sh run_n_experiments.sh configs/MOOC_rss_abl_lambda-man.py


