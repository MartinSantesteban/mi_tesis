#!/bin/bash

#SBATCH --job-name=VQA_KG
#SBATCH --nodes=1
#SBATCH --chdir="/home/msantesteban/tesis"
#SBATCH --error="./quantum_embeddings/cecar/VQA_KG_error-%j.err"
#SBATCH --output="./quantum_embeddings/cecar/VQA_KG_output-%j.out"
#SBATCH --partition=rtx4070,rtx2080
#SBATCH --time=71:58:00

echo "trabajo \"${SLURM_JOB_NAME}\""
echo "id: ${SLURM_JOB_ID}"
echo "particion: ${SLURM_JOB_PARTITION}"
echo "nodos: ${SLURM_JOB_NODELIST}"

date +"inicio %F - %T"

uv sync
uv run -m quantum_embeddings.run_experiment $1 

date +"fin %F - %T"
