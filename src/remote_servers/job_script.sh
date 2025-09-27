#!/bin/bash

#SBATCH --job-name=VQA_KG
#SBATCH --nodes=1
#SBATCH --chdir="/home/msantesteban/tesis"
#SBATCH --error="./cecar/VQA_KG_error-%j.err"
#SBATCH --output="./cecar/VQA_KG_output-%j.out"
#SBATCH --partition=mem
#SBATCH --time=71:58:00

echo "trabajo \"${SLURM_JOB_NAME}\""
echo "id: ${SLURM_JOB_ID}"
echo "particion: ${SLURM_JOB_PARTITION}"
echo "nodos: ${SLURM_JOB_NODELIST}"
date +"inicio %F - %T"

uv sync
uv run run_experiment.py ./input_exps/finredc0_ansatz1.json

date +"fin %F - %T"
