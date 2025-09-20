#!/bin/bash

INPUT_DIR="/home/msantesteban/tesis/quantum_embeddings/input_exps"

for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
      sbatch ./job_script_grid_search.sh "$file"
    fi
done

