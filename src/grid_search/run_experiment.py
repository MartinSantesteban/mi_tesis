from .grid_search_runner import GridSearchRunner
from .exp_types import Experiment

from pathlib import Path
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="JSON con los parametros para hacer grid search")
    parser.add_argument(
        "experiment_file",  
        type=str,
        help="JSON con los parametros para hacer grid search"
    )
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment_file)
    if not experiment_path.exists():
        raise FileNotFoundError(f"Archivo del experimento no encontrado: {experiment_path}")

    with open(experiment_path) as f:
        d = json.load(f)
        print(d)
    experimento = Experiment(**d)
    
    if not os.path.isdir(experimento.output_dir):
        os.makedirs(experimento.output_dir, exist_ok=True)

    runner = GridSearchRunner(experimento= experimento, full_train= False)
    
    runner.execute(experimento.dataset)

if __name__ == "__main__":
    main()

#mlflow server --host 127.0.0.1 --port 8080