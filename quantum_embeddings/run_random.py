from pykeen.nn import Interaction
from pykeen.pipeline import pipeline
from pykeen.models import make_model_cls
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import Nations

from quantum_embeddings.random_model.random_interaction import RandomInteraction
from quantum_embeddings.evaluators.link_prediction import custom_link_prediction
from quantum_embeddings.datasets.dataset_handler import dataset_handler

import argparse
import pandas as pd
import os

def main(dataset : str):
    RandomModel = make_model_cls(
        interaction = RandomInteraction,
        interaction_kwargs={},
        dimensions=50 ## la dimension no te importa, sigue siendo igual de random
    )

    model = RandomModel(triples_factory=training)

    evaluator = RankBasedEvaluator()
    results = evaluator.evaluate(
        model = model,
        mapped_triples=testing.mapped_triples, 
        additional_filter_triples=[training.mapped_triples, 
                                    validation.mapped_triples],
    )

    link_prediction_metrics = custom_link_prediction(model, training, testing, validation)
    link_prediction_df = pd.DataFrame([
        {
            "Side": "relation",           
            "Rank_type": "standard",      
            "Metric": metric_name,
            "Value": value
        }
        for metric_name, value in link_prediction_metrics.items()
        ])

    directory = f'./results/Random_{dataset}/'

    if not os.path.isdir(directory):
        os.mkdir(directory)

    evaluation_results_df = results.to_df()
    evaluation_results_df = pd.concat([evaluation_results_df, link_prediction_df], ignore_index=True)
    evaluation_results_df.to_csv(directory + 'evaluation_metrics.csv')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Dataset para correr random")
    parser.add_argument(
        "dataset",  
        type=str,
        help="Dataset para correr la clasificacion random"
    )
    args = parser.parse_args()
    dataset = args.dataset
    training, testing, validation = dataset_handler(dataset, proportions=[0.8,0.1,0.1])
    main(dataset)
    