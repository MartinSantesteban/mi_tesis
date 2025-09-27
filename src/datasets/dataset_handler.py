from pykeen.triples.triples_factory import TriplesFactory
from pykeen.datasets import get_dataset
import os
import torch
import numpy as np

def dataset_handler(dataset_name: str, proportions: [float]):
    try:
        dataset = get_dataset(dataset=dataset_name)
        training, evaluation, testing = dataset.training, dataset.testing, dataset.validation
    except KeyError:
        dataset_dir = f"quantum_embeddings/datasets/{dataset_name}"
        print(os.path.isdir(dataset_dir))
        triples_path = dataset_dir + f"/tripletas_{dataset_name}.tsv"
        if os.path.isdir(dataset_dir) and os.path.isfile(triples_path):
            triples_factory = TriplesFactory.from_path(triples_path)
            training, evaluation, testing = triples_factory.split(proportions)
        else:
            raise ValueError(f"Unknown dataset name or missing files: {dataset_name}")
    assert(sanity_check(training, evaluation, testing))
    return training, evaluation, testing
    

def sanity_check(training: torch.tensor, testing: torch.tensor, validation: torch.tensor):
    train_triples = training.mapped_triples
    test_triples = testing.mapped_triples
    valid_triples = validation.mapped_triples

    train_entities = set(train_triples[:, 0].tolist()) | set(train_triples[:, 2].tolist())
    test_entities = set(test_triples[:, 0].tolist()) | set(test_triples[:, 2].tolist())
    valid_entities = set(valid_triples[:, 0].tolist()) | set(valid_triples[:, 2].tolist())

    train_relations = set(train_triples[:, 1].tolist())
    test_relations = set(test_triples[:, 1].tolist())
    valid_relations = set(valid_triples[:, 1].tolist())

    dif_entities = len(test_entities - train_entities) + len(valid_entities - train_entities)
    dif_relations = len(test_relations - train_relations) + len(valid_relations - train_relations)
    return dif_relations == 0 and dif_entities == 0


if __name__ == '__main__':
    a,b,c = dataset_handler("FinRED")

    