import torch
import pandas as pd

from pykeen.models import ERModel
import pickle
from quantum_embeddings import exp_types
from quantum_embeddings.exp_types import (Hiperparameters,
                                          ClassicalHiperparameters,
                                          QuantumHiperparameters)

from qiskit import QuantumCircuit

from pykeen.triples import TriplesFactory
from pykeen.losses import (MSELoss, MarginRankingLoss)
import pykeen.models as models

from quantum_embeddings.quantum_knowledge_graph_embeddings.model import QuantumVariationalModel
from quantum_embeddings.quantum_knowledge_graph_embeddings.overlap_estimator import (QuantumForkingInnerProductRealPartEstimator,
                                                                                     ComputeUncomputeFidelityEstimator,
                                                                                     SwapTestFidelityEstimator)
from quantum_embeddings.quantum_knowledge_graph_embeddings.backends.pytorch_backend import PyTorchExpectedValueBackendBuilder

import sys
sys.modules["exp_types"] = exp_types

supported_quantum_models = {"ComputeUncompute" :  ComputeUncomputeFidelityEstimator,
                            "RealInnerProduct" : QuantumForkingInnerProductRealPartEstimator,
                            "SwapTest" : SwapTestFidelityEstimator}

def build_classical_model(model_name: str, loss : str, margin : float = None, **kwargs):
    try:
        model_class = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"Modelo no registrado en Pykeen: {model_name}")

    model = model_class(**kwargs)

    if loss == 'MSE':
        model.loss = MSELoss()
    else: 
        model.loss = MarginRankingLoss(margin = float(margin))

    return model

def build_quantum_model(triples_factory,overlap_estimator_name : str, num_qubits : int, U : QuantumCircuit, V : QuantumCircuit, loss : str, margin: int):
    overlap_estimator = supported_quantum_models[overlap_estimator_name]()
    backend_builder = PyTorchExpectedValueBackendBuilder()
    model = QuantumVariationalModel(num_qubits=num_qubits,
                                    backend_builder=backend_builder,
                                    overlap_estimator=overlap_estimator,
                                    entity_ansatz=U,
                                    relation_ansatz=V,
                                    triples_factory=triples_factory)
    if loss == 'MSE':
        model.loss = MSELoss()
    else: 
        model.loss = MarginRankingLoss(margin = float(margin))

    return model

def build_model(state_dict, h : Hiperparameters, triples_factory : TriplesFactory):

    if isinstance(h, ClassicalHiperparameters):
        model = build_classical_model(h.model_name, loss = h.loss_function, margin=h.margin, embedding_dim=h.embedding_dim,triples_factory=triples_factory)
    else: 
        model = build_quantum_model(triples_factory, h.overlap_estimator, num_qubits=h.num_qubits, U=h.U_ansatz, V=h.V_ansatz, loss=h.loss_function, margin=h.margin)

    model.load_state_dict(state_dict)

    return model


def load_model(model_folder : str):
    state_dict_path = model_folder + "/model_state_dict.pt"
    hiperparameters_path = model_folder + "/model_hiperparameters.pkl" 
    triples_factory_path = model_folder + "/model_triples_factory.pkl"

    state_dict = torch.load(state_dict_path, map_location="cpu")
    
    with open(hiperparameters_path, "rb") as h:
        hiperparameters = pickle.load(h)
    
    with open(triples_factory_path, "rb") as h:
        triples_factory = pickle.load(h)

    model = build_model(state_dict, hiperparameters, triples_factory)

    return model, triples_factory

