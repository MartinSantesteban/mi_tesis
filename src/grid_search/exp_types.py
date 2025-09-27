from pydantic import BaseModel
from typing import List, Optional, OrderedDict, Union, Mapping, Any
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.evaluation.rank_based_evaluator import RankBasedMetricResults
from qiskit import QuantumCircuit
import pandas as pd

class EarlyStopperParameters(BaseModel):
    frequency : int
    patience : int
    delta : float
    metric : str

class Hiperparameters(BaseModel):
    batch_size : Optional[int]
    loss_function : Optional[str]
    margin : Optional[float]

class ClassicalHiperparameters(Hiperparameters):
    model_name : str
    embedding_dim : int

class QuantumHiperparameters(Hiperparameters):
    class Config: arbitrary_types_allowed = True
    num_qubits : int
    overlap_estimator : str
    U_ansatz : QuantumCircuit
    V_ansatz : QuantumCircuit

class TrainedModel(BaseModel):
    class Config: arbitrary_types_allowed = True
    state_dict : OrderedDict
    hiperparameters : Hiperparameters
    evaluation_metrics : pd.DataFrame 
    loss_log : List[float]
    early_stopper_info : Mapping[str, Any]

class GraphEmbeddingContext(BaseModel):
    class Config: arbitrary_types_allowed = True
    dataset : str

class GraphEmbeddingsMetadata(BaseModel):
    class Config: arbitrary_types_allowed = True
    hiperparameters : List[Hiperparameters]
    early_stopper_parameters : EarlyStopperParameters
    training : TriplesFactory
    evaluation : TriplesFactory
    testing : Optional[TriplesFactory]
    seed : int
    initializer_high: Optional[float]
    initializer_low: Optional[float]

class GraphEmbeddingsResult(BaseModel):
    output : List[TrainedModel]
    metadata : GraphEmbeddingsMetadata
    local_context : GraphEmbeddingContext
    
    
#-------------------------------------------------------

class GridSearchAttrs(BaseModel):
    loss_function : List[str]
    margin : List[Optional[float]]
    batch_size : List[int]

class QGridSearchAttrs(GridSearchAttrs):
    num_qubits: List[int]
    overlap_estimator: List[str]
    rotations: List[List[str]]
    controls: List[List[str]]

class CGridSearchAttrs(GridSearchAttrs):
    model_name: List[str]
    embedding_dim: List[int]

class Experiment(BaseModel):
    dataset : str
    early_stopper_params : EarlyStopperParameters
    grid_search_attrs : Union[QGridSearchAttrs, CGridSearchAttrs]
    max_epochs : int
    output_dir : str
    initializer_low : Optional[float] = None
    initializer_high : Optional[float] = None

    