from qiskit import QuantumCircuit
from .ansatz_library.fully_entangled_ansatz import fully_entangled_ansatz
from .ansatz_library.ansatz_ma import MA

from .losses.signed_MSE import signed_MSELoss
from .exp_types import (GraphEmbeddingsResult,
                    GraphEmbeddingsMetadata,
                    GraphEmbeddingContext,
                    Hiperparameters,
                    ClassicalHiperparameters, 
                    QuantumHiperparameters,
                    EarlyStopperParameters,
                    Experiment,
                    GridSearchAttrs,
                    CGridSearchAttrs,
                    QGridSearchAttrs,
                    TrainedModel)

from .quantum_knowledge_graph_embeddings.overlap_estimator import (SwapTestFidelityEstimator,
                                                                  QuantumForkingInnerProductRealPartEstimator,
                                                                  ComputeUncomputeFidelityEstimator)
from .quantum_knowledge_graph_embeddings.backends.pytorch_backend import PyTorchExpectedValueBackendBuilder
from .evaluators.link_prediction import custom_link_prediction
from .quantum_knowledge_graph_embeddings.model import QuantumVariationalModel
from .random_model.random_interaction import RandomInteraction

import pykeen.models as models
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.trackers import CSVResultTracker
from pykeen.losses import MarginRankingLoss, MSELoss
from pykeen.stoppers import EarlyStopper
from pykeen.models import make_model_cls

import mlflow
from mlflow import MlflowClient

from .datasets.dataset_handler import dataset_handler

import os
import torch
import numpy as np
import pickle
import pandas as pd

from itertools import product
import random

device = "cuda" if torch.cuda.is_available() else "cpu" 
supported_quantum_models = {"ComputeUncompute" :  ComputeUncomputeFidelityEstimator,
                            "RealInnerProduct" : QuantumForkingInnerProductRealPartEstimator,
                            "SwapTest" : SwapTestFidelityEstimator}

def set_global_seed(seed: int):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed_all(seed)     
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False   

class GridSearchRunner:

    def __init__(self, experimento : Experiment, full_train : bool = False, ui : bool = False):

        self.models_directory = experimento.output_dir

        self.hiperparameters = self.expand_grid(experimento.grid_search_attrs)
        self.early_stopper_parameters = experimento.early_stopper_params
        self.full_train = full_train
        self.max_epochs = experimento.max_epochs

        self.experiment_name = experimento.output_dir.split('/')[-1]

        self.training_set = None
        self.evaluation_set = None
        self.testing_set = None

        self.initializer_high = experimento.initializer_high
        self.initializer_low = experimento.initializer_low

        self.ui = ui

    @staticmethod
    def expand_grid(attrs: GridSearchAttrs) -> list[Hiperparameters]:
        attrs_dict = attrs.model_dump()

        keys = attrs_dict.keys()
        values = attrs_dict.values()
        combos = product(*values)

        hips_class = ClassicalHiperparameters if isinstance(attrs, CGridSearchAttrs) else QuantumHiperparameters

        valid_combos = []
        for combo in combos:
            params = dict(zip(keys, combo))
            if params["loss_function"] == "MSE":
                params["margin"] = None
            if hips_class == QuantumHiperparameters:
                params["U_ansatz"] = MA(params["num_qubits"]) if params["rotations"] == ["ma"] else fully_entangled_ansatz(params["num_qubits"], params["rotations"], params["controls"], comprimir_controladas=True)
                params["V_ansatz"] = params["U_ansatz"]
                params.pop("rotations",None)
                params.pop("controls",None)
            valid_combos.append(params)
        return [hips_class(**params) for params in valid_combos]

    def create_experiment(self):
        if os.system(f"curl -s http://127.0.0.1:8080/health") == 0: 
            print("MLFlow server online.")
            mlflow.set_tracking_uri("http://127.0.0.1:8080")
            client = MlflowClient(tracking_uri='http://127.0.0.1:8080')
        else: 
            print("MLFlow server offline.")
            mlflow.set_tracking_uri(f'./mlruns')
            client = MlflowClient(tracking_uri=f'mlruns')

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is not None and experiment.lifecycle_stage == "deleted":
                    client.restore_experiment(experiment.experiment_id)

        if experiment is None:
            
            mlflow.create_experiment(
                name=self.experiment_name,
                tags={
                    "dataset": self.models_directory.split('_')[0],
                    "initializer_low": str(self.initializer_high),
                    "initializer_high":str(self.initializer_low),
                    "mlflow.note.content": self.models_directory 
                }
            )

        mlflow.set_experiment(self.experiment_name)

    def get_triples(self, dataset : str):
        print(f"Partitioning Knowledge Graph into training, testing and validation dataset.")
        proportions = [1.0, 0.0, 0.0] if self.full_train else [0.8, 0.1, 0.1]
        self.training, self.evaluation, self.testing = dataset_handler(dataset, proportions=proportions)
        print(f"Knowledge Graph partitioned. There are {len(self.training.mapped_triples)} training triples, {len(self.testing.mapped_triples)} testing triples and {len(self.evaluation.mapped_triples)} validation triples.")

    def train_model(self, model, batch_size : int):

        training_loop = SLCWATrainingLoop(model=model, triples_factory = self.training)

        evaluator = RankBasedEvaluator()
        
        early_stopper = None
        early_stopper_info = None
        if not self.full_train: 
            early_stopper = EarlyStopper(model=model,
                                        evaluator=evaluator, 
                                        training_triples_factory=self.training,
                                        evaluation_triples_factory=self.evaluation,
                                        frequency=self.early_stopper_parameters.frequency, 
                                        patience=self.early_stopper_parameters.patience,
                                        relative_delta=self.early_stopper_parameters.delta, 
                                        metric=self.early_stopper_parameters.metric, 
                                        larger_is_better=True,
                                        use_tqdm=False)
            early_stopper_info = early_stopper.get_summary_dict()

        losses = training_loop.train(
            triples_factory=self.training,
            num_epochs=self.max_epochs,
            batch_size=batch_size,
            stopper=early_stopper,
            use_tqdm=True,
        )

        return losses, early_stopper_info

    def evaluate_model(self, model):
        evaluator = RankBasedEvaluator()
        results = evaluator.evaluate(
            model = model,
            mapped_triples=self.testing.mapped_triples, 
            additional_filter_triples=[self.training.mapped_triples, 
                                       self.evaluation.mapped_triples],
            device=torch.device(device)
        )

        link_prediction_metrics = custom_link_prediction(model, self.training, self.evaluation, self.testing)
        link_prediction_df = pd.DataFrame([
            {
                "Side": "relation",           
                "Rank_type": "standard",      
                "Metric": metric_name,
                "Value": value
            }
            for metric_name, value in link_prediction_metrics.items()
        ])

        evaluation_results_df = results.to_df()
        evaluation_results_df = pd.concat([evaluation_results_df, link_prediction_df], ignore_index=True)
        return evaluation_results_df

    @staticmethod
    def build_classical_model(model_name: str, loss : str, margin : float = None,   **kwargs):
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
    
    def build_quantum_model(self, overlap_estimator_name : str, num_qubits : int, U : QuantumCircuit, V : QuantumCircuit, loss : str, margin : float = None):
        overlap_estimator = supported_quantum_models[overlap_estimator_name]()
        backend_builder = PyTorchExpectedValueBackendBuilder()
        model = QuantumVariationalModel(num_qubits=num_qubits,
                                        backend_builder=backend_builder,
                                        overlap_estimator=overlap_estimator,
                                        entity_ansatz=U,
                                        relation_ansatz=V,
                                        triples_factory=self.training, 
                                        initializer_high = self.initializer_high,
                                        initializer_low = self.initializer_low)

        if loss == 'MSE':
            model.loss = signed_MSELoss(overlap_estimator=supported_quantum_models[overlap_estimator_name])
        else: 
            model.loss = MarginRankingLoss(margin = float(margin))

        return model

    def build_model(self, h : Hiperparameters):
        if isinstance(h, ClassicalHiperparameters):
            model = self.build_classical_model(h.model_name, loss = h.loss_function, margin=h.margin, embedding_dim=h.embedding_dim,triples_factory=self.training)
        else: 
            model = self.build_quantum_model(h.overlap_estimator, num_qubits=h.num_qubits, U=h.U_ansatz, V=h.V_ansatz, loss=h.loss_function, margin=h.margin)
        model.to(torch.device(device))
        return model

    def execute_single_model(self, h : Hiperparameters):
        print(f"Executing: {h}")

        model = self.build_model(h)

        losses, early_stopper_info = self.train_model(model, h.batch_size)        

        evaluation_result = None
        if not self.full_train:
            evaluation_result = self.evaluate_model(model)  

        trained_model = TrainedModel(
            state_dict = model.state_dict(),
            hiperparameters = h,
            evaluation_metrics = evaluation_result,
            loss_log=losses,
            early_stopper_info=early_stopper_info
        )

        return trained_model
                
    def save_model(self, model : TrainedModel):
        if not os.path.isdir(self.models_directory):
            os.mkdir(self.models_directory)

        model_name = '_'.join([
            f"{key[0]}{value}" for key, value in dict(model.hiperparameters).items()
            if key not in ("U_ansatz", "V_ansatz")
        ])
        model_directory = self.models_directory + '/' + model_name + '/'

        if not os.path.isdir(model_directory):
            os.mkdir(model_directory)

        torch.save(model.state_dict, model_directory + "model_state_dict.pt")

        with open(model_directory + "model_hiperparameters.pkl", "wb") as f:
            pickle.dump(model.hiperparameters, f)
        
        with open(model_directory + "model_triples_factory.pkl", "wb") as f:
            pickle.dump(self.training, f)
        
        model.evaluation_metrics.to_csv(model_directory + "evaluation_metrics.csv", index = False)
        loss_log_df = pd.DataFrame({"epoch" : range(1, len(model.loss_log) + 1), "loss_value" : model.loss_log})
        loss_log_df.to_csv(model_directory + "loss.csv", index = False)

        if hasattr(model.hiperparameters, 'U_ansatz'):
            model.hiperparameters.U_ansatz.draw('mpl', filename= model_directory + "/U.pdf")

        ## log model para mlflow

        with mlflow.start_run() as run:
            #params
            mlflow.log_params(model.hiperparameters.model_dump())

            #loss
            for epoch, loss in enumerate(model.loss_log, start=1):
                mlflow.log_metric("loss", loss, step=epoch)
            
            #early_stopper
            for evaluation_step, value in enumerate(model.early_stopper_info["results"], start=1):
                mlflow.log_metric(model.early_stopper_info["metric"], value, step=evaluation_step * int(model.early_stopper_info["frequency"]))

            #model_dict, evaluation, triples_factory & hiperarameters
            mlflow.log_artifact(model_directory + "model_state_dict.pt")
            mlflow.log_artifact(model_directory + "evaluation_metrics.csv")
            mlflow.log_artifact(model_directory + "model_triples_factory.pkl")
            mlflow.log_artifact(model_directory + "model_hiperparameters.pkl")


    def save_experiment_result(self, res : GraphEmbeddingsResult):
        res_dict = {
            "output": res.output,
            "local_context": res.local_context.model_dump() if hasattr(res.local_context, "model_dump") else vars(res.local_context),
            "metadata": res.metadata.model_dump() if hasattr(res.metadata, "model_dump") else vars(res.metadata),
        }

        pickle_path = self.models_directory + '/experiment_metadata.pkl'
        with open(pickle_path, "wb") as f:
            pickle.dump(res_dict, f)

    def evaluate_random_model(self):
        RandomModel = make_model_cls(
            interaction = RandomInteraction,
            interaction_kwargs={},
            dimensions=50
        )
        random_model = RandomModel(triples_factory=self.training)

        evaluation_result = self.evaluate_model(random_model)

        model_directory = self.models_directory + '/RandomModel'
        if not os.path.isdir(model_directory) : 
            os.mkdir(model_directory)
        
        evaluation_result.to_csv(model_directory + '/evaluation_metrics.csv')
        
    def execute(self, dataset : str) -> GraphEmbeddingsResult:
        seed = random.randint(1,10000000)
        set_global_seed(seed)

        self.create_experiment()

        self.get_triples(dataset)

        trained_models = []
        for hiperparameters in self.hiperparameters:
            try:  
                trained_model = self.execute_single_model(hiperparameters) 
                self.save_model(trained_model)
                trained_models.append(trained_model)
            except Exception as e:
                print(f"\033[31mError ejecutando modelo\033[0m : {e}\n Hiperparametros: {hiperparameters}")
                continue

        try: 
            self.evaluate_random_model()
        except Exception as e:
                print(f"\033[31mError ejecutando modelo random.\033[0m : {e}")
                

        context = GraphEmbeddingContext(
            dataset = dataset,
        )
        
        metadata = GraphEmbeddingsMetadata(
            hiperparameters = self.hiperparameters,
            early_stopper_parameters = self.early_stopper_parameters,
            training = self.training,
            testing = self.testing,
            evaluation = self.evaluation, 
            seed = seed,
            initializer_high = self.initializer_high,
            initializer_low = self.initializer_low
        )

        res =  GraphEmbeddingsResult(
            output = trained_models,
            local_context = context,
            metadata = metadata,
        )

        self.save_experiment_result(res)

        return res
