from turtle import forward
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from torch import embedding, nn
import torch
from .interaction import QuantumVariationalInteraction
from math import log2, ceil
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from abc import ABC, abstractmethod


class QuantumLoss(nn.Module, ABC):
    def __init__(
        self,
        interaction: QuantumVariationalInteraction,
        estimator: BaseEstimatorV2,
        backend,
        nentities: int,
        nrelations: int,
    ):
        super().__init__()
        self.interaction = interaction
        self.estimator = estimator
        self.backend = backend
        embedding_dim = len(interaction.get_circuit().parameters) // 3
        #TODO: ver pykeen.nn.Embedding
        self.entity_embeddings = nn.Embedding(nentities, embedding_dim)
        self.relation_embeddings = nn.Embedding(nrelations, embedding_dim)

    def forward(
        self,
        h_idx: NDArray[np.float64],
        r_idx: NDArray[np.float64],
        t_idx: NDArray[np.float64],
        y: List[int],
    ):
        qc = self.get_quantum_loss_circuit(y)
        observable = self.get_observable(len(y))
        model = self.get_model(qc, observable)
        circuit_parameters = self.get_circuit_parameters(h_idx, r_idx, t_idx)
        return model(circuit_parameters)

    def get_model(self, qc, observable):
        qnn = EstimatorQNN(
            circuit=qc,
            estimator=self.estimator,
            input_params=qc.parameters,
            weight_params=[],
            observables=observable,
            input_gradients=True,
            pass_manager=generate_preset_pass_manager(
                backend=self.backend, optimization_level=3
            ),
        )
        model = TorchConnector(qnn)
        return model

    @abstractmethod
    def get_observable(self, num_triples: int) -> SparsePauliOp:
        pass

    def get_circuit_parameters(self, h_idx, r_idx, t_idx):
        h_representations = self.entity_embeddings(h_idx)
        r_representations = self.relation_embeddings(r_idx)
        t_representations = self.entity_embeddings(t_idx)
        circuit_parameters = torch.cat(
            [h_representations, r_representations, t_representations], dim=1
        )
        
        return circuit_parameters.reshape(-1)

    def get_entity_ansatz_circuit(self, i: int) -> QuantumCircuit:
        return self.interaction.get_circuit(measure=False, prefix="X" + str(i))

    @abstractmethod
    def get_quantum_loss_circuit(self, y: list) -> QuantumCircuit:
        pass


class StableQuantumLoss(QuantumLoss):
    def get_quantum_loss_circuit(self, y: list) -> QuantumCircuit:
        num_qubits = self.interaction.num_qubits
        entities = QuantumRegister(num_qubits, name="entities")
        indexes = QuantumRegister(ceil(log2(len(y))), name="indexes")
        labels = QuantumRegister(1, name="labels")

        qc = QuantumCircuit(entities, indexes, labels)

        qc.h(indexes)

        for i, label in enumerate(y):
            binary_idx = bin(i)[2:].zfill(len(indexes))
            for j, bit in enumerate(binary_idx):
                if bit == "1":
                    qc.x(indexes[j])
            qc.append(
                self.get_entity_ansatz_circuit(i).control(len(indexes), annotated=True),
                [*indexes, *entities],
            )
            if label == 1:
                qc.mcx([*indexes], labels[0])

            for j, bit in enumerate(binary_idx):
                if bit == "1":
                    qc.x(indexes[j])

        return qc
    
    def get_observable(self, num_triples: int) -> SparsePauliOp:
        n = self.interaction.num_qubits
        m = ceil(log2(num_triples))
        return SparsePauliOp.from_list(
            [("Z" + "I" * m + "I" * n, 1 / 2), ("Z" + "I" * m + "Z" * n, 1 / 2)]
        )

class EfficientQuantumLoss(QuantumLoss):
    def get_quantum_loss_circuit(self, y: list) -> QuantumCircuit:
        num_qubits = self.interaction.num_qubits
        entities = QuantumRegister(num_qubits, name="entities")
        discriminator = QuantumRegister(1, name="discriminator")
        labels = QuantumRegister(1, name="labels")

        qc = QuantumCircuit(entities, discriminator, labels)

        for i, label in enumerate(y):
            qc.rz(2 * np.pi * i / len(y), discriminator[0])
            qc.append(
                self.get_entity_ansatz_circuit(i).control(annotated=True),
                [discriminator[0], *entities],
            )
            if label == 1:
                qc.cx(discriminator[0], labels[0])

        return qc

    def get_observable(self, num_triples: int) -> SparsePauliOp:
        n = self.interaction.num_qubits
        return SparsePauliOp.from_list(
            [("ZI" + "I" * n, 1 / 2), ("ZI" + "Z" * n, 1 / 2)]
        )

""" def calculate_loss(
    interaction: QuantumVariationalInteraction,
    x: NDArray[np.float64],
    y: List[int],
    estimator: BaseEstimatorV2,
    backend,
) -> Tuple[float, NDArray[np.float64]]:


    gradient = ParamShiftEstimatorGradient(estimator)

    # Evaluate the gradient of the circuits using parameter shift gradients
    pse_grad_result = gradient.run([isa_circuit], [isa_observable], [circuit_parameters]).result()

    return -job.result()[0].data["evs"], np.array(pse_grad_result.gradients).reshape(x.shape)
 """
