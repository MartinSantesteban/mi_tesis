from pykeen.nn import Interaction
from .backends.backend_builder import (
    BaseBackendBuilder,
)
from .overlap_estimator import BaseOverlapEstimator
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class QuantumVariationalInteraction(Interaction):
    def __init__(
        self,
        num_qubits: int,
        backend_builder: BaseBackendBuilder,
        overlap_estimator: BaseOverlapEstimator,
        entity_ansatz: QuantumCircuit,
        relation_ansatz: QuantumCircuit
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.qc = self.get_circuit(
            overlap_estimator, entity_ansatz, relation_ansatz, measure=False
        )
        model = backend_builder.build_backend(
            num_qubits,
            overlap_estimator,
            self.qc,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)

    def forward(self, h, r, t):
        broadcasted_tensors = torch.broadcast_tensors(h, r, t)
        res = self.model(
            torch.cat(broadcasted_tensors, dim=-1).reshape(-1, len(self.qc.parameters))
        )
        shape = broadcasted_tensors[0].shape
        if len(shape) == 2:
            return res
        else:
            return res.reshape(shape[0], shape[1])

    def get_circuit(
        self,
        overlap_estimator: BaseOverlapEstimator,
        entity_ansatz: QuantumCircuit,
        relation_ansatz: QuantumCircuit,
        measure=True,
    ):
        vector = ParameterVector(
            "ϕ",
            length=entity_ansatz.num_parameters * 2 + relation_ansatz.num_parameters,
        )
        h_ansatz = reassign_parameter_names(entity_ansatz, vector, 0)
        r_ansatz = reassign_parameter_names(
            relation_ansatz, vector, h_ansatz.num_parameters
        )
        t_ansatz = reassign_parameter_names(
            entity_ansatz, vector, h_ansatz.num_parameters + r_ansatz.num_parameters
        )
        return overlap_estimator.get_circuit(
            self.num_qubits,
            h_ansatz,
            r_ansatz,
            t_ansatz,
            measure,
        )


def reassign_parameter_names(qc: QuantumCircuit, vector: ParameterVector, offset: int):
    return qc.assign_parameters(
        {p: vector[i + offset] for i, p in enumerate(qc.parameters)},
        inplace=False,
    )
