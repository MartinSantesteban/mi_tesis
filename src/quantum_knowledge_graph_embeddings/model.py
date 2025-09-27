from collections import OrderedDict
from typing import Any
from pykeen.nn import Embedding
from pykeen.models import ERModel
from .backends.backend_builder import BaseBackendBuilder
from .interaction import QuantumVariationalInteraction
from .overlap_estimator import BaseOverlapEstimator
from .uniform_initializer import UniformInitializer
from pykeen.nn.modules import interaction_resolver
from qiskit import QuantumCircuit


class QuantumVariationalModel(ERModel):
    def __init__(
        self,
        num_qubits: int,
        backend_builder: BaseBackendBuilder,
        overlap_estimator: BaseOverlapEstimator,
        entity_ansatz: QuantumCircuit,
        relation_ansatz: QuantumCircuit,
        initializer_high: float = None,
        initializer_low: float = None,
        **kwargs,
    ) -> None:

        initializer = None
        if initializer_high and initializer_low:
            print("Setting initializer")
            initializer = UniformInitializer(low =initializer_low, high = initializer_high)

        super().__init__(
            interaction=QuantumVariationalInteraction,
            interaction_kwargs={
                "num_qubits": num_qubits,
                "backend_builder": backend_builder,
                "overlap_estimator": overlap_estimator,
                "entity_ansatz": entity_ansatz, 
                "relation_ansatz": relation_ansatz
            },
            entity_representations=Embedding,
            entity_representations_kwargs=dict(
                embedding_dim=entity_ansatz.num_parameters,
                initializer=initializer
            ),
            relation_representations=Embedding,
            relation_representations_kwargs=dict(
                embedding_dim=relation_ansatz.num_parameters,
                initializer=initializer
            ),
            **kwargs,
        )
        self.interaction_kwargs={
                "num_qubits": num_qubits,
                "backend_builder": backend_builder,
                "overlap_estimator": overlap_estimator,
                "entity_ansatz": entity_ansatz, 
                "relation_ansatz": relation_ansatz
            }
        
    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._modules["interaction"] = interaction_resolver.make(
            QuantumVariationalInteraction, pos_kwargs=self._interaction_kwargs
        )

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # Remove the sampler from the state dictionary to avoid pickling issues
        modules: OrderedDict = state["_modules"]
        modules.pop("interaction", None)
        return state
