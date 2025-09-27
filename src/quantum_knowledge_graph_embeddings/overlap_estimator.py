from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from torch import Tensor
import itertools


class BaseOverlapEstimator(ABC):
    @abstractmethod
    def get_circuit(
        self,
        num_qubits: int,
        h_ansatz: QuantumCircuit,
        r_ansatz: QuantumCircuit,
        t_ansatz: QuantumCircuit,
        measure: bool = True,
    ) -> QuantumCircuit:
        pass

    @abstractmethod
    def get_score_observable(self, num_qubits: int) -> SparsePauliOp:
        pass

    @abstractmethod
    def test_measurement(self, measurement: int) -> bool:
        pass



class SwapTestFidelityEstimator(BaseOverlapEstimator):
    def get_circuit(
        self,
        num_qubits: int,
        h_ansatz: QuantumCircuit,
        r_ansatz: QuantumCircuit,
        t_ansatz: QuantumCircuit,
        measure: bool = True,
    ):
        swap_test_circuit = QuantumCircuit(2 * num_qubits + 1, 1)
        swap_test_circuit.compose(
            h_ansatz, qubits=[1 + i for i in range(num_qubits)], inplace=True
        )
        swap_test_circuit.compose(
            r_ansatz, qubits=[1 + i for i in range(num_qubits)], inplace=True
        )
        swap_test_circuit.compose(
            t_ansatz,
            qubits=[1 + i + num_qubits for i in range(num_qubits)],
            inplace=True,
        )
        swap_test_circuit.h(0)
        for i in range(num_qubits):
            swap_test_circuit.cswap(0, i + 1, i + num_qubits + 1)
        swap_test_circuit.h(0)

        if measure:
            swap_test_circuit.measure([0], [0])
        return swap_test_circuit
    
    def get_score_observable(self, num_qubits: int) -> SparsePauliOp:
        return SparsePauliOp.from_list(
            [
                ("I" * (num_qubits *2) + "I", 1 / 2),
                ("I" * (num_qubits *2) + "Z", 1 / 2),
            ]
        )
    def test_measurement(self, measurement: int) -> bool:
        return measurement % 2 == 0

class QuantumForkingInnerProductRealPartEstimator(BaseOverlapEstimator):
    def get_circuit(
        self,
        num_qubits: int,
        h_ansatz: QuantumCircuit,
        r_ansatz: QuantumCircuit,
        t_ansatz: QuantumCircuit,
        measure: bool = True,
    ):
        qc = QuantumCircuit(num_qubits + 1, 1)
        qc.h(0)
        qc.compose(
            h_ansatz.control(ctrl_state=1),
            qubits=list(range(num_qubits + 1)),
            inplace=True,
        )
        qc.compose(
            r_ansatz.control(ctrl_state=1),
            qubits=list(range(num_qubits + 1)),
            inplace=True,
        )
        qc.compose(
            t_ansatz.control(ctrl_state=0),
            qubits=list(range(num_qubits + 1)),
            inplace=True,
        )
        qc.h(0)
        if measure:
            qc.measure([0], [0])
        return qc

    def get_score_observable(self, num_qubits: int) -> SparsePauliOp:
        return SparsePauliOp.from_list(
            [
                ("I" * (num_qubits) + "I", 1 / 2),
                ("I" * (num_qubits) + "Z", 1 / 2),
            ]
        )
    
    def test_measurement(self, measurement: int) -> bool:
        return measurement % 2 == 0


class ComputeUncomputeFidelityEstimator(BaseOverlapEstimator):
    def get_circuit(
        self,
        num_qubits: int,
        h_ansatz: QuantumCircuit,
        r_ansatz: QuantumCircuit,
        t_ansatz: QuantumCircuit,
        measure: bool = True,
    ):
        qc = QuantumCircuit(num_qubits)
        qc.compose(h_ansatz, qubits=list(range(num_qubits)), inplace=True)
        qc.compose(r_ansatz, qubits=list(range(num_qubits)), inplace=True)
        qc.compose(
            t_ansatz.inverse(annotated=True),
            qubits=list(range(num_qubits)),
            inplace=True,
        )
        if measure:
            qc.measure_all()
        return qc

    def get_score_observable(self, num_qubits: int) -> SparsePauliOp:
        return SparsePauliOp.from_list(
            [
                ("".join(t), 1 / (2**num_qubits))
                for t in itertools.product("ZI", repeat=num_qubits)
            ]
        )

    def test_measurement(self, measurement: int) -> bool:
        return measurement == 0