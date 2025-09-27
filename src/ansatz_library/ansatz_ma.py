from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate

def MA(num_qubits: int):
    import math
    qc = QuantumCircuit(num_qubits)
    for qubit in range(0, num_qubits):
        qc.h(qubit)
        qc.u(Parameter(f"theta_{qubit}"),
             Parameter(f"phi_{qubit}"),
             Parameter(f"lambda_{qubit}"),qubit)

    pares = []
    for qubit in range(0, int(num_qubits * math.ceil((num_qubits-1)/2))): # en lugar de n(n-1)/2 asi agregamos las compuertas que tienen ellos, que parecen inutiles
        layer = (qubit // num_qubits) + 1
        controlado = qubit % num_qubits
        control =    (qubit - layer) % num_qubits
        pares.append((control,controlado))

    for par in pares: 
        control, controlado = par
        layer = (controlado // num_qubits) + 1
        qc.append(UGate(Parameter(f"theta_{control}_{controlado}_{layer}"),
                        Parameter(f"phi_{control}_{controlado}_{layer}"),
                        Parameter(f"lambda_{control}_{controlado}_{layer}"))
                        .control(1),[control, controlado])
    return qc

if __name__ == "__main__":
    for num_qubits in range(2,9):
        try: 
            qc = MA(num_qubits)
        except:
            print(f"Fallo {num_qubits}.")
            continue
    MA(6).draw("mpl", filename="ma.pdf")
    exit(0)