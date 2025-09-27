from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate

def comprimir(pares):
    res = []
    usados = [False for i in range(0, len(pares))]
    while not all(usados): 
        #busco al primero sin usar
        current_index = 0
        while usados[current_index] :
            current_index = current_index + 1
        current_end = pares[current_index][1]
        res.append(pares[current_index])
        usados[current_index] = True
        for tmp_idx in range(current_index, len(pares)):
            if not usados[tmp_idx]:
                tmp_par = pares[tmp_idx]
                if current_end < tmp_par[0]:
                    res.append(tmp_par)
                    current_end = tmp_par[1]
                    usados[tmp_idx] = True
    return res

single_qubit_rotations = {"rx" : (lambda qc, qubit, layer, i : qc.rx(Parameter(f"x_{qubit}_{layer}_{i}"), qubit)),
                          "ry" : (lambda qc, qubit, layer, i : qc.ry(Parameter(f"y_{qubit}_{layer}_{i}"), qubit)),
                          "rz" : (lambda qc, qubit, layer, i : qc.rz(Parameter(f"z_{qubit}_{layer}_{i}"), qubit)),
                          "u"  : (lambda qc, qubit, layer, i : qc.u(Parameter(f"theta_{qubit}_{layer}_{i}"),
                                                                        Parameter(f"phi_{qubit}_{layer}_{i}"),
                                                                        Parameter(f"lambda_{qubit}_{layer}_{i}"),
                                                                        qubit))}

controlled_gates = {
    "ecr" : (lambda qc, control, controlado : qc.ecr(control, controlado)), 
    "cx" : (lambda qc, control, controlado: qc.cx(control, controlado)), 
    "cy" : (lambda qc, control, controlado: qc.cy(control, controlado)), 
    "cz" : (lambda qc, control, controlado: qc.cz(control, controlado))
}

controlled_rotation_gates = {
    "crx" : (lambda qc, control, controlado, layer, i : qc.crx(Parameter(f"crx_{control}_{layer}_{i}"), control, controlado)), 
    "cry" : (lambda qc, control, controlado, layer, i : qc.cry(Parameter(f"cry_{control}_{layer}_{i}"), control, controlado)), 
    "crz" : (lambda qc, control, controlado, layer, i : qc.crz(Parameter(f"crz_{control}_{layer}_{i}"), control, controlado)),
    "cu" : (lambda qc, control, controlado, layer, i : qc.append(UGate(Parameter(f"theta_{control}_{controlado}_{layer}_{i}"),
                                                                        Parameter(f"phi_{control}_{controlado}_{layer}_{i}"),
                                                                        Parameter(f"lambda_{control}_{controlado}_{layer}_{i}"))
                                                                        .control(1),[control, controlado]))
}

def aplicar_rotaciones(qc : QuantumCircuit,qubit : int,layer :int,rotaciones : [str]):
    for i, rotacion in enumerate(rotaciones): 
        single_qubit_rotations[rotacion](qc, qubit, layer, i)
    return qc

def aplicar_compuertas_controladas(qc, par, compuertas_controladas, layer):
    control, controlado = par[0], par[1]
    for i, compuerta_controlada in enumerate(compuertas_controladas):
        if compuerta_controlada in controlled_gates.keys():
            controlled_gates[compuerta_controlada](qc, control, controlado)
        else: 
            controlled_rotation_gates[compuerta_controlada](qc, control, controlado, layer, i)
    return qc

        
def fully_entangled_ansatz(num_qubits: int, rotaciones : [str], compuertas_controladas : [str], comprimir_controladas: bool = True):
    #Calculamos pares de qubits sin comprimir
    pares = []
    for i in range(1, int(num_qubits / 2) + 1):
        for q in range(0, num_qubits):
            par = (q , ((q + i) % num_qubits))
            pares.append(par)
    pares = pares[:int(num_qubits * (num_qubits - 1)/2)]

    qc = QuantumCircuit(num_qubits)

    for layer in (range(0,((num_qubits) // 2))):
        #Rotaciones de la capa
        for qubit in range(0, num_qubits): 
            qc = aplicar_rotaciones(qc, qubit, layer, rotaciones)

        #Aplicamos compuertas controladas para los pares de qubits 
        pares_layer = pares[layer * num_qubits : min(len(pares), (layer + 1) * num_qubits)]

        if comprimir_controladas:
            pares_layer = comprimir(pares_layer)

        for par in pares_layer: 
            qc = aplicar_compuertas_controladas(qc, par, compuertas_controladas, layer)    

    return qc

if __name__ == "__main__":
    qc = fully_entangled_ansatz(6, ["rx","rz"], ["cx"])
    print(qc)
    qc = fully_entangled_ansatz(6, ["u"], ["cu"], comprimir_controladas = False)
    qc.draw("mpl", filename="ma_mal.pdf")
    exit(0)