import cirq
import numpy as np
import random as rand

from cirq import Circuit, ops, linalg, protocols
from cirq.devices import GridQubit
from cirq.google import XmonSimulator

#takes in parameters for a simulation of n qubits to be evolved to a state J
#Jmax, L and T are for defining the adiabatic evolution: the maximum J-value for evolution,
#the number of steps, and time of evolution, respectively.
def ising_circuit(n, J, Jmax, L, T):
    #for L = 2000 and T = 100, the circuit should produce very accurate values
    dt = T / (L + 1)
    m = int(np.log2(n)) + 1
    qubits = cirq.LineQubit.range(m)
    circuit = cirq.Circuit()
    
    #Initial states - H and S gates are for |+>(Y) state, bit flip is for mixed state    
    circuit.append([cirq.H(qubits[m - 1])])
    circuit.append([cirq.S(qubits[m - 1])])

    bit_flip = cirq.BitFlipChannel(0.5)
    for i in range(0, m - 1):
        circuit.append([bit_flip.on(qubits[i])])
        
    
    LJ = int(J * L / Jmax)
    for l in range(0, LJ):
        Jl = Jmax * l / L
        
        
        R0l = R0(-4 * dt)
        circuit.append([R0l.on(qubits[m-1])])
        
        shiftu = SHIFTU(m)
        circuit.append(shiftu(*qubits))
        
        #application of Rl, a rotation matrix on the whole state
        phil = 2 * Jl * dt
        r = cirq.SingleQubitMatrixGate(np.array([[np.cos(phil), -np.sin(phil)], [np.sin(phil), np.cos(phil)]]))
        circuit.append([r.on(qubits[m-1])])
        
        controls = qubits[0:m-1]
        Cr = cirq.ControlledGate(sub_gate = (r**-1), control_qubits = controls)
        circuit.append(Cr.on(qubits[m - 1]))
    
        shiftd = SHIFTD(m)
        circuit.append(shiftd(*qubits))
        
    #these are applied for measurement of Y on qubit m
    circuit.append([cirq.S(qubits[m - 1])**-1])
    circuit.append([cirq.H(qubits[m - 1])])
    circuit.append([cirq.measure(qubits[m - 1], key='x')])
    return circuit

#rotation matrix on the qubit we will eventually measure
def R0(theta):
    return cirq.Ry(theta)

#SHIFTU and SHIFTD are for the proper application of Rl, they 'shift' all of the wavefunction's states by one
#i.e. SHIFTU(a|00> + b|01> + c|10> + d|11>) = b|00> + c|01> + d|10> + a|11>)
class SHIFTU(cirq.Gate):
    def __init__(self, num_qubits):
        super(SHIFTU, self)
        self._num_qubits = num_qubits
        
    def num_qubits(self) -> int:
        return self._num_qubits   
        
    def _unitary_(self):
        unitary = np.eye(2**self._num_qubits, k=1)
        unitary[-1][0] = 1
        return unitary
    
    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(_shift_to_diagram_symbol(self._unitary_(), args, "ShiftU")), connected = True)

#SHIFTD = SHIFTU^-1
class SHIFTD(cirq.Gate):
    def __init__(self, num_qubits):
        super(SHIFTD, self)
        self._num_qubits = num_qubits
        
    def num_qubits(self) -> int:
        return self._num_qubits
        
    def _unitary_(self):
        unitary = np.eye(2**self._num_qubits, k=-1)
        unitary[0][-1] = 1
        return unitary
    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(_shift_to_diagram_symbol(self._unitary_(), args, "ShiftD")), connected = True)

#This is just for display purposes for SHIFTU, SHIFTD
def _shift_to_diagram_symbol(matrix: np.ndarray,
                              args: protocols.CircuitDiagramInfoArgs, shift: str) -> str:
    if args.precision is not None:
            matrix = matrix.round(args.precision)
    dimensionToAdd = args.known_qubit_count - 2
    result = []
    if args.use_unicode_characters:
        result.append('┌' + '-' * (2 + len(shift)) + '┐\n|' + ' ' * (2 + len(shift)) + '|')
        for i in range(dimensionToAdd):
            if (dimensionToAdd % 2 == 0):
                if(i == dimensionToAdd/2 - 1):
                    result.append('|' +' ' * (2 + len(shift)) + '|\n'+'| '+ shift +' |')
                else:
                    result.append('|' + ' ' * (2 + len(shift)) + '|\n' + '|' + ' ' * (2 + len(shift)) + '|')
            else:
                if(i == int(dimensionToAdd/2)):
                    result.append('| '+ shift +' |\n|' + ' ' * (2 + len(shift)) + '|')
                else:
                    result.append('|' + ' ' * (2 + len(shift)) + '|')
        result.append('└' + '-' * (2 + len(shift)) + '┘')
        return result