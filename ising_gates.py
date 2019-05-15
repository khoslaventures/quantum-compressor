import cirq
import numpy as np
from cirq import protocols

# rotation matrix of theta on the single qubit we will eventually measure
# returns a Y rotation matrix


def R0(theta):
    return cirq.Ry(theta)

# SHIFTU and SHIFTD are for the proper application of Rl, they 'shift' all of the wavefunction's states by one
# i.e. SHIFTU(a|00> + b|01> + c|10> + d|11>) = b|00> + c|01> + d|10> + a|11>)
# The unitary simply a shifted identity matrix


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
        return protocols.CircuitDiagramInfo(wire_symbols=(_shift_to_diagram_symbol(self._unitary_(), args, "ShiftU")), connected=True)

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
        return protocols.CircuitDiagramInfo(wire_symbols=(_shift_to_diagram_symbol(self._unitary_(), args, "ShiftD")), connected=True)

# This is just for display purposes for SHIFTU, SHIFTD in the circuit
# based on qubitMatrixgate from cirq docs.


def _shift_to_diagram_symbol(matrix: np.ndarray,
                             args: protocols.CircuitDiagramInfoArgs, shift: str) -> str:
    if args.precision is not None:
        matrix = matrix.round(args.precision)
    dimensionToAdd = args.known_qubit_count - 2
    result = []
    if args.use_unicode_characters:
        result.append('┌' + '-' * (2 + len(shift)) +
                      '┐\n|' + ' ' * (2 + len(shift)) + '|')
        for i in range(dimensionToAdd):
            if (dimensionToAdd % 2 == 0):
                if(i == dimensionToAdd/2 - 1):
                    result.append('|' + ' ' * (2 + len(shift)) +
                                  '|\n'+'| ' + shift + ' |')
                else:
                    result.append('|' + ' ' * (2 + len(shift)) +
                                  '|\n' + '|' + ' ' * (2 + len(shift)) + '|')
            else:
                if(i == int(dimensionToAdd/2)):
                    result.append('| ' + shift + ' |\n|' +
                                  ' ' * (2 + len(shift)) + '|')
                else:
                    result.append('|' + ' ' * (2 + len(shift)) + '|')
        result.append('└' + '-' * (2 + len(shift)) + '┘')
        return result
