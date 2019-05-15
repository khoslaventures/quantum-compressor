import cirq
import numpy as np
from cirq import protocols


def R0(theta):
    """Rotation matrix of theta on the single qubit we will eventually measure.

    Args:
        theta: Angle of rotation
    Returns: a Y rotation matrix
    """
    return cirq.Ry(theta)


"""SHIFTU and SHIFTD are for the proper application of Rl. They shift the
overall state's wave function """


class SHIFTU(cirq.Gate):
    """Gate acting on n qubits that shifts the basis of the all the
    wavefunction's states up by one using a (2^n x 2^n) shifted identity matrix.
    For a single qubit, this is an X gate. For two-qubit gate (4x4 unitary),
    the unitary will be:

    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

    i.e. SHIFTU(a|00> + b|01> + c|10> + d|11>) = b|00> + c|01> + d|10> + a|11>)

    Args:
        num_qubits: the number of qubits on which the gate is to be applied
    Returns: a gate with corresponding unitary as described above

    """

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
        return protocols.CircuitDiagramInfo(wire_symbols=(_shift_to_diagram_symbol(args, "ShiftU")), connected=True)


class SHIFTD(cirq.Gate):
    """Gate acting on n qubits that shifts the basis of the all the wave
    function's states down by one using a (2^n x 2^n) shifted identity matrix.
    For a single qubit, this is an X gate. For two-qubit gate (4x4 unitary),
    the unitary will be:

    [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]

    i.e. SHIFTD(a|00> + b|01> + c|10> + d|11>) = d|00> + a|01> + b|10> + c|11>)

    SHIFTD = SHIFTU^-1

    Args:
        num_qubits: the number of qubits on which the gate is to be applied
    Returns: a gate with corresponding unitary as described above

    """

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
        return protocols.CircuitDiagramInfo(wire_symbols=(
            _shift_to_diagram_symbol(args, "ShiftD")), connected=True)


def _shift_to_diagram_symbol(args: protocols.CircuitDiagramInfoArgs,
                             shift: str) -> str:
    """This takes the shift gates and makes them display more nicely on the
    circuit.

    Specifically this is for display purposes of SHIFTU, SHIFTD in the circuit.
    Based on cirq.ops.matrix_gates from cirq docs.

    Args:
        args: arguments from the circuit, namely (known_qubit_count) for the
              number of qubits to act on and (use_unicode_characters) to make 
              sure it is fine to use unicode
        shift: type of shift gate to display
        Returns: Tuple of strings to be displayed on circuit
    """
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
