import cirq
import numpy as np
import pytest

from ising_circuit import IsingCircuit


@pytest.mark.repeat(20)
def test_wave_function():
    """Test the wave function outputs. They should be the same for the
    given paramters."""
    # Make circuit for four qubit Ising chain using two qubits.
    # num_qubits is 4, J = 1, J_max = 2, L = 2400, T = 240.1, dt = 0.1
    simulator = cirq.Simulator()
    ic = IsingCircuit(4, 1, 2, 2400, 240.1)
    # wave function prior to measurement
    wave_function = simulator.simulate(ic.circuit, qubit_order=ic.qubits)
    # Calculate the magnetization $M(J)$ from the wave function by decomposing
    # the final state into the $+1$ and $-1$ eigenstates, determining the total
    # probability for each, and computing the expectation $\langle Y_{m}
    # \rangle$:

    final_state = wave_function.final_state

    p_1_wfn = (sum([np.absolute(i)**2 for i in final_state[0:4]]))
    p_n1_wfn = (sum([np.absolute(i)**2 for i in final_state[4:8]]))

    Y_wfn = 1 * p_1_wfn + -1 * p_n1_wfn

    M_wfn = -Y_wfn
    # Precalculated, possible values
    possible_M = [-0.5759276624275007, -0.4421064277467738,
                  0.44210643345027945, 0.5759276585630806]
    assert(M_wfn in possible_M)
    print(M_wfn)


@pytest.mark.repeat(20)
def test_j_is_1_decomposition():
    """Test the J=1 decomposition against the function generated circuit.
    We take some measurements and expect a small margin of difference."""
    qubits = cirq.LineQubit.range(3)
    qubit0, qubit1, qubit2 = qubits

    circuit = cirq.Circuit()

    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.S(qubit0)**-1])
    circuit.append([cirq.H(qubit2)])
    circuit.append([cirq.CNOT(control=qubit2, target=qubit1)])

    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.CNOT(control=qubit0, target=qubit1)])
    circuit.append([cirq.Z(qubit0)])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.CNOT(control=qubit0, target=qubit1)])
    circuit.append([cirq.S(qubit0)])
    circuit.append([cirq.T(qubit1)])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.Z(qubit1)])
    circuit.append([cirq.CNOT(control=qubit0, target=qubit1)])
    circuit.append([cirq.T(qubit0)])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.T(qubit0)])

    for _ in range(4):
        circuit.append([cirq.S(qubit0)])
        circuit.append([cirq.H(qubit0)])
        circuit.append([cirq.T(qubit0)])

    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.T(qubit0)])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.T(qubit0)**-1])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.T(qubit0)**-1])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.T(qubit0)])
    circuit.append([cirq.Z(qubit0)])
    circuit.append([cirq.H(qubit0)])
    circuit.append([cirq.measure(qubit0, key='x')])

    # Take measurements
    simulator = cirq.Simulator()
    reps = 10
    results = simulator.run(circuit, repetitions=reps)
    hist = results.histogram(key='x')
    for k in hist:
        v = hist[k]
        hist[k] = v

    assert(hist[0] in range(6, 11))
    assert(hist[1] in range(5))

    # Make circuit for four qubit Ising chain using two qubits.
    # num_qubits is 4, J = 1, J_max = 2, L = 2400, T = 240.1, dt = 0.1
    ic = IsingCircuit(4, 1, 2, 2400, 240.1)
    ic.apply_measurement()
    ic_results = simulator.run(ic.circuit, repetitions=reps)
    hist = ic_results.histogram(key='x')
    for k in hist:
        v = hist[k]
        hist[k] = v

    assert(hist[0] in range(6, 11))
    assert(hist[1] in range(5))
