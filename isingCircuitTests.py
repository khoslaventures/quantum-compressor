import numpy as np
import pytest
import warnings
import cirq
import isingCircuit as ic

#@Source This is based on the pytests from the homeworks earlier in the semester

#This will test the shift gates on n qubits
@pytest.mark.parametrize('n_qubits', range(2, 10))
def test_shiftu(n_qubits):

	#Makes a circuit, applies a hadamard to the first qubit, and shifts up
	qubits1 = cirq.LineQubit.range(n_qubits)
	qubits2 = cirq.LineQubit.range(n_qubits)
	circuit1 = cirq.Circuit()
	circuit1.append([cirq.H(qubits1[0])])
	circuit1.append(ic.SHIFTU(n_qubits).on(*qubits1))
	simulator1 = cirq.Simulator()
	result1 = simulator1.simulate(circuit1, qubit_order=qubits1)

	circuit2 = cirq.Circuit()
	circuit2.append([cirq.H(qubits2[0])])
	for q in qubits2[1:n_qubits]:
		circuit2.append([cirq.X(q)])
	simulator2 = cirq.Simulator()
	result2 = simulator2.simulate(circuit2, qubit_order=qubits2)
	#print(result1.final_state, result2.final_state)
	assert (result1.final_state == result2.final_state).all()