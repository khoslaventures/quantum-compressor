import cirq
import numpy as np

from ising_gates import R0, SHIFTD, SHIFTU


"""Demonstrates a compressed simulation of the transverse field 1D-Ising
Interaction. This is an example of compressed quantum computation simulating
the Ising chain of n qubits by using only log(n) qubits.

Compressed quantum simulation of the transverse field Ising model has already
been realized in an experiment using NMR quantum computing. Here, we also
simulate this model with open boundary conditions, whose evolution is governed
by the Hamiltonian:

         n          n-1
H(J) =   Σ  Z_k + J  Σ   X_k X_k+1
        k=1         k=1

Where X_k (Z_k) denote X (Z) acting on qubit k, respectively. In the limit
n->infinity, the system undergoes a quantum phase transition at J=1 that is
reflected in the discontinuity of the second derivative of the transverse
magnetization.

M(J), the magnetization can be measured by:
1. Initially prepare the system in the ground state of H(0)
2. Evolve the system to the ground state of H(J) by changing the parameter J
   adiabatically
3. Apply Z on a single qubit and measure to obtain the transverse magnetization
   M(J)

As this adiabatic evolution together with the measurement of the magnetization
is a matchgate circuit, the whole computation can be compressed into a
universal quantum computation running on only m = log(n) qubits.

=== REFERENCE ===

M. Hebenstreit, D. Alsina, J. I. Latorre: “Compressed quantum computation using
the IBM Quantum Experience”, 2017, Phys. Rev. A 95, 052339 (2017);
arXiv:1701.02970. DOI: 10.1103/PhysRevA.95.052339.

TODO(akash): Provide example of circuit with given inputs in docstring
"""


class IsingCircuit:
    def __init__(self, n, J, J_max, L, T):
        """Generate a circuit for compressed quantum simulation of the Ising
        model in the 1D case using a 1D non-circular arrangement of qubits,
        which will allow the hamiltonian to be compressed via matchgates.

        Args:
            n: The num of qubits to be evolved to a state J
            J: State of the magnetization, ratio of interaction strength and
               external field strength
            J_max: The maximium J value for evolution
            L: The number of steps
            T: The time of evolution

        Returns: circuit for compressed Ising simulation

        For L = 2000 and T = 100, the circuit should produce very accurate
        values.  However, it makes for very slow runtimes, so we've been using
        L = 200 and T = 10 which Kraus claimed to produce decent enough
        results.
        """
        self.dt = T / (L + 1)
        self.m = int(np.log2(n)) + 1
        self.qubits = cirq.LineQubit.range(self.m)
        self.circuit = cirq.Circuit()

        # Initial states - H and S gates are for |+>(Y) state, bit flip is for
        # mixed state
        self.circuit.append([cirq.H(self.qubits[self.m - 1])])
        self.circuit.append([cirq.S(self.qubits[self.m - 1])])

        bit_flip = cirq.BitFlipChannel(0.5)
        for i in range(0, self.m - 1):
            self.circuit.append([bit_flip.on(self.qubits[i])])

        # LJ determines the number of adiabatic steps to take
        self.LJ = int(J * L / J_max)
        for l in range(0, self.LJ):
            Jl = J_max * l / L

            # Rotate qubit m
            R0l = R0(-4 * self.dt)
            self.circuit.append([R0l.on(self.qubits[self.m-1])])

            # shift qubit states up so the rotation matrix Rl acts on the
            # states correctly
            shiftu = SHIFTU(self.m)
            self.circuit.append(shiftu(*self.qubits))

            # application of Rl, a rotation matrix on the whole state
            # phi_l is the angle
            # We apply the rotation gate (r) to the pair of states we care
            # about (they are on qubit m after shifting)
            phi_l = 2 * Jl * self.dt
            r = cirq.SingleQubitMatrixGate(
                np.array([[np.cos(phi_l), -np.sin(phi_l)], [np.sin(phi_l),
                                                            np.cos(phi_l)]]))
            self.circuit.append([r.on(self.qubits[self.m-1])])
            # We then apply a controlled inverse of (r), with all the other
            # qubits as controls This effectively gives us our desired Rl on
            # the wavefunction
            controls = self.qubits[0:self.m-1]
            Cr = cirq.ControlledGate(sub_gate=(r**-1), control_qubits=controls)
            self.circuit.append(Cr.on(self.qubits[self.m - 1]))

            # Shift back down for R0 to work correctly
            shiftd = SHIFTD(self.m)
            self.circuit.append(shiftd(*self.qubits))

        # these are applied for measurement of Y on qubit self.m
        self.circuit.append([cirq.S(self.qubits[self.m - 1])**-1])
        self.circuit.append([cirq.H(self.qubits[self.m - 1])])

    def apply_measurement(self):
        self.circuit.append([cirq.measure(self.qubits[self.m - 1], key='x')])
