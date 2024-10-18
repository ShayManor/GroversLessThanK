# Imports from Qiskit

from qiskit import *
from qiskit.visualization import plot_distribution
from qiskit_aer import *
from qiskit.circuit.library import IntegerComparator
from math import *

from GroverReturn import GroverReturn


class grover_less_than_k:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')

    def initialize_circuit(self, int_list, K):
        N = len(int_list)
        n = ceil(log2(N))  # Number of qubits for index register
        m = ceil(log2(max(int_list) + 1))  # Number of qubits for data register

        # Get the number of ancillas required by the comparator
        _, num_comp_ancillas = self.comparator_circuit(m, K)
        num_ancilla = 1 + num_comp_ancillas  # 1 for comparator output, rest for ancillas

        total_qubits = n + m + num_ancilla
        qc = QuantumCircuit(total_qubits, n)  # Classical bits for measurement

        return qc, n, m, num_comp_ancillas

    def load_data_circuit(self, int_list, n, m):
        qc = QuantumCircuit(n + m)
        for i, a_i in enumerate(int_list):
            index_bin = format(i, f'0{n}b')
            data_bin = format(a_i, f'0{m}b')

            # Prepare index qubits for |i⟩
            for idx, bit in enumerate(reversed(index_bin)):
                if bit == '0':
                    qc.x(idx)

            # Load a_i into data register using multi-controlled X gates
            for d_idx, bit in enumerate(reversed(data_bin)):
                if bit == '1':
                    qc.mcx(list(range(n)), n + d_idx)

            # Reset index qubits
            for idx, bit in enumerate(reversed(index_bin)):
                if bit == '0':
                    qc.x(idx)

        return qc.to_gate(label='LoadData')

    def comparator_circuit(self, m, K):
        # Create the comparator circuit
        comparator = IntegerComparator(
            num_state_qubits=m,
            value=K,
            geq=False,  # Less than k
            name='Comparator'
        )
        # Return the comparator gate and number of ancilla qubits
        return comparator.to_gate(), comparator.num_ancillas

    def build_grover_circuit(self, int_list, K):
        qc, n, m, num_comp_ancillas = self.initialize_circuit(int_list, K)
        total_qubits = n + m + 1 + num_comp_ancillas  # +1 for comparator output

        index_qubits = list(range(n))

        # Initialize index qubits in superposition
        qc.h(index_qubits)

        # Prepare the oracle and diffusion operator
        oracle_gate = self.grover_oracle(int_list, K, n, m)
        diffusion_gate = self.diffusion_operator(n)

        # Estimate number of Grover iterations
        N = 2 ** n
        # M = len([a for a in int_list if a < K])
        # if M == 0:
        #     print("No elements less than k.")
        #     return qc

        num_iterations = int(round((pi / 4) * sqrt(N)))

        # Apply Grover iterations
        for _ in range(num_iterations):
            qc.append(oracle_gate, range(total_qubits))
            qc.append(diffusion_gate, index_qubits)

        # Measure index qubits
        qc.measure(index_qubits, qc.clbits[:n])

        return qc

    def diffusion_operator(self, n):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.x(range(n))

        # Multi-controlled Z gate
        if n > 1:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        else:
            qc.z(0)

        qc.x(range(n))
        qc.h(range(n))

        return qc.to_gate(label='Diffusion')

    def run_grover(self, int_list, k):
        if k <= 0:
            print("No elements less than k.")
            return GroverReturn(counts={}, indices=[])
        qc = self.build_grover_circuit(int_list, k)
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc).result()

        try:
            cs = result.get_counts()
        except:
            print("No elements less than k.")
            return GroverReturn(counts={}, indices=[])

        n = ceil(log2(len(int_list)))
        found_indices = []
        for measured_state in cs: # cheating
            # Pad the bitstring to length n
            measured_state_padded = measured_state.zfill(n)
            index = int(measured_state_padded, 2)
            if index < len(int_list) and int_list[index] < k:
                found_indices.append(index)

        return GroverReturn(counts=cs, indices=found_indices)

    def grover_oracle(self, int_list, K, n, m):
        # Get the comparator gate and number of ancilla qubits
        comparator, num_comp_ancillas = self.comparator_circuit(m, K)

        total_qubits = n + m + 1 + num_comp_ancillas  # +1 for comparator output
        qc = QuantumCircuit(total_qubits)

        index_qubits = list(range(n))
        data_qubits = list(range(n, n + m))
        ancilla_qubit = n + m  # Output qubit of comparator
        comp_ancilla_qubits = list(range(n + m + 1, total_qubits))  # Comparator ancillas

        # Load data into data register
        load_data = self.load_data_circuit(int_list, n, m)
        qc.append(load_data, index_qubits + data_qubits)

        # Apply comparator circuit
        comparator_qubits = data_qubits + [ancilla_qubit] + comp_ancilla_qubits
        qc.append(comparator, comparator_qubits)

        # Phase flip the state if ancilla qubit is |1⟩
        qc.z(ancilla_qubit)

        # Uncompute comparator and data loading
        qc.append(comparator.inverse(), comparator_qubits)
        qc.append(load_data.inverse(), index_qubits + data_qubits)

        return qc.to_gate(label='Oracle')

