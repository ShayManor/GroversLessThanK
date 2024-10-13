import math
from lib2to3.pytree import convert

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, ZGate, GroverOperator
from qiskit.primitives import Sampler
from qiskit_aer import *

def run_grover(int_list, marked_list, backend):
    int_list = convert_to_bitstring(int_list, int(math.ceil(math.log2(max(int_list) + 1))))
    marked_list = convert_to_bitstring(marked_list, int(math.ceil(math.log2(max(marked_list) + 1))))
    backend = Aer.get_backend(backend)
    qc = grovers_circuit(int_list, marked_list)
    return backend.run(qc).result().get_counts()

def grover_oracle(int_list, marked_list):
    if type(marked_list) != list:
        marked_list = [marked_list]

    m = len(marked_list[0])
    qc = QuantumCircuit(m)
    for target in marked_list:
        if len(target) != m:
            raise ValueError("All marked elements must have the same length.")
        rev_target = target[::-1] # reverse the target to match qiskit's qubit ordering
        zero_inds = [ind for ind in range(m) if rev_target.startswith("0", ind)]
        qc.x(zero_inds)
        qc.compose(MCMT(ZGate(), m - 1, 1), inplace=True)
        qc.x(zero_inds)
    return qc

def convert_to_bitstring(nums, num_qubits):
    """Convert an integer to a bitstring"""
    for i in range(len(nums)):
        nums[i] = format(nums[i], '0{}b'.format(num_qubits))
    return nums

def grovers_circuit(int_list, marked_list):
    oracle = grover_oracle(int_list, marked_list)
    grover_op = GroverOperator(oracle)
    optimal_num_iterations = math.floor(
    math.pi / (4 * math.asin(math.sqrt(len(marked_list) / 2**grover_op.num_qubits)))
    )
    qc = QuantumCircuit(grover_op.num_qubits)
    # Create even superposition of all basis states
    qc.h(range(grover_op.num_qubits))
    # Apply Grover operator the optimal number of times
    qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
    # Measure all qubits
    qc.measure_all()
    qc.draw(output="mpl", style="iqp")
    return qc

print(run_grover(list(range(8)), [4], 'qasm_simulator'))