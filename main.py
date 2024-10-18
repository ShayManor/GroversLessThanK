from qiskit.visualization import plot_distribution

import grover_less_than_k

int_list = [4, 6, 9, 1, 2, 5, 2, 11, 15, 4]
# result should be [4, 1, 2, 2, 4]
K = 5
grover = grover_less_than_k.grover_less_than_k()
results = grover.run_grover(int_list, K)
print(results.counts)
print(results.indices)
plt = plot_distribution(results.counts)
plt.show()
