# test_grover_less_than_k.py

import unittest
from math import ceil, log2, pi, sqrt
from qiskit_aer import *
# from qiskit import Aer, transpile, execute
from qiskit.visualization import plot_histogram
import warnings

from GroverReturn import GroverReturn

# Suppress deprecation warnings from Qiskit (if any)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the functions from your Grover's algorithm implementation
# Assuming the code is saved in a file named 'grover_less_than_k.py'
import grover_less_than_k


# from grover_less_than_k import (
#     load_data_circuit,
#     comparator_circuit,
#     grover_oracle,
#     diffusion_operator,
#     initialize_circuit,
#     build_grover_circuit,
#     run_grover
# )
grover = grover_less_than_k.grover_less_than_k()

class TestGroverLessThanK(unittest.TestCase):

    def test_basic_functionality(self):

        """Test the basic functionality with a simple list and k."""
        int_list = [1, 2, 3, 4, 5]
        K = 4
        counts = grover.run_grover(int_list, K)
        expected_indices = [0, 1, 2]  # Indices of elements less than 4
        found_indices = grover.run_grover(int_list, K).indices
        print(found_indices)
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_no_elements_less_than_K(self):
        """Test when no elements are less than k."""
        int_list = [5, 6, 7, 8]
        K = 5
        counts = grover.run_grover(int_list, K)
        self.assertEqual(counts.counts, GroverReturn({}, []).counts)
        self.assertEqual(counts.indices, GroverReturn({}, []).indices)
        print("No elements less than k.")

    def test_all_elements_less_than_K(self):
        """Test when all elements are less than k."""
        int_list = [1, 2, 3, 4]
        K = 5
        counts = grover.run_grover(int_list, K)
        expected_indices = list(range(len(int_list)))
        found_indices = counts.indices
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_large_K_value(self):
        """Test when k is larger than the maximum element."""
        int_list = [1, 3, 5, 7]
        K = 10
        counts = grover.run_grover(int_list, K)
        expected_indices = list(range(len(int_list)))
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_K_is_zero(self):
        """Test when k is zero."""
        int_list = [1, 2, 3, 4]
        K = 0
        counts = grover.run_grover(int_list, K)
        self.assertEqual(counts.counts, GroverReturn({}, []).counts)
        self.assertEqual(counts.indices, GroverReturn({}, []).indices)

    def test_single_element_less_than_K(self):
        """Test when only one element is less than k."""
        int_list = [5, 6, 1, 7, 8]
        K = 2
        counts = grover.run_grover(int_list, K)
        expected_indices = [2]
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        self.assertIn(expected_indices[0], found_indices)

    def test_duplicate_elements(self):
        """Test list with duplicate elements less than k."""
        int_list = [2, 2, 3, 4, 5]
        K = 3
        counts = grover.run_grover(int_list, K)
        expected_indices = [0, 1]
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_large_list(self):
        """Test with a larger list to check scalability."""
        int_list = list(range(30))  # Reduced size for practical execution time
        K = 15
        counts = grover.run_grover(int_list, K)
        expected_indices = list(range(15))
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_max_integer_value(self):
        """Test with the maximum integer value in the list."""
        int_list = [0, 2 ** 5 - 1]  # Using 5 bits to keep the circuit size reasonable
        K = 2 ** 5
        counts = grover.run_grover(int_list, K)
        expected_indices = [0, 1]
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        for index in expected_indices:
            self.assertIn(index, found_indices)

    def test_negative_integers(self):
        """Test with negative integers in the list."""
        int_list = [-3, -2, -1, 0, 1]
        k = 0
        counts = grover.run_grover(int_list, k)
        expected_indices = []
        found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.counts.keys()]
        self.assertEqual(len(found_indices), len(expected_indices))

    def test_non_integer_values(self):
        """Test with non-integer values (should raise an error)."""
        int_list = [1.5, 2.3, 3.7]
        K = 3
        with self.assertRaises(ValueError):
            counts = grover.run_grover(int_list, K)

    def test_empty_list(self):
        """Test with an empty list."""
        int_list = []
        K = 5
        with self.assertRaises(ValueError):
            counts = grover.run_grover(int_list, K)

    def test_K_equals_maximum_integer(self):
        """Test when k equals the maximum integer in the list."""
        int_list = [2, 4, 6, 8]
        K = 8
        counts = grover.run_grover(int_list, K)
        print(counts.to_dict())
        expected_indices = [0, 1, 2]
        # found_indices = [int(key.replace(' ', '')[:ceil(log2(len(int_list)))], 2) for key in counts.indices]
        for index in expected_indices:
            print(counts)
            self.assertIn(index, counts)
        self.assertNotIn(3, counts)  # Index 3 has value equal to k

    def test_non_positive_K(self):
        """Test with negative k (should raise an error)."""
        int_list = [1, 2, 3, 4]
        K = -1
        counts = grover.run_grover(int_list, K)
        self.assertEqual(counts.counts, GroverReturn({}, []).counts)
        self.assertEqual(counts.counts, GroverReturn({}, []).counts)


if __name__ == '__main__':
    unittest.main()
