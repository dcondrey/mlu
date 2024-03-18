from modules.quantum_ml.quantum_ml import QuantumML
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

class TestQuantumML(unittest.TestCase):
    def setUp(self):
        self.quantum_ml = QuantumML()

    @patch('modules.quantum_ml.quantum_ml.QuantumML.quantum_algorithm')
    def test_quantum_algorithm(self, mock_quantum_algorithm):
        mock_quantum_algorithm.return_value = None
        self.assertIsNone(self.quantum_ml.quantum_algorithm(), "Quantum algorithm execution should return None.")
        mock_quantum_algorithm.assert_called_once()

    @patch('modules.quantum_ml.quantum_ml.execute')
    @patch('modules.quantum_ml.quantum_ml.QuantumCircuit')
    def test_quantum_algorithm_execution(self, mock_quantum_circuit, mock_execute):
        captured_output = StringIO()          # Create StringIO object to capture output
        sys.stdout = captured_output          # Redirect stdout to the StringIO object

        mock_circuit_instance = MagicMock()
        mock_quantum_circuit.return_value = mock_circuit_instance
        mock_result = MagicMock()
        mock_result.get_counts.return_value = {'00': 100, '11': 100}
        mock_execute.return_value.result.return_value = mock_result

        self.quantum_ml.quantum_algorithm()   # Call the method that prints

        sys.stdout = sys.__stdout__           # Reset redirect
        self.assertIn("Counts: {'00': 100, '11': 100}", captured_output.getvalue(), "Quantum algorithm execution should print correct counts.")

if __name__ == '__main__':
    unittest.main()