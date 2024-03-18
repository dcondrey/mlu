from modules.accelerators.accelerators import Accelerators
import unittest
from unittest.mock import patch

class TestAccelerators(unittest.TestCase):
    def setUp(self):
        self.accelerators = Accelerators()

    @patch('modules.accelerators.accelerators.Accelerators.use_gpu')
    def test_use_gpu(self, mock_use_gpu):
        self.accelerators.use_gpu()
        mock_use_gpu.assert_called_once()
        self.assertTrue(mock_use_gpu.called, "use_gpu method was not called as expected.")

    @patch('modules.accelerators.accelerators.Accelerators.use_tpu')
    def test_use_tpu(self, mock_use_tpu):
        self.accelerators.use_tpu()
        mock_use_tpu.assert_called_once()
        self.assertTrue(mock_use_tpu.called, "use_tpu method was not called as expected.")

    @patch('modules.accelerators.accelerators.Accelerators.check_gpu_memory')
    def test_check_gpu_memory(self, mock_check_gpu_memory):
        self.accelerators.check_gpu_memory()
        mock_check_gpu_memory.assert_called_once()
        self.assertTrue(mock_check_gpu_memory.called, "check_gpu_memory method was not called as expected.")

    @patch('modules.accelerators.accelerators.Accelerators.check_tpu_memory')
    def test_check_tpu_memory(self, mock_check_tpu_memory):
        self.accelerators.check_tpu_memory()
        mock_check_tpu_memory.assert_called_once()
        self.assertTrue(mock_check_tpu_memory.called, "check_tpu_memory method was not called as expected.")

if __name__ == '__main__':
    unittest.main()