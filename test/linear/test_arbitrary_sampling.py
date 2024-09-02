import unittest
import numpy as np
from src.linear.arbitrary_sampling import ArbitrarySampling

class TestArbitrarySampling(unittest.TestCase):
    def test_get_next_sampling_point(self):
        def mock_probability_distribution_function():
            return 1.0
        
        arbitrary_sampling = ArbitrarySampling(mock_probability_distribution_function)
        next_point = arbitrary_sampling.get_next_sampling_point([], [], 0.0)
        
        self.assertEqual(next_point, 1.0)

if __name__ == '__main__':
    unittest.main()
