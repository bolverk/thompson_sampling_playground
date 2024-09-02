import unittest
import numpy as np
from src.linear.thompson_sampling import ThompsonSampling

class TestThompsonSampling(unittest.TestCase):
    def setUp(self):
        # Define a simple prior distribution for testing
        self.prior_distribution = lambda: np.random.normal(0, 1)
        self.noise_amplitude = 1.0
        self.sampling_strategy = ThompsonSampling(self.prior_distribution, self.noise_amplitude)

    def test_get_next_sampling_point(self):
        previous_x = [1, 2, 3, 4, 5]
        previous_y = [2, 4, 6, 8, 10]
        next_point = self.sampling_strategy.get_next_sampling_point(previous_x, previous_y, self.noise_amplitude)
        
        # Check if the next point is a float (x-coordinate)
        self.assertIsInstance(next_point, float)
        
        # Check if the next point is within a reasonable range
        self.assertTrue(-10 < next_point < 10)

if __name__ == '__main__':
    unittest.main()
