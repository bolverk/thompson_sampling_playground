import unittest
import numpy as np
from src.alignment.random_uniform_sampler import RandomUniformSampler

class TestRandomUniformSampler(unittest.TestCase):
    def test_get_next_sampling_angles(self):
        num_angles = 5
        lower_bound = 0
        upper_bound = 2 * np.pi
        sampler = RandomUniformSampler(num_angles, lower_bound, upper_bound)
        
        previous_angles = []
        previous_measurements = []
        noise_amplitude = 0.1
        
        angles = sampler.get_next_sampling_angles(previous_angles, previous_measurements, noise_amplitude)
        
        self.assertEqual(len(angles), num_angles)
        for angle in angles:
            self.assertGreaterEqual(angle, lower_bound)
            self.assertLessEqual(angle, upper_bound)

if __name__ == '__main__':
    unittest.main()
