import unittest

import numpy as np
from src.linear.noisy_line import NoisyLine

class TestNoisyLine(unittest.TestCase):
    def test_evaluate(self):
        slope = 2.0
        intercept = 1.0
        noise_amplitude = 0.5
        x = 3.0

        noisy_line = NoisyLine(slope, intercept, noise_amplitude)
        result = np.mean([noisy_line.evaluate(x) for _ in range(1000)])

        # Since noise is random, we can't check for an exact value
        expected_value = slope * x + intercept
        self.assertAlmostEqual(result, expected_value, delta=noise_amplitude)

if __name__ == '__main__':
    unittest.main()
