import unittest
import numpy as np
from src.alignment.detector import Detector

class TestDetector(unittest.TestCase):
    def test_measure(self):
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        noise_amplitude = 0.1
        detector = Detector(angles, noise_amplitude)
        
        # Test with the same angles
        measured_distance = detector.measure(angles)
        expected_distance = 16
        self.assertAlmostEqual(measured_distance, expected_distance, places=5)
        
        # Test with different angles
        test_angles = np.zeros_like(angles)
        measured_distance = detector.measure(test_angles)
        expected_distance = 0
        self.assertAlmostEqual(measured_distance, expected_distance, places=5)

if __name__ == '__main__':
    unittest.main()
