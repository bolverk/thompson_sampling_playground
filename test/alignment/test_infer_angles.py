import unittest
import numpy as np
from src.alignment.detector import Detector
from src.alignment.infer_angles import infer_angles

class TestInferAngles(unittest.TestCase):
    def test_infer_angles(self):
        # Define test data
        decryption_angles = [
            [0, angle] for angle in np.linspace(0, 2*np.pi, 100)
        ]
        detector = Detector([0, 1.23], 1e-9)
        intensity_measurements = [
            detector.measure(angles) for angles in decryption_angles
        ]

        # Call the function
        inferred_angles = infer_angles(decryption_angles, intensity_measurements)

        # Compare the inferred angle to the expected angle
        assert np.std(inferred_angles - detector.angles) < 1e-9

if __name__ == '__main__':
    unittest.main()
