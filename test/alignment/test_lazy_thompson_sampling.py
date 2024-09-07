import unittest
import numpy as np
from src.alignment.lazy_thompson_sampling import LazyThompsonSampling
from src.alignment.detector import Detector

class TestLazyThompsonSampling(unittest.TestCase):
    def setUp(self):
        self.num_angles = 2
        self.lts = LazyThompsonSampling(self.num_angles)
        self.detector = Detector([0.5, 1.2], 1e-9)

    def test_initial_sampling(self):
        angles = self.lts.get_next_sampling_angles([],[],self.detector.noise_amplitude)
        self.assertEqual(len(angles), self.num_angles)
        self.assertTrue(all(0 <= angle < 2*np.pi for angle in angles))

    def test_subsequent_sampling(self):
        # Simulate a few rounds of sampling
        previous_angles = []
        previous_intensities = []
        for _ in range(5):
            angles = self.lts.get_next_sampling_angles(previous_angles, previous_intensities, self.detector.noise_amplitude)
            intensity = self.detector.measure(angles)
            previous_angles.append(angles)
            previous_intensities.append(intensity)

        # Test that the subsequent is similar to the previous
        subsequent_angles = self.lts.get_next_sampling_angles(previous_angles, previous_intensities, self.detector.noise_amplitude)
        self.assertEqual(len(subsequent_angles), self.num_angles)
        self.assertTrue(np.allclose(subsequent_angles, previous_angles[-1], atol=1e-1))

    def test_convergence(self):
        # Run many iterations and check if it converges to the true angles
        previous_angles = []
        previous_intensities = []
        for _ in range(100):
            angles = self.lts.get_next_sampling_angles(previous_angles, previous_intensities, self.detector.noise_amplitude)
            intensity = self.detector.measure(angles)
            previous_angles.append(angles)
            previous_intensities.append(intensity)

        final_angles = self.lts.get_next_sampling_angles(previous_angles, previous_intensities, self.detector.noise_amplitude)
        final_angles_centered = final_angles - np.mean(final_angles)
        detector_angles_centered = self.detector.angles - np.mean(self.detector.angles)
        self.assertTrue(np.allclose(final_angles_centered, detector_angles_centered, atol=1e-1))

if __name__ == '__main__':
    unittest.main()
