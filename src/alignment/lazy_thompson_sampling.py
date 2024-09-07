import numpy as np
from .sampling_strategy import SamplingStrategy
from .infer_angles import infer_angles

class LazyThompsonSampling(SamplingStrategy):
    def __init__(self, num_angles):
        self.num_angles = num_angles

    def get_next_sampling_angles(self, previous_angles, previous_measurements, noise_amplitude):
        if len(previous_angles) < self.num_angles + 1:
            # For the first iteration, use random angles
            return np.random.uniform(0, 2*np.pi, self.num_angles)
        else:
            # Infer angles based on previous measurements
            inferred_angles = infer_angles(previous_angles, previous_measurements)
            
            # Use the inferred angles as the next set, with some added noise for exploration
            return inferred_angles