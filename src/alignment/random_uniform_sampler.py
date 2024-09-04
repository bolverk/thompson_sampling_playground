import numpy as np
from src.alignment.sampling_strategy import SamplingStrategy

class RandomUniformSampler(SamplingStrategy):
    def __init__(self, num_angles, lower_bound=0, upper_bound=2*np.pi):
        """
        Initialize the RandomUniformSampler with the number of angles and the bounds for the uniform distribution.
        
        :param num_angles: The number of angles to sample
        :param lower_bound: The lower bound of the uniform distribution (default is 0)
        :param upper_bound: The upper bound of the uniform distribution (default is 2*pi)
        """
        self.num_angles = num_angles
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_next_sampling_angles(self, previous_angles, previous_measurements, noise_amplitude):
        """
        Get the next list of sampling angles using a uniform distribution.
        
        :param previous_angles: List of lists of previous angle measurements
        :param previous_measurements: List of corresponding measurements
        :param noise_amplitude: The amplitude of the noise
        :return: The next list of angles
        """
        return np.random.uniform(self.lower_bound, self.upper_bound, self.num_angles).tolist()
