import numpy as np
from src.linear.sampling_strategy import SamplingStrategy

class ArbitrarySampling(SamplingStrategy):
    def __init__(self, probability_distribution_function):
        """
        Initialize the ArbitrarySampling with a given probability distribution function.
        
        :param probability_distribution_function: A function that returns a sample from the desired distribution
        """
        self.probability_distribution_function = probability_distribution_function

    def get_next_sampling_point(self, previous_x, previous_y, noise_amplitude):
        """
        Get the next sampling point by sampling from the provided probability distribution function.
        
        :param previous_x: List of previous x measurements (ignored)
        :param previous_y: List of previous y measurements (ignored)
        :param noise_amplitude: The amplitude of the noise (ignored)
        :return: The next sampling point (x, y)
        """
        return self.probability_distribution_function()
