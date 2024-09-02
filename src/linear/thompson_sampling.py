
import numpy as np
from src.linear.sampling_strategy import SamplingStrategy

class ThompsonSampling(SamplingStrategy):
    def __init__(self, prior_distribution, noise_amplitude):
        """
        Initialize the ThompsonSampling with a given prior distribution.
        
        :param prior_distribution: A function that returns a sample from the prior distribution
        """
        self.prior_distribution = prior_distribution
        self.noise_amplitude = noise_amplitude

    def get_next_sampling_point(self, previous_x, previous_y, noise_amplitude):
        """
        Get the next sampling point using the Thompson Sampling strategy.
        
        :param previous_x: List of previous x measurements
        :param previous_y: List of previous y measurements
        :param noise_amplitude: The amplitude of the noise
        :return: The next sampling point (x, y)
        """
        # Placeholder for the actual implementation

        fit = np.polyfit(previous_x, previous_y, 1)
        # Calculate the covariance matrix of the previous measurements
        X = np.vstack([previous_x, np.ones(len(previous_x))]).T
        hessian = X@X.T/self.noise_amplitude**2
        covariance = np.linalg.pinv(hessian)
        variance = (fit[1]/fit[0])**2*(
            covariance[0,0]/fit[0]**2 + 2*covariance[0,1]/fit[0]/fit[1] + covariance[1,1]/fit[1]**2
        )
        return np.random.normal(-fit[1]/fit[0], np.sqrt(variance))
