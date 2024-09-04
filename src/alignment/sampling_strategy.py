from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    @abstractmethod
    def get_next_sampling_angles(self, previous_angles, previous_measurements, noise_amplitude):
        """
        Get the next list of sampling angles using the specified strategy.
        
        :param previous_angles: List of lists of previous angle measurements
        :param previous_measurements: List of corresponding measurements
        :param noise_amplitude: The amplitude of the noise
        :return: The next list of angles
        """
        pass
