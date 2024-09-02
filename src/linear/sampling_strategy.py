class SamplingStrategy:
    def get_next_sampling_point(self, previous_x, previous_y, noise_amplitude):
        """
        This method should be implemented by subclasses to determine the next sampling point
        based on previous x and y measurements and noise amplitude.
        
        :param previous_x: List of previous x measurements
        :param previous_y: List of previous y measurements
        :param noise_amplitude: The amplitude of the noise
        :return: The next sampling point (x, y)
        """
        raise NotImplementedError("Subclasses should implement this method")
