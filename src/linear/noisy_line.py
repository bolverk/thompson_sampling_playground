import numpy as np

class NoisyLine:
    def __init__(self, slope, intercept, noise_amplitude):
        self.slope = slope
        self.intercept = intercept
        self.noise_amplitude = noise_amplitude

    def evaluate(self, x):
        noise = np.random.normal(0, self.noise_amplitude)
        return self.slope * x + self.intercept + noise
