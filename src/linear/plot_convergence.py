import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
plt.rcParams['font.size'] = 16

from src.linear.lazy_thompson_sampling import LazyThompsonSampling
from src.linear.arbitrary_sampling import ArbitrarySampling
from src.linear.thompson_sampling import ThompsonSampling

def compare_sampling_strategies(num_points_list, num_trials=100):
    arbitrary_results = []
    thompson_results = []
    lazy_thompson_results = []

    for num_points in num_points_list:
        arbitrary_errors = []
        thompson_errors = []
        lazy_thompson_errors = []

        for _ in range(num_trials):
            # Generate true function and noisy observations
            true_slope = 2.0
            true_intercept = 1.0
            noise_amplitude = 1.0
            true_root = -true_intercept / true_slope
            x_values = np.linspace(0, 10, num_points)
            y_values = true_slope * x_values + true_intercept + np.random.normal(0, noise_amplitude, num_points)

            # Arbitrary Sampling
            arbitrary_sampling = ArbitrarySampling(lambda: np.random.uniform(0, 10))
            arbitrary_x = []
            arbitrary_y = []
            for _ in range(num_points):
                x = arbitrary_sampling.get_next_sampling_point(arbitrary_x, arbitrary_y, noise_amplitude)
                y = true_slope * x + true_intercept + np.random.normal(0, noise_amplitude)
                arbitrary_x.append(x)
                arbitrary_y.append(y)
            arbitrary_fit = np.polyfit(arbitrary_x, arbitrary_y, 1)
            arbitrary_error = np.abs(true_root + arbitrary_fit[1]/arbitrary_fit[0])
            arbitrary_errors.append(arbitrary_error)

            # Thompson Sampling
            thompson_sampling = ThompsonSampling(lambda: np.random.normal(0, 1), noise_amplitude)
            thompson_x = []
            thompson_y = []
            for _ in range(num_points):
                x = thompson_sampling.get_next_sampling_point(thompson_x, thompson_y, noise_amplitude)
                y = true_slope * x + true_intercept + np.random.normal(0, noise_amplitude)
                thompson_x.append(x)
                thompson_y.append(y)
            thompson_fit = np.polyfit(thompson_x, thompson_y, 1)
            thompson_error = np.abs(true_root + thompson_fit[1]/thompson_fit[0])
            thompson_errors.append(thompson_error)

            # Lazy Thompson Sampling
            lazy_thompson_sampling = LazyThompsonSampling(lambda: np.random.normal(0, 1), noise_amplitude)
            lazy_thompson_x = []
            lazy_thompson_y = []
            for _ in range(num_points):
                x = lazy_thompson_sampling.get_next_sampling_point(lazy_thompson_x, lazy_thompson_y, noise_amplitude)
                y = true_slope * x + true_intercept + np.random.normal(0, noise_amplitude)
                lazy_thompson_x.append(x)
                lazy_thompson_y.append(y)
            lazy_thompson_fit = np.polyfit(lazy_thompson_x, lazy_thompson_y, 1)
            lazy_thompson_error = np.abs(true_root + lazy_thompson_fit[1]/lazy_thompson_fit[0])
            lazy_thompson_errors.append(lazy_thompson_error)

        arbitrary_results.append(np.mean(arbitrary_errors))
        thompson_results.append(np.mean(thompson_errors))
        lazy_thompson_results.append(np.mean(lazy_thompson_errors))
    # Get default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.loglog(num_points_list, arbitrary_results, 'o', label='Arbitrary Sampling', color=color_cycle[0])
    plt.loglog(num_points_list, thompson_results, 'o', label='Thompson Sampling', color=color_cycle[1])
    plt.loglog(num_points_list, lazy_thompson_results, 'o', label='Lazy Thompson Sampling', color=color_cycle[2])

    # Fit power laws
    arbitrary_fit = np.polyfit(np.log(num_points_list), np.log(arbitrary_results), 1)
    thompson_fit = np.polyfit(np.log(num_points_list), np.log(thompson_results), 1)
    lazy_thompson_fit = np.polyfit(np.log(num_points_list), np.log(lazy_thompson_results), 1)

    # Plot power law fits
    plt.loglog(num_points_list, np.exp(np.polyval(arbitrary_fit, np.log(num_points_list))), '--', label=f'Arbitrary Fit: slope={arbitrary_fit[0]:.2f}', color=color_cycle[0])
    plt.loglog(num_points_list, np.exp(np.polyval(thompson_fit, np.log(num_points_list))), '--', label=f'Thompson Fit: slope={thompson_fit[0]:.2f}', color=color_cycle[1])
    plt.loglog(num_points_list, np.exp(np.polyval(lazy_thompson_fit, np.log(num_points_list))), '--', label=f'Lazy Thompson Fit: slope={lazy_thompson_fit[0]:.2f}', color=color_cycle[2])
    plt.xlabel('Number of Sampling Points')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Comparison of Sampling Strategies')
    plt.tight_layout()
    plt.show()

# Example usage
num_points_list = [2**n for n in range(3,12)]
compare_sampling_strategies(num_points_list)        
