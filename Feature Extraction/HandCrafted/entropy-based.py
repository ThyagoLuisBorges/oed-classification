import numpy as np
from libpysal.weights import lat2W
from esda.moran import Moran
from matplotlib import image
from scipy import stats

def shannon_entropy_vectorized(arr):

    # Get unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(arr)

    # Calculate entropy using NumPy's vectorized operations
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def renyi_entropy_vectorized(arr, order):

    # Get unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(arr)

    # Calculate Rényi entropy using NumPy's vectorized operations
    sum_probability = np.sum(probabilities ** order)
    entropy = np.log2(sum_probability) / (1 - order)

    return entropy

def tsallis_entropy_vectorized(arr, order):

    # Get unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(arr)

    # Calculate Tsallis entropy using NumPy's vectorized operations
    sum_probability = np.sum(probabilities ** order)
    entropy = 2 * (1 - sum_probability) / (order - 1)

    return entropy

def kapur_entropy_vectorized(arr, alpha, beta):

    # Get unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(arr)

    # Calculate Kapur entropy using NumPy's vectorized operations
    sum_prob_alpha = np.sum(probabilities ** alpha)
    sum_prob_beta = np.sum(probabilities ** beta)
    entropy = np.log2(sum_prob_alpha / sum_prob_beta) / (beta - alpha)

    return entropy

def moran_index_vectorized(arr):

    # Create a spatial weights matrix
    w = lat2W(arr.shape[0], arr.shape[1], rook=False)

    # Calculate Moran's I for each layer
    mi_values = np.array([Moran(arr[:, :, idx], w).I for idx in range(arr.shape[2])])

    # Calculate the average Moran's I
    mi_mean = np.mean(mi_values)

    return mi_mean

# Returns the highest and lowest levels between the r, g, and b axes
def get_min_max_in_axes(arr):
    return np.stack([arr.min(axis=(0, 1)), arr.max(axis=(0, 1))], axis=1)

# Calculates the number of slices on a certain axis
def calculate_slices(arr, min_max_array, slice_size):
    min_levels, max_levels = min_max_array.T
    num_slices_per_axis = np.ceil((max_levels - min_levels + 1) / slice_size).astype(int)

    # Remove empty boxes by checking unique levels
    for idx in range(arr.shape[2]):
        levels = np.unique(arr[:, :, idx])
        slice_boundaries = min_levels[idx] + np.arange(0, num_slices_per_axis[idx]) * slice_size
        empty_boxes = ~np.isin(slice_boundaries, levels)
        num_slices_per_axis[idx] -= np.sum(empty_boxes)

    return num_slices_per_axis

def count_boxes(arr, s):
    (x, y, layers) = arr.shape
    slice_size = np.floor((s * 256) / y).astype(int)

    # Compute all slices in one go using stride tricks to avoid loops
    x_indices = np.arange(0, x, s - 1)
    y_indices = np.arange(0, y, s - 1)

    total_slices = 0
    for x_start, y_start in np.ndindex(x_indices.shape[0], y_indices.shape[0]):
        x_end = min(x_indices[x_start] + s, x)
        y_end = min(y_indices[y_start] + s, y)

        # Slice the array
        arr_slice = arr[x_indices[x_start]:x_end, y_indices[y_start]:y_end, :]

        # Find min and max along the RGB axes and calculate slices
        min_max_array = get_min_max_in_axes(arr_slice)
        num_slices = calculate_slices(arr_slice, min_max_array, slice_size)

        # Multiply to get the number of boxes
        total_slices += np.prod(num_slices)

    return total_slices

def fractal_dimension_vectorized(arr):
    points = [[], []]
    (x, y, layers) = arr.shape
    smallest_size = min(x, y)
    max_block_size = smallest_size // 2
    block_size = 9

    prev_num_blocks = -1
    while block_size <= max_block_size:
        curr_num_blocks = np.ceil(smallest_size / block_size).astype(int)
        if curr_num_blocks != prev_num_blocks:
            points[1].append(np.log(count_boxes(arr, block_size)))
            points[0].append(np.log(1 / block_size))
            prev_num_blocks = curr_num_blocks
        block_size += 1

    # Perform linear regression on the log-log data to calculate the slope
    res = stats.linregress(points[0], points[1])
    return res.slope