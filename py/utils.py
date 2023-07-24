import numpy as np

# percent level accuracy of all of the frequencies in a frequency bins
THRESHOLD = 0.03

# Inverse velocity squared of dm
INV_VEL_SQ = 10**6

# Canonical minimum multiple for frequency bin
I_MIN = int(INV_VEL_SQ / (1.0 + THRESHOLD))

# Number of seconds in a sidereal day
SIDEREAL_DAY_SECONDS = 86164.0905

# Sidereal frequency
FD = 1.0 / SIDEREAL_DAY_SECONDS

# Directory where X and H data lives
NPY_DIRECTORY = "proj_aux_np"


def coherence_times(total_time):
    """
    Calculates the coherence times used in an analysis given the total time and threshold

    Equation 66:

    T_n = T_tot / (1+q)^(2n)

    with q the threshold, n the coherence time index
    """

    # Initialize return value
    times = []

    # Find max n in eq (??)
    max_n = int(
        round(-0.5 * np.log(1_000_000.0 / total_time) / np.log(1.0 + THRESHOLD)))

    for n in range(max_n + 1):
        # Find the raw coherence time
        raw_coherence_time = total_time / (1.0 + THRESHOLD) ** (2 * n)
        rounded = int(round(raw_coherence_time))

        # Find number in [rounded-10, .., rounded+10] with smallest max prime
        number = 0
        max_prime = np.iinfo(np.int64).max
        for candidate in range(rounded - 10, rounded + 11):
            x = maxprime(candidate)
            if x < max_prime:
                number = candidate
                max_prime = x

        times.append(number)

    # Values should be in descending order.
    return times


def maxprime(n):
    """
    Given a number n, calculate its largest prime factor
    """

    # Deal with base case
    if n == 1:
        return 1

    # Find upper_bound for checks
    upper_bound = int(np.sqrt(n))

    # Iterate through all numbers between 2 and the upper_bound
    for i in range(2, upper_bound + 1):
        if n % i == 0:
            # If n is divisible by i, recursively find the maximum prime of n / i
            return maxprime(n // i)

    # Because we are iterating up, this will return the largest prime factor
    return n


def frequencies_from_coherence_times(coherence_times):
    assert max(coherence_times) > I_MIN, "In order to have frequency multiple i_min be O(v_dm^-2), need max coherence time to be at least v_dm^-2 seconds"

    # Calculate base frequencies, i.e. reciprocal of coherence times
    # Since coherence_times is in descending order, these will be in ascending order
    base_frequencies = [1.0 / x for x in coherence_times]

    # This constructs all frequency bins except for the highest one
    frequency_bins = []
    for bin_index in range(len(base_frequencies) - 1):
        lower, higher = base_frequencies[bin_index], base_frequencies[bin_index + 1]
        assert higher > lower, "frequencies are not in correct order"

        # Find start and end multiples of frequency
        start = 0 if bin_index == 0 else I_MIN
        end = int(higher * I_MIN / lower)
        if end * lower >= higher * I_MIN:
            end -= 1

        assert end * lower < higher * \
            I_MIN, "highest frequency in bin is higher than lowest in next bin"

        # Append tuple containing lowest_multiple, highest_multiple and spacing to frequency_bins
        frequency_bins.append((start, end, lower))

    # Add highest frequency bin
    last_coherence_time_in_seconds = coherence_times[-1]
    highest_frequency = last_coherence_time_in_seconds - 1
    highest_frequency_bin_start = 1.0 / last_coherence_time_in_seconds

    frequency_bins.append(
        (I_MIN, highest_frequency, highest_frequency_bin_start))

    return frequency_bins


def frequency_bin_iter(lo: int, hi_inclusive: int, spacing: float):
    return np.array([spacing * m for m in range(lo, hi_inclusive+1)])


def approximate_sidereal(df: float):

    # Initial guess
    multiple = int(1.0 / SIDEREAL_DAY_SECONDS / df)

    # Check guess and guess + 1
    candidate_1 = np.abs((multiple * df) - 1.0 / SIDEREAL_DAY_SECONDS)
    candidate_2 = np.abs(((multiple + 1) * df) - 1.0 / SIDEREAL_DAY_SECONDS)

    # Return whichever is closest to SIDEREAL_DAY_SECONDS
    if candidate_1 < candidate_2:
        return multiple
    else:
        return multiple + 1


def calculate_stationarity_chunks(num_chunks, chunk_size, chunk_mod):
    return [(k * chunk_size + min(k, chunk_mod), (k + 1) * chunk_size + min(k + 1, chunk_mod)) for k in range(num_chunks)]
