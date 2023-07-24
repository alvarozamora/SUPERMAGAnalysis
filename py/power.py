# Precomputes spectra by stationarity time and save to disk
# for each stationarity time.
import numpy as np
from numpy import pi as PI, sqrt
from utils import coherence_times, frequencies_from_coherence_times, approximate_sidereal, FD, NPY_DIRECTORY, calculate_stationarity_chunks
from tqdm import tqdm
import os
from pathlib import Path

# Stationarity time in seconds
STATIONARITY_TIME = 24 * 60 * 60

# Every stationarity time is broken up into chunks of size tau
TAU = 16384

# Frequency range when taking an FFT with length = 2 * TAU, (1 second spacing)
SPECTRA_FREQS = np.append(np.fft.fftfreq(2*TAU) % 1.0, 1.0)


LEAP_DAYS = 5
TIME_2003_TO_2020 = 18 * 365 * 24 * 60 * 60 + LEAP_DAYS * 24 * 60 * 60


def find_overlap_chunks(coherence_time, coherence_chunk):
    stationarity_chunks = []

    # First calculate start and end second for the coherence_time
    start_coh = coherence_time * coherence_chunk
    end_coh = min(start_coh + coherence_time, TIME_2003_TO_2020)

    # Calculate number of stationarity chunks
    full_chunks = TIME_2003_TO_2020 // STATIONARITY_TIME
    last_chunk = TIME_2003_TO_2020 % STATIONARITY_TIME != 0
    total_chunks = full_chunks + int(last_chunk)

    # Calculate tau chunks
    # This is the same for all full stationarity chunks, but not for the last one (if it exists)
    num_chunks = STATIONARITY_TIME // TAU
    chunk_size = STATIONARITY_TIME // num_chunks
    chunk_mod = STATIONARITY_TIME % num_chunks
    tau_chunks = calculate_stationarity_chunks(
        num_chunks, chunk_size, chunk_mod)

    for chunk in range(total_chunks):
        # Calculate start and end second
        # Note: This end_second is exclusive
        start_second = chunk * STATIONARITY_TIME
        end_second = min(start_second + STATIONARITY_TIME,
                         TIME_2003_TO_2020)  # minimum deals with last case

        # Record overlap if any
        overlap = calculate_overlap(
            start_coh, end_coh, start_second, end_second)
        if overlap > 0:
            stationarity_chunks.append((chunk, overlap))

    return stationarity_chunks


def calculate_overlap(start1, end1, start2, end2):
    return max(0, (min(end1, end2)) - max(start1, start2))


def get_stationarity_chunk_dir(chunk):
    return f"spectra/chunk_{chunk:05d}"


if __name__ == "__main__":
    def calculate_spectra(stationarity_chunk_1, stationarity_chunk_2, tau_chunks):
        """
        Given a stationarity_chunk of length STATIONARITY_TIME and a set of tau_chunks,
        this breaks down the stationarity chunk into tau chunks and sums up the mul ffts
        X1 * conj(X2) divided by the length of the tau chunk
        """
        power = np.zeros(2*TAU) + 0j
        fft_buffer_1 = np.zeros(2*TAU) + 0j
        fft_buffer_2 = np.zeros(2*TAU) + 0j

        # Break into tau chunks
        for (lo, hi) in tau_chunks:

            # Fill in fft_buffer with tau chunk
            fft_buffer_1[:] = 0
            fft_buffer_2[:] = 0
            fft_buffer_1[:hi-lo] = stationarity_chunk_1[lo:hi]
            fft_buffer_2[:hi-lo] = stationarity_chunk_2[lo:hi]

            # Calculate fft square and add to power
            power += np.fft.fft(fft_buffer_1) * \
                np.conj(np.fft.fft(fft_buffer_2)) / (hi - lo)

        return power / len(tau_chunks)

    # Calculate number of stationarity chunks
    full_chunks = TIME_2003_TO_2020 // STATIONARITY_TIME
    last_chunk = TIME_2003_TO_2020 % STATIONARITY_TIME != 0
    total_chunks = full_chunks + int(last_chunk)

    # Calculate tau chunks
    # This is the same for all full stationarity chunks, but not for the last one (if it exists)
    num_chunks = STATIONARITY_TIME // TAU
    chunk_size = STATIONARITY_TIME // num_chunks
    chunk_mod = STATIONARITY_TIME % num_chunks
    tau_chunks = calculate_stationarity_chunks(
        num_chunks, chunk_size, chunk_mod)

    for chunk in tqdm(range(total_chunks)):
        # Calculate start and end second
        # Note: This end_second is exclusive
        start_second = chunk * STATIONARITY_TIME
        end_second = min(start_second + STATIONARITY_TIME,
                         TIME_2003_TO_2020)  # minimum deals with last case

        # Create directory
        chunk_dir = get_stationarity_chunk_dir(chunk)
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)

        # Load chunk
        for i in range(5):
            # Deal with diagonal first
            ichunk = np.load(f"../{NPY_DIRECTORY}/X{i+1}",
                             mmap_mode="r")[start_second:end_second]

            # Hermitian so only need i to 5
            for j in range(i, 5):
                jchunk = np.load(f"../{NPY_DIRECTORY}/X{i+1}",
                                 mmap_mode="r")[start_second:end_second]

                spectra = calculate_spectra(ichunk, jchunk, tau_chunks)
                # 0Hz == 1Hz periodic boundary conditions
                spectra = np.append(spectra, spectra[0])

                np.save(f"{chunk_dir}/X{i+1}X{j+1}_{chunk}", spectra)
