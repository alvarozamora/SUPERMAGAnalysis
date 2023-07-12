import numpy as np
from utils import coherence_times, frequencies_from_coherence_times, approximate_sidereal


def log_memory():
    import psutil
    process = psutil.Process()
    print(f"{process.memory_info().rss:,}")  # in bytes


if __name__ == "__main__":
    # Calculate coherence times and their respective frequency bins
    LEAP_DAYS = 5
    TIME_2003_TO_2020 = 18 * 365 * 24 * 60 * 60 + LEAP_DAYS * 24 * 60 * 60
    coherence_times = coherence_times(TIME_2003_TO_2020)
    freq_bins = frequencies_from_coherence_times(coherence_times)

    for (coh, (lof, hif_inclusive, df)) in zip(coherence_times, freq_bins):

        # Calculate number of exact chunk
        exact_chunks = TIME_2003_TO_2020 // coh
        # a last_chunk is a chunk which does not cover all of the coherence_time
        has_last_chunk = TIME_2003_TO_2020 % exact_chunks > 0

        # Calculate the approximate sidereal df multiple for this frequency bin
        approx_sidereal = approximate_sidereal(df)
        # and the frequency
        f_as = approx_sidereal * df

        s_and_z = []
        for x in [1, 2, 3, 4, 5]:
            series = np.load(f"../proj_aux_np/X{x}", mmap_mode="r")

            # calculate s and z for each chunk of this coherence time
            # data_vector = TODO
            for chunk in range(exact_chunks):
                print(f"X{x} chunk {chunk+1} of {exact_chunks}")

                # STEP 1: DATA VECTOR
                # a) gather chunk
                # b) perform fft and get subseries
                # c) build data vector

                # Start and end of this chunk in the series
                start = chunk * coh
                end = (chunk + 1) * coh

                # Start and end of the relevant frequency window (including sidereal pad)
                #
                # max(x, 0) is really only here for the first coh which has lof = 0
                start_padded = max(lof - approx_sidereal, 0)
                end_padded_inclusive = hif_inclusive + approx_sidereal
                end_padded_exclusive = end_padded_inclusive + 1
                assert end_padded_exclusive < len(series), "out of bounds"

                # Take FFT of series and then zoom in on relevant window
                subseries_fft = np.fft.fft(series[start:end])[
                    start_padded:end_padded_exclusive]
                subseries_freq = np.fft.fftfreq(
                    end-start+1, d=1.0)[start_padded:end_padded_exclusive]  # d = 1 second spacing

                # Get data vector
                lo = subseries_fft[:-2*approx_sidereal]
                mid = subseries_fft[approx_sidereal:-approx_sidereal]
                hi = subseries_fft[2*approx_sidereal:]
                print(
                    f"sizes {len(lo):,} {len(mid):,} {len(hi):,}")

                # STEP 2: Theory Mean
                # a) gather relevant h's
                # b) write out mux, muy, muz

                # STEP 3: Theory Var
                # a) Load in overlapping stationarity times
                # b)

                # STEP 4:
                # 1) Carry out Cholesky decomoposition on Sigma_k = A_k * Adag_k, obtaining A_k
                # 2) Invert A_k, obtaining Ainv_k
                # 3) Calculate Y_k = Ainv_k * X_k
                # 4) Calculate nu_ik = Ainv_k * mu_ik
                # 5) SVD into Nk = nu_ik -> U_k * S_k * Vdag_k, obtaining U_k
                # 6) Calculate Zk = Udag_k * Y_k
                # 7) Calculate likelihood -ln Lk = |Z_k - eps * S_k * d_k|^2
