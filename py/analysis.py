# This script calculates the likelihood pipeline on the projections and auxiliary values for the dark photon case
# 1) Data vector
# 1) Theory mean
# 3) Theory var
# 4) Calculate s&z
# 5) Likelihood
import numpy as np
from numpy import pi as PI, sqrt
from utils import coherence_times, frequencies_from_coherence_times, approximate_sidereal, FD, NPY_DIRECTORY
from power import STATIONARITY_TIME, SPECTRA_FREQS, TAU, find_overlap_chunks, get_stationarity_chunk_dir
from bound import calculate_bounds

RHO = 6.04e7
R = 0.0212751


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
        print(f"Working on coherence time {coh}")

        # Calculate number of exact chunk
        exact_chunks = TIME_2003_TO_2020 // coh
        # A last_chunk is a chunk which does not cover all of the coherence_time
        has_last_chunk = TIME_2003_TO_2020 % exact_chunks > 0
        # Calculate total chunks
        total_chunks = exact_chunks + int(has_last_chunk)

        # Calculate the approximate sidereal df multiple for this frequency bin
        approx_sidereal = approximate_sidereal(df)
        # and the frequency
        f_as = approx_sidereal * df

        # Start and end of the relevant frequency window (including sidereal pad)
        #
        # max(x, 0) is really only here for the first coh which has lof = 0
        # min(x, MAX) is really only here for the first coh which has hif = 1 / TIME_2003_TO_2020
        start_padded = max(lof - approx_sidereal, 0)
        end_padded_inclusive = min(
            hif_inclusive + approx_sidereal, TIME_2003_TO_2020-1)
        end_padded_exclusive = end_padded_inclusive + 1
        num_frequencies = end_padded_exclusive - start_padded - 2 * approx_sidereal
        coh_freqs = (lof + np.arange(num_frequencies)) * df

        # The s and z coherent elements are collected over all coherence time chunks
        s_chunks = np.zeros(total_chunks)
        z_chunks = np.zeros(total_chunks)
        for chunk in range(total_chunks):
            # STEP 1: DATA VECTOR
            # a) gather every x_i chunk
            # b) perform fft and get subseries
            # c) build data vector
            data_vector = np.zeros((15, num_frequencies)) + 0j

            for x in [1, 2, 3, 4, 5]:
                series = np.load(f"../{NPY_DIRECTORY}/X{x}", mmap_mode="r")

                print(f"X{x} chunk {chunk+1} of {total_chunks}")

                # Start and end of this chunk in the series
                start = chunk * coh
                end = min((chunk + 1) * coh, len(series))

                # Take FFT of series and then zoom in on relevant window
                subseries_fft = np.fft.fft(series[start:end])
                subseries_fft = subseries_fft[
                    start_padded:end_padded_exclusive]
                subseries_freq = np.fft.fftfreq(
                    end-start+1, d=1.0)[start_padded:end_padded_exclusive]  # d = 1 second spacing

                # Get data vector
                lo = subseries_fft[:-2*approx_sidereal]
                mid = subseries_fft[approx_sidereal:-approx_sidereal]
                hi = subseries_fft[2*approx_sidereal:]

                # Write to data_vector
                data_vector[x - 1] = lo
                data_vector[x - 1 + 5] = mid
                data_vector[x - 1 + 10] = hi

            # STEP 2: Theory Mean
            # 1) construct DFT kernels
            # 2) load H_i and add their DFT contributions to mux, muy, muz
            mux = np.zeros(15) + 0j
            muy = np.zeros(15) + 0j
            muz = np.zeros(15) + 0j

            # Construct DFT kernels
            # cis_f_fh is (cos - i sin)@(fdhat-fd)
            cis_fh_f = np.exp(
                -1j * 2 * np.pi * ((approx_sidereal * df) - FD) * np.arange(start, end))

            # cis_f is (cos - i sin)@(fd)
            cis_f = np.exp(-1j * 2 * np.pi * FD * np.arange(start, end))

            # cis_f_fh is (cos - i sin)@(fd-fdhat)
            cis_f_fh = np.exp(-1j * 2 * np.pi *
                              (FD - (approx_sidereal * df)) * np.arange(start, end))

            # Here we load the appropriate chunks for h
            h1 = np.load(f"../{NPY_DIRECTORY}/H1",
                         mmap_mode="r")[start:end]
            h2 = np.load(f"../{NPY_DIRECTORY}/H2",
                         mmap_mode="r")[start:end]
            h3 = np.load(f"../{NPY_DIRECTORY}/H3",
                         mmap_mode="r")[start:end]
            h4 = np.load(f"../{NPY_DIRECTORY}/H4",
                         mmap_mode="r")[start:end]
            h5 = np.load(f"../{NPY_DIRECTORY}/H5",
                         mmap_mode="r")[start:end]
            h6 = np.load(f"../{NPY_DIRECTORY}/H6",
                         mmap_mode="r")[start:end]
            h7 = np.load(f"../{NPY_DIRECTORY}/H7",
                         mmap_mode="r")[start:end]

            # Calculate mux's
            # first do f=fd-fdhat using cis_f_fh
            # FT(1 - H1 + iH2)
            # FT(H2 + iH1)
            # FT(H4 - iH5)
            # FT(-H5 + i(H3-H4))
            # FT(H6 - iH7)
            mux[0] = np.sum(cis_f_fh * (1 - h1 + 1j*h2))
            mux[1] = np.sum(cis_f_fh * (h2 + 1j*h1))
            mux[2] = np.sum(cis_f_fh * (h4 - 1j*h5))
            mux[3] = np.sum(cis_f_fh * (-h5 + 1j*(h3 - h4)))
            mux[4] = np.sum(cis_f_fh * (h6 - 1j*h7))

            # then do f=fd using cis_f
            # Real(FT(2*(1-H1)) - Im(FT(2*H2))
            # Real(FT(2*H2)) - Im(FT(2*H1))
            # Real(FT(2*H4)) + Im(FT(2*H5))
            # Real(FT(-2*H5)) - Im(FT(2*(H3-H4)))
            # Real(FT(2*H6)) + Im(FT(2*H7))
            mux[5] = 2 * (np.sum(cis_f * (1 - h1)).real -
                          np.sum(cis_f * h2).imag)
            mux[6] = 2 * (np.sum(cis_f * h2).real -
                          np.sum(cis_f * h1).imag)
            mux[7] = 2 * (np.sum(cis_f * h4).real +
                          np.sum(cis_f * h5).imag)
            mux[8] = 2 * (np.sum(cis_f * -h5).real -
                          np.sum(cis_f * (h3 - h4)).imag)
            mux[9] = 2 * (np.sum(cis_f * h6).real +
                          np.sum(cis_f * h7).imag)

            # then do f=fdhat-fd using cis_fh_f
            # FT(1 - H1 - iH2)
            # FT(H2 - iH1)
            # FT(H4 + iH5)
            # FT(-H5 + i*(H4 - H3))
            # FT(H6 + iH7)
            mux[10] = np.sum(cis_fh_f * (1 - h1 - 1j*h2))
            mux[11] = np.sum(cis_fh_f * (h2 - 1j*h1))
            mux[12] = np.sum(cis_fh_f * (h4 + 1j*h5))
            mux[13] = np.sum(cis_fh_f * (-h5 + 1j*(h4 - h3)))
            mux[14] = np.sum(cis_fh_f * (h6 + 1j*h7))

            # Calculate muy's
            # first do f=fd-fdhat using cis_f_fh
            # FT(H2 + i*(H1-1))
            # FT(H1 - iH2)
            # FT(-H5 - iH4)
            # FT(H3 - H4 + iH5)
            # FT(-H7 - iH6)
            muy[0] = np.sum(cis_f_fh * (h2 + 1j*(h1 - 1)))
            muy[1] = np.sum(cis_f_fh * (h1 - 1j*h2))
            muy[2] = np.sum(cis_f_fh * (-h5 - 1j*h4))
            muy[3] = np.sum(cis_f_fh * (h3 - h4 + 1j*h5))
            muy[4] = np.sum(cis_f_fh * (-h7 - 1j*h6))

            # then do f=fd using cis_f
            # 2*Re(FT(H2)) + 2*Im(FT(1-H1))
            # 2*Re(FT(H1)) + Im(FT(H2))
            # -2*Re(FT(H5)) + 2*Im(FT(H4))
            # 2*Re(FT(H3-H4)) - 2*Im(FT(H5))
            # -2*Re(FT(H7)) + 2*Im(FT(H6))
            muy[5] = 2 * (np.sum(cis_f * h2).real +
                          np.sum(cis_f * (1 - h1)).imag)
            muy[6] = 2 * (np.sum(cis_f * h1).real +
                          np.sum(cis_f * h2).imag)
            muy[7] = 2 * (-np.sum(cis_f * h5).real +
                          np.sum(cis_f * h4).imag)
            muy[8] = 2 * (np.sum(cis_f * (h3 - h4)).real -
                          np.sum(cis_f * h5).imag)
            muy[9] = 2 * (np.sum(cis_f * -h7).real +
                          np.sum(cis_f * h6).imag)

            # then do f=fdhat-fd using cis_fh_f
            # FT(H2 + i*(1-H1))
            # FT(H1 + iH2)
            # FT(-H5 + iH4)
            # FT(H3-H4-iH5)
            # FT(-H7 + iH6)
            muy[10] = np.sum(cis_fh_f * (h2 + 1j*(1-h1)))
            muy[11] = np.sum(cis_fh_f * (h1 + 1j*h2))
            muy[12] = np.sum(cis_fh_f * (-h5 + 1j*h4))
            muy[13] = np.sum(cis_fh_f * (h3 - h4 - 1j*h5))
            muy[14] = np.sum(cis_fh_f * (-h7 + 1j*h6))

            # Calculate muz's
            #
            # need different DFT kernels
            cis_mfh = np.exp(
                1j * 2 * np.pi * approx_sidereal * df * np.arange(start, end))
            cis_fh = np.exp(
                -1j * 2 * np.pi * approx_sidereal * df * np.arange(start, end))

            # mu0, mu1, mu5, mu6, mu10, mu11 are all zero
            muz[0] = muz[1] = muz[5] = muz[6] = muz[10] = muz[11] = 0.0

            # FTs at f=-fdhat
            # FT(h6)
            # FT(-h7)
            # FT(1-h3)
            muz[2] = np.sum(cis_mfh * h6)
            muz[3] = -np.sum(cis_mfh * h7)
            muz[4] = np.sum(cis_mfh * (1 - h3))

            # FTs at f=0
            # FT(h6)
            # FT(-h7)
            # FT(1-h3)
            muz[7] = np.sum(h6)
            muz[8] = -np.sum(h7)
            muz[9] = np.sum(1 - h3)

            # FTs at f = fdhat
            # FT(h6)
            # FT(-h7)
            # FT(1-h3)
            muz[12] = np.sum(cis_fh * h6)
            muz[13] = -np.sum(cis_fh * h7)
            muz[14] = np.sum(cis_fh * (1 - h3))

            # Multiply by prefactors
            mux_prefactor = PI * R * sqrt(2.0 * RHO) / 4.0
            muy_prefactor = - mux_prefactor
            muz_prefactor = - 2 * mux_prefactor
            mux *= mux_prefactor
            muy *= muy_prefactor
            muz *= muz_prefactor

            # Package into one array
            mu = np.ascontiguousarray(np.array([mux, muy, muz]))
            del mux, muy, muz

            # STEP 3: Theory Var
            # 1) For this coherence time/chunk, what stationarity times does this overlap with?
            # 2) Calculate overlap
            # 3) Load the spectra for those stationarity times
            # 4)
            #   - (4.x.0) Interpolate to correct frequencies
            #   - (4.x.1) add overlap * interpolated_power to sigma
            #
            #   This is done for f = fc-fdh, fc, fc+fdh, which is x=0,1,2 (e.g. 4.0.0)

            # NOTE: This is initialized this way so that each of the 5x5 matrices are contiguous in memory
            # I don't like that this is a sparse matrix but it shouldn't be too bad...
            sigma = np.zeros((15, 15, num_frequencies))
            # sigma = np.zeros((5, 5, num_frequencies + 2*approx_sidereal)) # Potential change later

            # find_overlap_chunks does
            # 1) Find overlapping stationarity times
            # 2) Calculate overlap
            overlapping_stationarity_times = find_overlap_chunks(coh, chunk)
            for i in range(5):
                for j in range(i, 5):
                    for (stationarity_chunk, overlap) in overlapping_stationarity_times:

                        # 3) Load the spectra for this stationarity time for this chunk
                        ijchunk_dir = get_stationarity_chunk_dir(
                            stationarity_chunk)
                        ijchunk = np.load(
                            f"{ijchunk_dir}/X{i+1}X{j+1}_{stationarity_chunk:05d}")

                        # 4.0.0) Interpolate to correct frequencies (one sidereal day to the left)
                        # TODO: confirm modulus
                        interpolated_power = np.interp(
                            (coh_freqs - approx_sidereal) % 1.0, SPECTRA_FREQS, ijchunk)

                        # 4.0.1) Add overlap * interpolated_power to sigma
                        if i == j:
                            sigma[i, i] += overlap * interpolated_power
                        else:
                            # TODO: check conjugation
                            sigma[i, j] += overlap * interpolated_power
                            sigma[j, i] += overlap * \
                                np.conj(interpolated_power)

                        # 4.1.0) Interpolate to correct frequencies
                        interpolated_power = np.interp(
                            coh_freqs, SPECTRA_FREQS, ijchunk)

                        # 4.1.1) Add overlap * interpolated_power to sigma
                        if i == j:
                            sigma[i+5, i+5] += overlap * interpolated_power
                        else:
                            # TODO: check conjugation
                            sigma[i+5, j+5] += overlap * interpolated_power
                            sigma[j+5, i+5] += overlap * \
                                np.conj(interpolated_power)

                        # 4.2.0) Interpolate to correct frequencies (one sidereal day to the left)
                        # TODO: confirm modulus
                        interpolated_power = np.interp(
                            (coh_freqs + approx_sidereal) % 1.0, SPECTRA_FREQS, ijchunk)

                        # 4.2.1) Add overlap * interpolated_power to sigma
                        if i == j:
                            sigma[i+10, i+10] += overlap * interpolated_power
                        else:
                            # TODO: check conjugation
                            sigma[i+10, j+10] += overlap * interpolated_power
                            sigma[j+10, i+10] += overlap * \
                                np.conj(interpolated_power)

            # STEP 4: Calculate s and z for each chunk of this coherence time to obtain likelihood
            # 1) Carry out Cholesky decomoposition on Sigma_k = A_k * Adag_k, obtaining A_k
            # 2) Invert A_k, obtaining Ainv_k
            # 3) Calculate Y_k = Ainv_k * X_k
            # 4) Calculate nu_ik = Ainv_k * mu_ik
            # 5) SVD into Nk = nu_ik -> U_k * S_k * Vdag_k, obtaining U_k
            # 6) Calculate Z_k = Udag_k * Y_k

            # Switch sigma to be contiguous for linalg
            sigma = np.ascontiguousarray(np.transpose(sigma, (2, 0, 1)))
            assert sigma.shape == (num_frequencies, 15, 15)

            # 1) Carry out Cholesky decomoposition on Sigma_k = A_k * Adag_k, obtaining A_k
            # 2) Invert A_k, obtaining Ainv_k
            a_inv = np.linalg.inv(np.linalg.cholesky(sigma))
            assert nu.shape == (num_frequencies, 5, 5)

            # 3) Calculate Y_k = Ainv_k * X_k
            yk = a_inv @ data_vector

            # 4) Calculate nu_ik = Ainv_k * mu_ik
            nu = a_inv @ mu
            assert nu.shape == (num_frequencies, 15, 3)

            # 5) SVD into Nk = nu_ik -> U_k * S_k * Vdag_k, obtaining U_k
            u, s, vh = np.linalg.svd(nu, full_matrices=False)
            assert u.shape == (num_frequencies, 15, 3)
            assert s.shape == (num_frequencies, 3)

            # 6) Calculate Z_k = Udag_k * Y_k
            udag = np.conj(np.transpose(u, (-1, -2)))
            assert udag.shape == (num_frequencies, 3, 15)
            z = udag * yk
            assert z.shape == (num_frequencies, 3)

            # append
            s.append(s)
            z.append(z)
        # Here we swap so that the "independent" measurements across different coherence times
        # are contiguous in memory
        s = np.ascontiguousarray(np.transpose(np.array(s), (0, 1)))
        z = np.ascontiguousarray(np.transpose(np.array(z), (0, 1)))
        assert s.shape == (num_frequencies, total_chunks, 3)
        assert z.shape == (num_frequencies, total_chunks, 3)

        # STEP 5: Calculate likelihood -ln Lk = |Z_k - eps * S_k * d_k|^2
        bound = calculate_bounds(coh, coh_freqs, s, z)
