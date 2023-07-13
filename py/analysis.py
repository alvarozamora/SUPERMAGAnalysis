import numpy as np
from numpy import pi as PI, sqrt
from utils import coherence_times, frequencies_from_coherence_times, approximate_sidereal, FD

RHO = 6.04e7
R = 0.0212751
NPY_DIRECTORY = "proj_aux_np"


def log_memory():
    import psutil
    process = psutil.Process()
    print(f"{process.memory_info().rss:,}")  # in bytes


if __name__ == "__main__":
    # Calculate coherence times and their respective frequency bins
    LEAP_DAYS = 5
    TIME_2003_TO_2020 = 18 * 365 * 24 * 60 * 60 + LEAP_DAYS * 24 * 60 * 60
    # calculate s and z for each chunk of this coherence time
    s_and_z = []
    coherence_times = coherence_times(TIME_2003_TO_2020)
    freq_bins = frequencies_from_coherence_times(coherence_times)

    for (coh, (lof, hif_inclusive, df)) in zip(coherence_times, freq_bins):
        print(f"Working on coherence time {coh}")

        # Calculate number of exact chunk
        exact_chunks = TIME_2003_TO_2020 // coh
        # A last_chunk is a chunk which does not cover all of the coherence_time
        has_last_chunk = TIME_2003_TO_2020 % exact_chunks > 0
        # Calculate total chunks
        total_chunks = exact_chunks + (1 if has_last_chunk else 0)

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
            mux_prefactor = PI * R * sqrt(2.0 * RHO) / 4.0
            muy = np.zeros(15) + 0j
            muz = np.zeros(15) + 0j

            # Construct DFT kernels
            # cis_f_fh is (cos + i sin)@(fdhat-fd)
            cis_fh_f = np.exp(
                1j * 2 * np.pi * ((approx_sidereal * df) - FD) * np.arange(start, end))

            # cis_f is (cos + i sin)@(fd)
            cis_f = np.exp(1j * 2 * np.pi * FD * np.arange(start, end))

            # cis_f_fh is (cos + i sin)@(fd-fdhat)
            cis_f_fh = np.exp(-1j * 2 * np.pi *
                              ((approx_sidereal * df) - FD) * np.arange(start, end))

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
            # Real(FT(2*(1-H1)) + Im(FT(2*H2))
            # Real(FT(2*H2)) + Im(FT(2*H1))
            # Real(FT(2*H4)) - Im(FT(2*H5))
            # Real(FT(-2*H5)) + Im(FT(2*(H3-H4)))
            # Real(FT(2*H6)) - Im(FT(2*H7))
            mux[5] = 2 * (np.sum(cis_f * (1 - h1)).real +
                          np.sum(cis_f * h2).imag)
            mux[6] = 2 * (np.sum(cis_f * h2).real +
                          np.sum(cis_f * h1).imag)
            mux[7] = 2 * (np.sum(cis_f * h4).real -
                          np.sum(cis_f * h5).imag)
            mux[8] = 2 * (np.sum(cis_f * -h5).real +
                          np.sum(cis_f * (h3 - h4)).imag)
            mux[9] = 2 * (np.sum(cis_f * h6).real +
                          np.sum(cis_f * -h7).imag)

            # then do f=fdhat-fd using cis_fh_f
            # FT(1 - H1 - iH2) at f = fdhat-fd
            # FT(H2 - iH1) at f = fdhat-fd
            # FT(H4 + iH5) at f = fdhat-fd
            # FT(-H5 + i*(H4 - H3)) at f = fdhat-fd
            # FT(H6 + iH7) at f = fdhat-fd
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
            # 2*Re(FT(H2)) + 2*Im(FT(H1-1)) at f = fd
            # 2*Re(FT(H1)) - Im(FT(H2)) at f = fd
            # -2*Re(FT(H5)) - 2*Im(FT(H4)) at f = fd
            # 2*Re(FT(H3-H4)) + 2*Im(FT(H5)) at f = fd
            # -2*Re(FT(H7)) - 2*Im(FT(H6))
            muy[5] = 2 * (np.sum(cis_f * h2).real +
                          np.sum(cis_f * (h1 - 1)).imag)
            muy[6] = 2 * (np.sum(cis_f * h1).real -
                          np.sum(cis_f * h2).imag)
            muy[7] = 2 * (-np.sum(cis_f * h5).real -
                          np.sum(cis_f * h4).imag)
            muy[8] = 2 * (np.sum(cis_f * (h3 - h4)).real +
                          np.sum(cis_f * h5).imag)
            muy[9] = 2 * (np.sum(cis_f * -h7).real -
                          np.sum(cis_f * -h6).imag)

            # then do f=fdhat-fd using cis_fh_f
            # FT(H2 + i*(1-H1)) at f = fdhat - fd
            # FT(H1 + iH2) at f = fdhat - fd
            # FT(-H5 + iH4) at f = fdhat-fd
            # FT(H3-H4-iH5) at f = fdhat-fd
            # FT(-H7 + iH6) at f = fdhat-fd
            muy[10] = np.sum(cis_fh_f * (h2 + 1j*(1-h1)))
            muy[11] = np.sum(cis_fh_f * (h1 + 1j*h2))
            muy[12] = np.sum(cis_fh_f * (-h5 + 1j*h4))
            muy[13] = np.sum(cis_fh_f * (h3 - h4 - 1j*h5))
            muy[14] = np.sum(cis_fh_f * (-h7 + 1j*h6))

            # Calculate muz's
            #
            # need different DFT kernels
            cis_mfh = np.exp(
                -1j * 2 * np.pi * approx_sidereal * df * np.arange(start, end))
            cis_fh = np.exp(
                1j * 2 * np.pi * approx_sidereal * df * np.arange(start, end))

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
            muz[8] = np.sum(h7)
            muz[9] = np.sum(1 - h3)

            # FTs at f = fdhat
            # FT(h6)
            # FT(-h7)
            # FT(1-h3)
            muz[12] = np.sum(cis_fh * h6)
            muz[13] = -np.sum(cis_fh * h7)
            muz[14] = np.sum(cis_fh * (1 - h3))

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
