import unittest
from utils import maxprime, coherence_times, frequencies_from_coherence_times, approximate_sidereal, SIDEREAL_DAY_SECONDS, calculate_stationarity_chunks


class TestMaxPrime(unittest.TestCase):

    def test_maxprime(self):
        self.assertEqual(maxprime(1), 1)
        self.assertEqual(maxprime(2), 2)
        self.assertEqual(maxprime(3), 3)
        self.assertEqual(maxprime(4), 2)
        self.assertEqual(maxprime(5), 5)
        self.assertEqual(maxprime(15), 5)
        self.assertEqual(maxprime(100), 5)
        self.assertEqual(maxprime(101), 101)


class TestFrequencyBins(unittest.TestCase):

    def test_frequency_bins(self):
        EIGHTEEN_YEARS = (18 * 365 * 24 * 60 * 60) + (5 * 24 * 60 * 60)
        times = coherence_times(EIGHTEEN_YEARS)
        freqs = frequencies_from_coherence_times(times)
        last = None
        for (t, f) in zip(times, freqs):
            elements = (f[1]-f[0])+1
            # last coherence time, last first multiple, last last multiple, last size
            last = (t, f[0], f[1], elements)

        # This is compared to the output from the rust code
        LAST_COH = 1016821
        LAST_LOW = 970873
        LAST_LAST = 1016820  # Note this is inclusive!
        LAST_SIZE = 45948
        self.assertEqual(last, (LAST_COH, LAST_LOW, LAST_LAST, LAST_SIZE))


class TestSidereal(unittest.TestCase):

    def test_approx_sidereal(self):
        # Checking that (1 / (SIDEREAL_DAY_SECONDS +/-1) ≈ 1 / SIDEREAL_DAY_SECONDS
        self.assertEqual(approximate_sidereal(
            1.0 / (SIDEREAL_DAY_SECONDS + 1)), 1)
        self.assertEqual(approximate_sidereal(
            1.0 / (SIDEREAL_DAY_SECONDS - 1)), 1)

        # Checking that i * (1 / (SIDEREAL_DAY_SECONDS +/-1) / i) ≈ 1 / SIDEREAL_DAY_SECONDS
        for i in range(2, 10):
            self.assertEqual(approximate_sidereal(
                1.0 / (SIDEREAL_DAY_SECONDS + 1) / i), i)
            self.assertEqual(approximate_sidereal(
                1.0 / (SIDEREAL_DAY_SECONDS - 1) / i), i)


class TestStationarityChunks(unittest.TestCase):

    def test_stationarity_chunks_neatly_divisible(self):
        TEST_TAU = 8
        TEST_TIME = 32
        num_chunks = TEST_TIME // TEST_TAU
        chunk_size = TEST_TIME // num_chunks
        chunk_mod = TEST_TIME % num_chunks

        # Neatly divisible into TAU chunks
        expected = [
            (0, 8),  # 8
            (8, 16),  # 8
            (16, 24),  # 8
            (24, 32),  # 8
        ]

        result = calculate_stationarity_chunks(
            num_chunks, chunk_size, chunk_mod)

        assert result == expected

    def test_stationarity_chunks_even_extra(self):
        TEST_TAU = 8
        TEST_TIME = 36
        num_chunks = TEST_TIME // TEST_TAU
        chunk_size = TEST_TIME // num_chunks
        chunk_mod = TEST_TIME % num_chunks

        # Not neatly divisible into TAU chunks
        # These should all have one extra over neatly divisible case
        expected = [
            (0, 9),  # 9
            (9, 18),  # 9
            (18, 27),  # 9
            (27, 36),  # 9
        ]

        result = calculate_stationarity_chunks(
            num_chunks, chunk_size, chunk_mod)

        assert result == expected, "failed even extra case"

    def test_stationarity_chunks_uneven_extra(self):
        TEST_TAU = 8
        TEST_TIME = 38
        num_chunks = TEST_TIME // TEST_TAU
        chunk_size = TEST_TIME // num_chunks
        chunk_mod = TEST_TIME % num_chunks

        # Not neatly divisible into TAU chunks
        # First two should have a two extra
        # Last two should have one extra
        expected = [
            (0, 10),  # 10
            (10, 20),  # 10
            (20, 29),  # 9
            (29, 38),  # 9
        ]

        result = calculate_stationarity_chunks(
            num_chunks, chunk_size, chunk_mod)

        assert result == expected, "failed uneven extra case"


if __name__ == '__main__':
    unittest.main()
