pub mod coordinates;
pub mod loader;
// pub mod balancer;
pub mod async_balancer;
pub mod fft;
pub mod igrf_decl;
pub mod io;
pub mod svd;
pub mod vec_sph;

use crate::constants::*;
use crate::weights::FrequencyBin;

use std::collections::HashSet;

use self::loader::day_since_first;

/// Given some `usize` collection, find the largest contiguous subset.
pub fn get_largest_contiguous_subset(set: &[usize]) -> (usize, usize) {
    // First construct `HashSet` for O(1) lookup
    let mut hash_set = HashSet::new();
    for &item in set {
        hash_set.insert(item);
    }

    // Initialize longest streak (size, starting value)
    let mut longest_streak = (0, 0);

    for &num in hash_set.iter() {
        // If this is the smallest possible number or smallest number in a contiguous subset,
        // get length of subset
        if num == 0 || !hash_set.contains(&num.saturating_sub(1)) {
            // Initialize streak counter state
            let mut current_num = num;
            let mut current_streak = 1;

            // Find length of streak
            while hash_set.contains(&(current_num + 1)) {
                current_num += 1;
                current_streak += 1;
            }

            if current_streak > longest_streak.0 {
                // Overwrite longest streak if we found a longer streak
                longest_streak = (current_streak, num);
            } else if current_streak == longest_streak.0 && num > longest_streak.1 {
                // Overwrite with later streak if we found an equally long streak
                longest_streak = (current_streak, num);
            }
        }
    }

    return longest_streak;
}

/// Given a number `n`, this function finds its largest prime factor
pub fn maxprime(n: usize) -> usize {
    // Deal with base case
    if n == 1 {
        return 1;
    }

    // Find upper_bound for checks
    let upper_bound = (n as f64).sqrt() as usize;

    // Iterate through all odd numbers between 2 and the upper_bound
    for i in (2..=2).chain((3..=upper_bound).step_by(2)) {
        if n % i == 0 {
            return maxprime(n / i);
        }
    }

    // Because we are iterating up, this will return the largest prime factor
    return n;
}

pub fn approximate_sidereal(frequency_bin: &FrequencyBin) -> usize {
    // Spacing
    let df = frequency_bin.lower;

    // Initial guess
    let multiple = ((SIDEREAL_DAY_SECONDS.recip()) / df) as usize;

    // Check guess and guess + 1
    let candidate_1 = (
        ((multiple as f64 * df) - SIDEREAL_DAY_SECONDS.recip()).abs(),
        multiple,
    );
    let candidate_2 = (
        (((multiple + 1) as f64 * df) - SIDEREAL_DAY_SECONDS.recip()).abs(),
        multiple + 1,
    );

    // Return whichever is closest to SIDEREAL_DAY_SECONDS
    if candidate_1.0 < candidate_2.0 {
        candidate_1.1
    } else {
        candidate_2.1
    }
}

pub fn approximate_frequency(frequency_bin: &FrequencyBin, target_frequency: f64) -> usize {
    // Spacing
    let df = frequency_bin.lower;

    // Initial guess
    let multiple = (target_frequency / df) as usize;

    // Check guess and guess + 1
    let candidate_1 = (((multiple as f64 * df) - target_frequency).abs(), multiple);
    let candidate_2 = (
        (((multiple + 1) as f64 * df) - target_frequency).abs(),
        multiple + 1,
    );

    // Return whichever is closest to SIDEREAL_DAY_SECONDS
    if candidate_1.0 < candidate_2.0 {
        candidate_1.1
    } else {
        candidate_2.1
    }
}

pub fn sec_to_year(sec: usize) -> (usize, usize) {
    let num_days = sec / 24 / 60 / 60;
    let mut year = 1998;
    let mut day = 0;
    while num_days != day_since_first(day, year) {
        if leap_year(year) && day == 365 {
            day = 0;
            year += 1;
        } else if day == 364 {
            day = 0;
            year += 1;
        } else {
            day += 1;
        }
    }
    (day, year)
}

/// not exact definition but works for this year range
fn leap_year(year: usize) -> bool {
    year % 4 == 0
}

#[test]
fn test_approximate_sidereal_candidate1() {
    let frequency_bin = FrequencyBin {
        lower: (SIDEREAL_DAY_SECONDS - 1.0).recip(),
        multiples: 1..=10,
    };

    assert_eq!(approximate_sidereal(&frequency_bin), 1)
}

#[test]
fn test_approximate_sidereal_candidate2() {
    let frequency_bin = FrequencyBin {
        lower: (SIDEREAL_DAY_SECONDS + 1.0).recip(),
        multiples: 1..=10,
    };

    assert_eq!(approximate_sidereal(&frequency_bin), 1)
}

#[test]
fn test_get_longest_streak_when_all_is_streak() {
    let set = [3, 1, 2, 4, 6, 5];
    let longest_streak = get_largest_contiguous_subset(&set);
    assert_eq!(longest_streak, (6, 1))
}

#[test]
fn test_get_longest_streak_with_one_off() {
    let set = [3, 1, 2, 4, 6, 5, 12];
    let longest_streak = get_largest_contiguous_subset(&set);
    assert_eq!(longest_streak, (6, 1))
}

#[test]
fn test_get_longest_and_latest_streak() {
    let set = [1, 2, 3, 6, 7, 8];
    let longest_streak = get_largest_contiguous_subset(&set);
    assert_eq!(longest_streak, (3, 6))
}

#[test]
fn test_get_longest_and_latest_streak_single() {
    let set = [1, 3, 5, 7, 9];
    let longest_streak = get_largest_contiguous_subset(&set);
    assert_eq!(longest_streak, (1, 9))
}

#[test]
fn test_maxprime_small_prime() {
    assert_eq!(maxprime(7), 7);
}

#[test]
fn test_maxprime_bigger_prime() {
    assert_eq!(maxprime(53), 53);
}

#[test]
fn test_maxprime_prime_squared() {
    assert_eq!(maxprime(49), 7);
}

#[test]
fn test_maxprime_multiple_primes() {
    assert_eq!(maxprime(24), 3);
}

#[test]
fn test_maxprime_multiple_primes_prime_bigger_number() {
    assert_eq!(maxprime(150), 5);
}

#[test]
#[ignore] /* u32 max is O(4 billion), so this takes a long time */
fn test_f64_u32_casting() {
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    (0..std::u32::MAX).into_par_iter().for_each(|i| {
        assert_eq!(i, (i as f64 + 0.1) as u32);
    });
}

#[test]
#[should_panic] /* this is why we need f64s for u32 casting */
fn test_f32_u32_casting() {
    for i in 0..std::u32::MAX {
        assert_eq!(i, (i as f32 + 0.1) as u32);
    }
}
