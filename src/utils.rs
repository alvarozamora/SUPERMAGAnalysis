pub mod coordinates;
pub mod loader;
// pub mod balancer;
pub mod async_balancer;
pub mod fft;
pub mod vec_sph;

use std::{collections::HashSet, ops::Add};

use rayon::iter::IntoParallelIterator;

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

    return longest_streak
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
