// /// Given the frequency_target
// fn find_nearest_frequency_1s(frequency_target: f64, total_time: f64) -> f64 {

//     // Find fourier grid spacing
//     let df: f64 = 1.0 / total_time;

//     // Initialize multiple as floor of div
//     let multiple: usize = (frequency_target / df) as usize;

//     // Initialize candidate frequencies
//     let candidate_1 =  multiple as f64 * df;
//     let candidate_2 = (multiple + 1) as f64 * df;

//     // Return candidate which is closest to target
//     if (candidate_1 - frequency_target).abs() < (candidate_2 - frequency_target).abs() {
//         candidate_1
//     } else {
//         candidate_2
//     }
// }

use crate::FloatType;

pub(crate) fn get_frequency_range_1s(domain_size_in_seconds: usize) -> Vec<FloatType> {
    // hardcoded 1s spacing
    const ONE_SECOND_SPACING: FloatType = 1.0;

    // Calculate foruier grid spacing
    let fourier_grid_spacing: FloatType =
        1.0 / (domain_size_in_seconds as FloatType * ONE_SECOND_SPACING);

    // Calculate integer multiples of fourier grid spacing
    (0..domain_size_in_seconds)
        .map(|index| index as FloatType * fourier_grid_spacing)
        .collect()
}
