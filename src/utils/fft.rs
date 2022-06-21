

/// Given the frequency_target
pub fn find_nearest_frequency_1s(frequency_target: f32, total_time: f32) -> f32 {

    // Find fourier grid spacing
    let df: f32 = 1.0 / total_time;

    // Initialize multiple as floor of div
    let mut multiple: usize = (frequency_target / df) as usize; 

    // Initialize candidate frequencies
    let candidate_1 =  multiple as f32 * df;
    let candidate_2 = (multiple + 1) as f32 * df;

    // Return candidate which is closest to target
    if (candidate_1 - frequency_target).abs() < (candidate_2 - frequency_target).abs() {
        candidate_1
    } else {
        candidate_2
    }
}