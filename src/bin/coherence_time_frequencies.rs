use supermag_analysis::{
    constants::THRESHOLD,
    weights::{coherence_times, frequencies_from_coherence_times, FrequencyBin},
};

fn main() {
    let total_time = (365 * 24 * 60 * 60 * 18) + (5 * 24 * 60 * 60);
    let coherence_times: Vec<usize> = coherence_times(total_time as f64, THRESHOLD);

    let frequency_bins: Vec<FrequencyBin> = frequencies_from_coherence_times(&coherence_times);

    for (i, (bin, coherence_time)) in frequency_bins.iter().zip(coherence_times).enumerate() {
        println!(
            "frequency range for time {i} = {coherence_time} is {}..{} and has {} elements",
            *bin.multiples.start() as f64 * bin.lower,
            *bin.multiples.end() as f64 * bin.lower,
            bin.multiples.end() - bin.multiples.start() + 1,
        )
    }
}
