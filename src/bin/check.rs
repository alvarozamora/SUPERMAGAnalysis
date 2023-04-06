use supermag_analysis::{
    constants::THRESHOLD,
    utils::{approximate_sidereal, loader::day_since_first, sec_to_year},
    weights::{coherence_times, frequencies_from_coherence_times, FrequencyBin},
};

fn main() {
    let total_time = (365 * 24 * 60 * 60 * 18) + (5 * 24 * 60 * 60);
    let coherence_times: Vec<usize> = coherence_times(total_time as f64, THRESHOLD);

    let frequency_bins: Vec<FrequencyBin> = frequencies_from_coherence_times(&coherence_times);

    for (i, (bin, coherence_time)) in frequency_bins.iter().zip(coherence_times).enumerate() {
        if i == 68 {
            let approx_sidereal: usize = approximate_sidereal(&bin);
            let check_coh = coherence_time;
            let check_chunk = 20;
            let check_window = 20;

            // let check_frequency_multiple = bin.multiples.clone().skip(check_window).next().unwrap();
            // let check_frequency = bin.lower * check_frequency_multiple as f64;

            // which years and how much overlap
            let start_sec_2003 = day_since_first(0, 2003) * 24 * 60 * 60;

            let start_chunk_from_2003 = check_coh * check_chunk;
            let end_chunk_from_2003_inclusive = check_coh * (check_chunk + 1) - 1;

            let (start_chunk_day, start_chunk_year) =
                sec_to_year(start_chunk_from_2003 + start_sec_2003);
            let (end_chunk_day, end_chunk_year) =
                sec_to_year(end_chunk_from_2003_inclusive + start_sec_2003);

            println!(
                "20th 68 coh time chunk starts on day {start_chunk_day} of {start_chunk_year}"
            );
            println!("20th 68 coh time chunk ends on day {end_chunk_day} of {end_chunk_year}");
        }
    }
}
