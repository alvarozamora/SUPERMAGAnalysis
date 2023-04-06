use nalgebra::Complex;
use ndarray::{s, Array1};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rustfft::FftPlanner;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use supermag_analysis::constants::THRESHOLD;
use supermag_analysis::theory::dark_photon::{
    DarkPhoton, DarkPhotonVec, DARK_PHOTON_NONZERO_ELEMENTS,
};
use supermag_analysis::theory::Theory;
use supermag_analysis::utils::approximate_sidereal;
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::weights::{
    coherence_times, frequencies_from_coherence_times, Analysis, ProjectionsComplete, Stationarity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Theory
    let theory = DarkPhoton::new();

    // Load in projections and auxiliary values
    let (projections_complete, auxiliary_complete) =
        read_dark_photon_projections_auxiliary().unwrap();

    // Get first component X1
    let first_series = projections_complete
        .projections_complete
        .get(&DARK_PHOTON_NONZERO_ELEMENTS[0])
        .unwrap();

    // Get subseries
    let check_coherence = 10_198_740_usize;
    let check_chunk = 20_usize;
    let start_chunk = check_chunk * check_coherence;
    let end_chunk = (check_chunk + 1) * check_coherence;
    let mut subseries: Array1<Complex<f32>> = first_series
        .slice(s![start_chunk..end_chunk])
        .map(|x| x.into());

    // Do FFT of X1
    let mut planner = FftPlanner::new();
    let fft_handler = planner.plan_fft_forward(check_coherence);
    fft_handler.process(subseries.as_slice_mut().unwrap());

    // Get relevant range
    let coherence_times = coherence_times(projections_complete.num_secs() as f64, THRESHOLD);
    let frequency_bin = frequencies_from_coherence_times(&coherence_times).remove(68);
    // let approx_sidereal = approximate_sidereal(&frequency_bin);

    println!(
        "size of series is {}, relevant multiples is {:?} -> freq range is {}..{}",
        subseries.len(),
        &frequency_bin.multiples,
        *frequency_bin.multiples.start() as f64 * frequency_bin.lower,
        *frequency_bin.multiples.end() as f64 * frequency_bin.lower,
    );

    for (window_index, element) in subseries
        .iter()
        .skip(*frequency_bin.multiples.start())
        .take(20)
        .enumerate()
    {
        let db_key = [check_coherence, check_chunk, window_index]
            .map(usize::to_le_bytes)
            .concat();
        let element_in_db = theory.data_vector.get(db_key)?.unwrap();
        let data_vector: DarkPhotonVec<f32> = bincode::deserialize(&element_in_db)?;
        println!(
            "|{element})|^2 = {} vs value in db {}",
            element.norm_sqr(),
            data_vector.mid[0].norm_sqr()
        );
    }

    Ok(())
}

fn read_dark_photon_projections_auxiliary() -> Result<
    (
        Arc<ProjectionsComplete>,
        Arc<<DarkPhoton as Theory>::AuxiliaryValue>,
    ),
    Box<dyn std::error::Error>,
> {
    // Open projections_complete and auxiliary_complete file
    let mut projections_file = File::open("projections_complete").expect("failed to open file");
    let mut auxiliary_file = File::open("auxiliary_complete").expect("failed to open file");

    // Initialize buffer for projections and auxiliary values
    let mut projection_buffer = Vec::new();
    let mut auxiliary_buffer = Vec::new();

    // Read bytes in files
    projections_file
        .read_to_end(&mut projection_buffer)
        .expect("failed to read projections");
    auxiliary_file
        .read_to_end(&mut auxiliary_buffer)
        .expect("failed to read auxiliary");

    // Deserialize bytes into respective types
    let projections_complete: Arc<ProjectionsComplete> = Arc::new(
        bincode::deserialize(&projection_buffer)
            .expect("failed to deserialize projections_complete"),
    );
    let auxiliary_complete: Arc<<DarkPhoton as Theory>::AuxiliaryValue> = Arc::new(
        bincode::deserialize(&auxiliary_buffer)
            .expect("failed to deserialize projections_complete"),
    );

    Ok((projections_complete, auxiliary_complete))
}
