use supermag_analysis::constants::{DATA_DAYS, NUM_LEAP_YEARS};
use supermag_analysis::theory::dark_photon::DarkPhoton;
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::utils::loader::day_since_first;
use supermag_analysis::weights::{Analysis, Coherence, Stationarity};

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Debug);

    // Define stationarity time, Coherence time
    const STATIONARITY_TIME: Stationarity = Stationarity::Yearly;
    // let coherence = Coherence::Days(1); // TODO: stationarity

    // Which subset of the data to use
    let days_to_use = DATA_DAYS;
    // let days_to_use = day_since_first(0, 2004)..day_since_first(0, 2020);

    // Start Balancer
    let mut balancer = Balancer::new(32, 2);

    // Initialize Theory
    let theory = DarkPhoton::initialize(1.0);

    // Compute weights for this coherence time
    let complete_series_fut = Analysis::calculate_projections_and_auxiliary(
        STATIONARITY_TIME,
        theory,
        Some(days_to_use),
        &mut balancer.manager,
    );
    balancer
        .runtime
        .block_on(complete_series_fut)
        .expect("failed to build the complete series");
}
