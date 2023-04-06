use supermag_analysis::constants::DATA_DAYS;
use supermag_analysis::theory::dark_photon::DarkPhoton;
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::weights::{Analysis, Stationarity};

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()
        .unwrap();

    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    // Define stationarity time, Coherence time
    const STATIONARITY_TIME: Stationarity = Stationarity::Yearly;

    // Start Balancer
    let mut balancer = Balancer::new(12, 1);

    // Initialize Theory
    let theory = DarkPhoton::new();

    // Compute weights for this coherence time
    let complete_series_fut = Analysis::calculate_projections_and_auxiliary(
        STATIONARITY_TIME,
        theory,
        &mut balancer.manager,
    );
    balancer
        .runtime
        .block_on(complete_series_fut)
        .expect("failed to build the complete series");
}
