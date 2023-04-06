use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use supermag_analysis::theory::dark_photon::{read_dark_photon_projections_auxiliary, DarkPhoton};
use supermag_analysis::theory::Theory;
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::weights::{Analysis, ProjectionsComplete, Stationarity};

fn main() {
    // env_logger::builder()
    //     .filter_level(log::LevelFilter::Debug)
    //     .init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    simple_logging::log_to_file("analysis.log", log::LevelFilter::Trace).unwrap();

    use sysinfo::{System, SystemExt};
    let mut sys = System::new_all();

    // Define stationarity time, Coherence time
    const STATIONARITY_TIME: Stationarity = Stationarity::Yearly;

    // Initialize Theory
    let theory = DarkPhoton::new();

    // Load in projections and auxiliary values
    let (projections_complete, auxiliary_complete) =
        read_dark_photon_projections_auxiliary().unwrap();
    println!(
        "proj has {} elements, the first has length {}",
        projections_complete.len(),
        projections_complete.iter().next().unwrap().value().len()
    );
    println!("auxiliary has {} elements", auxiliary_complete.h[0].len());
    sys.refresh_all();
    println!("total memory: {} bytes", sys.total_memory());
    println!("used memory : {} bytes", sys.used_memory());

    // NaN checker
    projections_complete
        .projections_complete
        .par_iter()
        .map(|element| {
            assert!(
                element.value().par_iter().all(|e| !e.is_nan()),
                "proj has nan"
            )
        })
        .collect::<Vec<_>>();

    // Start Balancer
    let mut balancer = Balancer::new(12, 2);

    // Compute weights for this coherence time
    Analysis::analysis(
        STATIONARITY_TIME,
        projections_complete,
        auxiliary_complete,
        &theory,
        &mut balancer.manager,
    )
    .expect("failed to run analysis");
}
