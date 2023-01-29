use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use supermag_analysis::theory::dark_photon::DarkPhoton;
use supermag_analysis::theory::Theory;
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::weights::{Analysis, ProjectionsComplete, Stationarity};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(24)
    //     .build_global()
    //     .unwrap();

    use sysinfo::{NetworkExt, NetworksExt, ProcessExt, System, SystemExt};
    let mut sys = System::new_all();

    // Define stationarity time, Coherence time
    const STATIONARITY_TIME: Stationarity = Stationarity::Yearly;

    // Initialize Theory
    let theory = DarkPhoton::initialize(1.0);

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

    // Start Balancer
    let mut balancer = Balancer::new(32, 2);

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
        serde_cbor::from_slice(&projection_buffer)
            .expect("failed to deserialize projections_complete"),
    );
    let auxiliary_complete: Arc<<DarkPhoton as Theory>::AuxiliaryValue> = Arc::new(
        serde_cbor::from_slice(&auxiliary_buffer)
            .expect("failed to deserialize projections_complete"),
    );

    Ok((projections_complete, auxiliary_complete))
}
