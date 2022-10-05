use supermag_analysis::theory::Theory;
use supermag_analysis::weights::{Analysis, Stationarity, ProjectionsComplete};
use supermag_analysis::theory::dark_photon::DarkPhoton;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

fn main() {

    env_logger::init();
    
    // Define stationarity time, Coherence time 
    const STATIONARITY_TIME: Stationarity = Stationarity::Daily(1);

    // Initialize Theory
    let theory = DarkPhoton::initialize(1.0);

    // Load in projections and auxiliary values
    let mut projections_file = File::open("projections_complete")
        .expect("failed to open file");
    let mut auxiliary_file = File::open("auxiliary_complete")
        .expect("failed to open file");
    let mut projection_buffer = Vec::new();
    let mut auxiliary_buffer = Vec::new();
    projections_file.read_to_end(&mut projection_buffer).expect("failed to read projections");
    auxiliary_file.read_to_end(&mut auxiliary_buffer).expect("failed to read auxiliary");
    let projections_complete: Arc<ProjectionsComplete> = Arc::new(serde_cbor::from_slice(&projection_buffer)
        .expect(" failed to deserialize projections_complete"));
    let auxiliary_complete: Arc<<DarkPhoton as Theory>::AuxiliaryValue> = Arc::new(serde_cbor::from_slice(&auxiliary_buffer)
        .expect(" failed to deserialize projections_complete"));
    println!("proj has {} elements, the first has length {}", projections_complete.len(), projections_complete.iter().next().unwrap().value().len());
    println!("auxiliary has {} elements", auxiliary_complete.h[0].len());

    // Compute weights for this coherence time
    if true {
        Analysis::analysis(
            STATIONARITY_TIME,
            projections_complete,
            auxiliary_complete,
            &theory,
        ).expect("failed to run analysis");
    }
}