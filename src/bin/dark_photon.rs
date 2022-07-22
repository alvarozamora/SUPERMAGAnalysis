use supermag_analysis::weights::{Analysis, Stationarity, Coherence};
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::theory::dark_photon::DarkPhoton;
use std::sync::Arc;
use std::io::Write;

fn main() {

    env_logger::init();
    
    // Define stationarity time, Coherence time 
    const STATIONARITY_TIME: Stationarity = Stationarity::Daily(1);

    // Start Balancer
    let mut balancer = Balancer::<()>::new(32, 10);

    // Initialize Theory
    let theory = DarkPhoton::initialize(1.0);

    // Compute weights for this coherence time
    let analysis_fut = Analysis::new(STATIONARITY_TIME, theory, &mut balancer.manager);
    let analysis = balancer.runtime.block_on(analysis_fut);

    println!("calculated weights on rank {}", balancer.manager.rank);
    println!("weights len is {}", analysis.weights.we.len());
    println!("data_vector len is {}", analysis.data_vector.len());


}