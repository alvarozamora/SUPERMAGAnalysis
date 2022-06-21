use supermag_analysis::weights::{Analysis, Coherence};
use supermag_analysis::utils::async_balancer::Balancer;
use supermag_analysis::theory::dark_photon::DarkPhoton;
use std::sync::Arc;

fn main() {

    // Coherence time in days
    static COHERENCE_TIME: Coherence = Coherence::Daily(2);

    // Start Balancer
    let mut balancer = Balancer::<()>::new(16);

    // Initialize Theory
    let theory = DarkPhoton::initialize(1.0);

    // Compute weights for this coherence time
    let analysis_fut = Analysis::new(COHERENCE_TIME, theory, &mut balancer.manager);
    let analysis = balancer.runtime.block_on(analysis_fut);

    println!("calculated weights on rank {}", balancer.manager.rank);
    println!("weights len is {}", analysis.weights.we.len());
    println!("data_vector len is {}", analysis.data_vector.len());
}