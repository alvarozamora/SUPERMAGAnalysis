pub mod weights;
pub mod utils;
pub mod constants;
pub mod theory;

use weights::Weights;
use utils::balancer::Balancer;


fn main() {

    // Initialize mpi
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let balancer = Balancer::new_from_world(world);

    // Define dataset

    // Define theory

    // Run analysis
    // analyze_theory(theory: impl Theory, )
    if balancer.rank == 0 {
        let weights = Weights::new_from_year(2015, &balancer);
        println!("calculated weights on rank {}", balancer.rank);
        println!("len is {}", weights.we.len());
    } else {
        println!("rank {} is chillin out", balancer.rank);
    }

}