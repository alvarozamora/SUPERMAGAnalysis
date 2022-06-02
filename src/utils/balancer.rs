use mpi::topology::{Communicator, SystemCommunicator};
use std::thread::JoinHandle;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer {
    pub world: SystemCommunicator,
    pub workers: usize,
    pub rank: usize,
    pub size: usize,
}

impl Balancer {

    /// Constructs a new `Balancer` from an `mpi::SystemCommunicator` a.k.a. `world`.
    pub fn new_from_world(world: SystemCommunicator) -> Self {
        
        // This is the number of cores available on this node
        let workers = std::thread::available_parallelism().unwrap().get();

        // This is the node id and total number of nodes
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        Balancer {
            world,
            workers,
            rank,
            size,
        }
    }

    /// Calculates local set of items on which to work on.
    fn local_set<'a, T>(&self, items: &'a [T]) -> &'a [T] {

        // Gather and return local set of items
        &items
            .chunks(self.size)
            .nth(self.rank)
            .unwrap()
    }
}

pub struct Handles<T>(Vec<JoinHandle<T>>);

impl<T> Handles<T> {

    pub fn new() -> Self {
        Handles(
            vec![]
        )
    }
    /// Adds a handle
    pub fn add(&mut self, handle: JoinHandle<T>) {
        self.0.push(handle);
    }

    /// Waits for all threads to finish.
    pub fn wait(self) {
        for handle in self.0 {
            handle.join().unwrap();
        }
    }
}