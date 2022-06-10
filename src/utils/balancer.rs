use mpi::topology::{SystemCommunicator, Communicator};
use mpi::traits::*;
use std::thread::{JoinHandle, spawn};
use std::ops::{Add, Sub, Div};

type Handles<T> = Vec<JoinHandle<T>>;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<T> {
    pub world: SystemCommunicator,
    pub workers: usize,
    pub rank: usize,
    pub size: usize,
    handles: Handles<T>,
}

impl<T> Balancer<T> {

    /// Constructs a new `Balancer` from an `mpi::SystemCommunicator` a.k.a. `world`.
    pub fn new(reduce: usize) -> Self {

        // Initialize mpi
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let workers: usize = std::thread::available_parallelism().unwrap().get() - reduce;

        // This is the node id and total number of nodes
        let rank: usize = world.rank() as usize;
        let size: usize = world.size() as usize;

        if rank == 0 {
            println!("--------- Balancer Activated ---------");
            println!("            Nodes : {size}");
            println!(" Workers (rank 0) : {workers} ");
            println!("--------------------------------------");
        } 
        Balancer {
            world,
            workers,
            rank,
            size,
            handles: vec![],
        }
    }

    /// Constructs a new `Balancer` from an `mpi::SystemCommunicator` a.k.a. `world`.
    pub fn new_from_world(world: SystemCommunicator, reduce: usize) -> Self {
        
        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let workers: usize = std::thread::available_parallelism().unwrap().get() - reduce;

        // This is the node id and total number of nodes
        let rank: usize = world.rank() as usize;
        let size: usize = world.size() as usize;

        if rank == 0 {
            println!("--------- Balancer Activated ---------");
            println!("            Nodes : {size}");
            println!(" Workers (rank 0) : {workers} ");
            println!("--------------------------------------");
        } 
        Balancer {
            world,
            workers,
            rank,
            size,
            handles: vec![],
        }
    }


    /// Calculates local set of items on which to work on.
    pub fn local_set<I: Copy + Clone>(&self, items: &Vec<I>) -> Vec<I> {

        // Gather and return local set of items
        items
            .chunks(div_ceil(items.len(), self.size))
            .nth(self.rank)
            .unwrap()
            .to_vec()
    }

    /// Adds a handle
    pub fn add(&mut self, handle: JoinHandle<T>) {
        self.wait_limit();
        self.handles.push(handle);
    }

    /// Adds a handle
    pub fn spawn<F>(&mut self, f: F)
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        self.wait_limit();
        self.handles.push(spawn(f));
    }

    /// Waits for all threads to finish (only on this rank! see `barrier` for blocking across all ranks).
    pub fn wait(&mut self) {
        while self.handles.len() > 0  {
            semi_spinlock();
            self.handles
                .retain(|task| !task.is_finished());
        }
    }

    /// Waits for all threads to finish (across all ranks! see `barrier` for blocking on one rank).
    pub fn barrier(&mut self) {
        self.wait();
        self.world.barrier();
    }

    /// Wait until there is a free worker on this rank
    fn wait_limit(&mut self) {
       while self.handles.len() >= self.workers  {
            semi_spinlock();
            self.handles
                .retain(|task| !task.is_finished());
        }
    }
}

const SEMI_SPINLOCK: u64 = 10;
fn semi_spinlock() { std::thread::sleep(std::time::Duration::from_millis(SEMI_SPINLOCK)) }


fn div_ceil(a: usize, b: usize) -> usize
{
    // Note to self:
    // If a is zero this will be zero.
    // If b is zero this will panic.
    (a + b - 1) / b
}