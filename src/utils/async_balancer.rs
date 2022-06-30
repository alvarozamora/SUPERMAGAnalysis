use mpi::topology::{SystemCommunicator, Communicator};
use mpi::traits::*;
use tokio::task::spawn;
use tokio::runtime::Runtime;
use futures::prelude::*;
use futures::stream::FuturesOrdered;
use std::pin::Pin;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<T> {
    pub manager: Manager<T>,
    pub runtime: Runtime,
}

pub struct Manager<T> {
    pub world: SystemCommunicator,
    pub workers: usize,
    pub rank: usize,
    pub size: usize,
    tasks: Vec<Box<dyn Future<Output=T>>>,
    done: bool,
    buffer: usize,
}

impl<T> Balancer<T> {

    /// Constructs a new `Balancer` after initializing mpi
    pub fn new(async_tasks: usize, buffer: usize) -> Self {

        // Initialize mpi
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let max_available_threads = std::thread::available_parallelism().unwrap().get();
        let workers: usize = if async_tasks > max_available_threads {

            println!("async_tasks provided ({async_tasks}) exceeds max_available_threads");
            println!("defaulting to max_available_threads");

            max_available_threads
        } else {

            async_tasks
        };

        // Initialize tokio runtime
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .unwrap();



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
            manager: Manager {
                world,
                workers,
                rank,
                size,
                tasks: vec![],
                done: false,
                buffer,
            },
            runtime
        }
}
}

impl<T> Manager<T> { 

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
    pub fn task(&mut self, fut: Box<dyn Future<Output=T>>)
    {
        self.done = false;
        self.tasks.push(fut);
    }

    pub async fn buffer_await(&mut self) -> Vec<T>
    {
        // Mark as not done
        self.done = false;

        // Pin then execute futures
        let result: Vec<T> = futures::stream::iter(
            self
            .tasks
            .drain(..)
            .map(|fut| Pin::from(fut))
        )
            .buffered(self.size * self.buffer)
            .collect::<Vec<_>>()
            .await;

        // Mark as done and return result
        self.done = true;

        result
    }

    /// Waits for all threads to finish (across all ranks! see `barrier` for blocking on one rank).
    pub fn barrier(&self) {
        self.world.barrier();
    }
}

fn div_ceil(a: usize, b: usize) -> usize
{
    // Note to self:
    // If a is zero this will be zero.
    // If b is zero this will panic.
    (a + b - 1) / b
}