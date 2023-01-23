use futures::prelude::*;
use futures::stream::FuturesOrdered;
use mpi::environment::Universe;
use mpi::topology::{Communicator, SystemCommunicator};
use mpi::traits::*;
use std::pin::Pin;
use tokio::runtime::Runtime;
use tokio::task::spawn;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<T = ()> {
    pub manager: Manager<T>,
    pub runtime: Runtime,
}

pub struct Manager<T> {
    pub universe: Universe,
    pub world: SystemCommunicator,
    pub workers: usize,
    pub rank: usize,
    pub size: usize,
    tasks: Vec<Box<dyn Future<Output = T>>>,
    done: bool,
    /// Size of Futures Buffer (per worker)
    buffer: usize,
}

impl<T> Balancer<T> {
    /// Constructs a new `Balancer` after initializing mpi.
    pub fn new(parallel_async_tasks: usize, buffer: usize) -> Self {
        // Initialize mpi
        let universe = mpi::initialize().expect("Failed to initialize mpi");
        let world = universe.world();

        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let max_available_threads = std::thread::available_parallelism()
            .expect("failed to retrieve number of threads on this system")
            .get();
        let workers: usize = if parallel_async_tasks > max_available_threads {
            println!("parallel_async_tasks provided ({parallel_async_tasks}) exceeds max_available_threads {max_available_threads}");
            println!("defaulting to max_available_threads");

            max_available_threads
        } else {
            parallel_async_tasks
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
                universe,
                world,
                workers,
                rank,
                size,
                tasks: vec![],
                done: false,
                buffer,
            },
            runtime,
        }
    }
}

impl<T> Manager<T> {
    /// Calculates local set of items on which to work on.
    pub fn local_set<I: Clone>(&mut self, items: &Vec<I>) -> Option<Vec<I>> {
        // Gather and return local set of items
        let mut local_sets = items.chunks(div_ceil(items.len(), self.size));

        // Return nth local set
        Some(local_sets.nth(self.rank)?.to_vec())
    }

    /// Adds a handle
    pub fn task(&mut self, fut: Box<dyn Future<Output = T>>) {
        self.done = false;
        self.tasks.push(fut);
    }

    /// Adds a set of handles
    pub fn tasks(&mut self, mut fut: Vec<Box<dyn Future<Output = T>>>) {
        self.done = false;
        self.tasks.append(&mut fut);
    }

    /// Buffered awaits all futures on current node without waiting for other nodes. (Use in conjunction with [`barrier`] for blocking across all ranks).
    ///
    /// [`barrier`]: #method.barrier
    pub async fn buffer_await(&mut self) -> Vec<T> {
        // Mark as not done
        self.done = false;

        // Pin then execute futures
        let result: Vec<T> = futures::stream::iter(self.tasks.drain(..).map(|fut| Pin::from(fut)))
            .buffered(self.size * self.buffer)
            .collect::<Vec<_>>()
            .await;

        // Mark as done and return result
        self.done = true;

        result
    }

    /// Waits for all ranks to reach this point. Pairs well with [`buffer_await`].
    ///
    /// Basic usage:
    ///
    /// ```skip
    /// // Wait for all futures to be awaited on this rank.
    /// manager.buffer_await();
    /// // Wait for all ranks to finish awaiting all futures.
    /// manager.barrier();
    /// ```
    /// [`buffer_await`]: #method.buffer_await
    pub fn barrier(&self) {
        self.world.barrier();
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    // Note to self:
    // If a is zero this will be zero.
    // If b is zero this will panic.
    (a + b - 1) / b
}
