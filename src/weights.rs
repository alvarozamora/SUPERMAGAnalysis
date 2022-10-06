use crate::{
    constants::*,
    utils::*,
    utils::loader::*,
    utils::async_balancer::*,
    theory::*,
};
use itertools::Itertools;
use mpi::{topology::Communicator, traits::Root};
use ndarray::{s, Array1, arr1, ArrayViewMut, Array2};
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use tokio::{fs::File, io::AsyncWriteExt};
use std::{sync::Arc, ops::{RangeInclusive, Add}, error::Error, marker::PhantomData};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::ops::AddAssign;
use dashmap::try_result::TryResult;
use std::ops::Range;
use rayon::iter::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use mpi::point_to_point::{Source, Destination};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

use log::*;

macro_rules! debug {
    ($($e:expr),+) => {
        {
            #[cfg(debug_assertions)]
            {
                dbg!($($e),+)
            }
            #[cfg(not(debug_assertions))]
            {
                ($($e),+)
            }
        }
    };
}

macro_rules! debug_print {
    ($($e:expr),+) => {
        {
            #[cfg(debug_assertions)]
            {
                println!($($e),+)
            }
            #[cfg(not(debug_assertions))]
            {
                ($($e),+)
            }
        }
    };
}


type Index = usize;
type Weight = f32;
type StationName = String;
type TimeSeries = Array1<f32>;
type ComplexSeries = Array1<ndrustfft::Complex<f32>>;

#[derive(Clone, PartialEq, Debug)]
pub struct FrequencyBin {
    pub lower: f64,
    pub multiples: RangeInclusive<usize>,
}

/// This holds the weights (inverse white noise). These weights are a measurement 
/// intrinsic to the dataset and do not depend on the theory.
pub struct Weights {
    pub n: DashMap<Index, DashMap<StationName, Weight>>,
    pub e: DashMap<Index, DashMap<StationName, Weight>>,
    pub wn: DashMap<Index, TimeSeries>,
    pub we: DashMap<Index, TimeSeries>,
    pub coherence: Coherence,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectionsComplete {

    /// The complete time series of the projections for every nonzero element
    pub projections_complete: DashMap<NonzeroElement, TimeSeries>,

    /// The second at which this series begins, where t=0 is the first second available in the dataset.
    pub start_second: usize,

    /// Days used
    pub days: Range<usize>,
}
use dashmap::iter::Iter;

impl ProjectionsComplete {
    pub fn iter(&self) -> Iter<NonzeroElement, TimeSeries> {
        self.projections_complete.iter()
    }
    /// Note this returns number of elements, not length of time series
    pub fn len(&self) -> usize {
        self.projections_complete.len()
    }
    /// Returns the time domain of the first series, which should be the same for all series.
    pub fn secs(&self) -> Range<usize> {
        self.start_second..self.start_second.add(self.projections_complete.iter().next().unwrap().len())
    }
}


// pub struct StitchedWeights {
//     pub n: DashMap<StationName, Weight>,
//     pub e: DashMap<StationName, Weight>,
//     pub wn: TimeSeries,
//     pub we: TimeSeries,
//     pub stationarity: Stationarity,
// }

// impl Weights {

//     pub fn stitch(self) -> StitchedWeights {
//         let stitched_n = DashMap::<StationName, Weight>::new();
//         self.n
//             .into_iter()
//             .sorted_by_key(|a| a.0)
//             .for_each(|(chunk_idx, map)| {
//                 map
//                     .into_par_iter()
//                     .for_each(|(name, series)| {
//                         stitched_n
//                             .entry(name)
//                             .and_modify(|s| s.)
//                     })

//             })

//         StitchedWeights {
//             n: DashMap::new(),
//             e: DashMap::new(),
//             wn: TimeSeries::zeros(5),
//             we: TimeSeries::zeros(5),
//             stationarity: self.stationarity
//         }
//     }
// }


pub struct Analysis<T>(PhantomData<T>);
// <T: Theory + Send>;
// {
//     pub weights: Arc<Weights>,
//     pub theory: T,
//     pub projections: Arc<DashMap<Index, DashMap<NonzeroElement, TimeSeries>>>,
//     pub data_vector: DashMap<Index, DashMap<NonzeroElement, ComplexSeries>>,
// }

impl<T: Theory + Send + Sync + 'static> Analysis<T> {

    /// This calculates the inverse white noise (i.e. weights) for a given yearly dataset,
    /// as per eq 13 and 14 in the paper. 
    // pub fn new_from_year(year: usize, balancer: &Balancer) -> Arc<Self> {

    //     // Grab dataset loader for the year
    //     let mut loader: YearlyDatasetLoader = YearlyDatasetLoader::new_from_year(year);

    //     // Initialize empty `Weights`
    //     let weights: Arc<Weights> = Arc::new(Weights {
    //             n: DashMap::new(),
    //             e: DashMap::new(),
    //             wn: DashMap::new(),
    //             we: DashMap::new(),
    //     });

    //     while loader.station_files.len() > 0 {

    //         // Grab station file
    //         let station_file: String = loader.station_files.pop().unwrap();

    //         // Get Arc to dashmap/hashmap
    //         let local_weights = weights.clone();
    //         let coordinate_map = loader.coordinate_map.clone();

    //         // Spawn a new thread for each `station_file`
    //         balancer.
    //             spawn(move || {
                
    //                 // Load dataset
    //                 let dataset: Dataset = YearlyDatasetLoader::load(year, station_file, coordinate_map);

    //                 // Valid samples
    //                 let num_samples: usize = dataset.field_1.fold(0, |acc, &x| if x != SUPERMAG_NAN { acc + 1 } else { acc } );
    //                 println!(
    //                     "Station {} has {} null entries ({:.1}%) for year {}",
    //                     &dataset.station_name,
    //                     dataset.field_1.len() - num_samples,
    //                     (dataset.field_1.len() - num_samples) as f64 / dataset.field_1.len() as f64,
    //                     year,
    //                 );

    //                 // Clean the fields
    //                 let clean_field_1 = dataset.field_1.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });
    //                 let clean_field_2 = dataset.field_2.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });

    //                 if cfg!(debug_assertions) {
    //                     println!(
    //                         "Station {} has {:?} min/max entries for year {}",
    //                         &dataset.station_name,
    //                         clean_field_1.fold((0.0, 0.0), |mut acc, &x| {
    //                             if acc.0 > x {
    //                                 acc.0 = x; 
    //                             }
    //                             if acc.1 < x {
    //                                 acc.1 = x;
    //                             }
    //                             acc
    //                         }),
    //                         year,
    //                     );
    //                 }

    //                 // Calculate weights
    //                 let n_weight: f32 = (clean_field_1.dot(&clean_field_1) / num_samples as f32).recip();
    //                 let e_weight: f32 = (clean_field_2.dot(&clean_field_2) / num_samples as f32).recip();
    //                 let wn_weight: TimeSeries = clean_field_1.map(|&x| if x != 0.0 { n_weight } else { 0.0 });
    //                 let we_weight: TimeSeries = clean_field_2.map(|&x| if x != 0.0 { e_weight } else { 0.0 });

    //                 // Add to dashmap
    //                 local_weights.n.insert(dataset.station_name.clone(), n_weight);
    //                 local_weights.e.insert(dataset.station_name.clone(), e_weight);
    //                 if local_weights.wn.contains_key(&dataset.station_name) {
    //                     let mut array = local_weights.wn.get_mut(&dataset.station_name).unwrap();
    //                     array.add_assign(&wn_weight);
    //                 } else {
    //                     local_weights.wn.insert(dataset.station_name.clone(), wn_weight);
    //                 }
    //                 if local_weights.we.contains_key(&dataset.station_name) {
    //                     let mut array = local_weights.we.get_mut(&dataset.station_name).unwrap();
    //                     array.add_assign(&we_weight);
    //                 } else {
    //                     local_weights.we.insert(dataset.station_name.clone(), we_weight);
    //                 }
                    
    //         });
           
    //     }

    //     // Wait for threads on all nodes to finish
    //     balancer.barrier();

    //     weights
    // }

    /// This runs an analysis for a given theory. It chunks the data into the specified intervals,
    /// calculates the inverse white noise (i.e. weights) (similar to eq 13 and 14 in the paper)
    ///  and calculates the data vector on said chunks of data.
    /// 
    /// NOTES
    /// [ ] Change station criteria so that series with at least one valid point are included
    /// Chuck all data prior to 2003
    /// Do coordinate transformation: linearly interpolate between the values in `IGRF_declinations_for_1sec.txt`. The values given are for the start of the 180th day of each year.
    pub async fn calculate_projections_and_auxiliary(
        // stationarity: Stationarity,
        coherence: Coherence,
        theory: T,
        days: Option<Range<usize>>,
        balancer: &mut Manager<()>
    ) -> Result<(), Box<dyn Error>> {

        println!("Running Analysis");

        // // Number of days per chunk
        // let stationarity: usize = match stationarity {
        //     Stationarity::Yearly => todo!(),
        //     Stationarity::Daily(days) => days,
        // };

        let days_per_chunk = match coherence {
            Coherence::Days(days) => days,
            Coherence::Yearly => todo!(),
        };

        // Grab dataset loader for the chunks
        let loader = Arc::new(DatasetLoader::new(days.clone(), coherence));

        // Initialize empty `Weights`
        let weights: Arc<Weights> = Arc::new(Weights {
                n: DashMap::new(),
                e: DashMap::new(),
                wn: DashMap::new(),
                we: DashMap::new(),
                coherence,
        });

        // Initialize empty data vector, projections
        let projections = Arc::new(DashMap::new());
        let auxiliary_values = Arc::new(DashMap::new());

        // Wrap theory in an Arc
        let theory = Arc::new(theory);

        // Calculate local set of tasks
        let set: Range<usize> = 0..loader.semivalid_chunks.len();
        let local_set: Vec<usize> = balancer.local_set(&set.collect());

        // This loop calculates weights
        for entry in local_set {

            // Clone all relevant Arcs
            let local_weights: Arc<_> = weights.clone();
            let local_loader: Arc<_> = loader.clone();
            let local_theory: Arc<_> = theory.clone();
            let local_projections: Arc<_> = projections.clone();
            let local_auxiliary: Arc<_> = auxiliary_values.clone();

            // Create a new task for each `chunk`
            balancer.
                task(Box::new( async move { tokio::task::spawn( async move {

                    // Get chunk index
                    let index: Index = local_loader.semivalid_chunks.entry(entry).into_key();
                    debug_print!("{:?} working on index {index}", std::thread::current().id());

                    // Load datasets for this chunk
                    let datasets: DashMap<StationName, Dataset> = local_loader.load_chunk(index).await.unwrap();

                    // e.g. on a year boundary where all stations change
                    if datasets.len() == 0 {
                        debug_print!("Empty chunk. Proceeding to next chunk");
                        return ()
                    }

                    // Local hashmaps and time series
                    let local_hashmap_n: DashMap<StationName, Weight> = DashMap::with_capacity(datasets.len());
                    let local_hashmap_e: DashMap<StationName, Weight> = DashMap::with_capacity(datasets.len());
                    let mut local_wn: TimeSeries = Array1::from_vec(vec![0.0_f32; days_per_chunk * SECONDS_PER_DAY]);
                    let mut local_we: TimeSeries = Array1::from_vec(vec![0.0_f32; days_per_chunk * SECONDS_PER_DAY]);
                    
                    // Calculate weights based on datasets for this chunk (stationarity period)
                    calculate_weights_for_chunk(
                        index,
                        &local_hashmap_n,
                        &local_hashmap_e,
                        &mut local_wn,
                        &mut local_we,
                        &datasets,
                    ).await;
                    debug_print!("Finished weights for index {index}");
                    debug_print!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    debug_print!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    // e.g. all stations have nans for all values for this chunk
                    if local_hashmap_n.len() < 0*T::MIN_STATIONS { 
                        println!("Invalid chunk, as there are less than {} stations with data in this chunk. Proceeding to next chunk", T::MIN_STATIONS);
                        return ()
                    } else if local_wn.iter().any(|&x| x == 0.0_f32) {
                        println!("Invalid chunk, as there is at least one time slot with a normalization weight of 0.0");
                        return ()
                    }

                    // Calculate projections for this chunk. 
                    // This is done here despite potentially discarding result later if chunk is not in largest contiguous subset because
                    // we do not want to repeat I/O with hundreds of gigabytes of data. I.e. we already have the data in memory here.
                    debug_print!("calculating projections");
                    let chunk_projections: DashMap<NonzeroElement, TimeSeries> = local_theory
                        .calculate_projections(
                            &local_hashmap_n,
                            &local_hashmap_n,
                            &local_wn,
                            &local_we,
                            &datasets
                        );
                    let chunk_auxiliary = local_theory
                        .calculate_auxiliary_values(
                            &local_hashmap_n,
                            &local_hashmap_n,
                            &local_wn,
                            &local_we,
                            &datasets
                        );
                    assert!(local_projections.insert(index, chunk_projections).is_none(), "A duplicate projection entry was made");
                    assert!(local_auxiliary.insert(index, chunk_auxiliary).is_none(), "A duplicate auxiliary entry was made");
                    
                    // Add weights to dashmap
                    local_weights.n.insert(index,local_hashmap_n);
                    local_weights.e.insert(index,local_hashmap_e);
                    // This is completely unnecessary but was a cool piece of code that I wrote to async-ify dashmap access.
                    // Keeping it for future reference. Should not do much for our particular application, and may even positively
                    // affect performance during simultaneous attempts to access the same shard.
                    loop {
                        match local_weights.wn.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },
                            TryResult::Absent => { local_weights.wn.insert(index, local_wn);  break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }
                    loop {
                        match local_weights.we.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },
                            TryResult::Absent => { local_weights.we.insert(index, local_we);  break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }
                }).await.unwrap()
            }));
        }

        // Wait for all tasks on this node to finish, and check that they all succeeded
        println!("About to buffer_await");
        let result: Vec<()> = balancer.buffer_await().await;
        assert!(result.iter().all(|&x| x == ()));
        balancer.barrier();

        // Flatten all chunks over which this largest contiguous subset spans
        // to get T::NONZERO_ELEMENTS number of large series. This essentially flattens
        // `projections` from DashMap<Index, DashMap<NonzeroElement, TimeSeries>> to
        // DashMap<NonzeroElement, TimeSeries> (i.e. the complete, stiched-together time series)
        let projections_complete: DashMap<NonzeroElement, TimeSeries> = {

            // First communicate about which nonzero chunks were obtained
            let my_nonzero_chunks: Vec<Index> = projections
                .iter()
                .fold(vec![], |mut v, pair| { v.push(*pair.key()); v});
            let mut all_nonzero_chunks = my_nonzero_chunks.clone();
            
            let empty: Vec<usize> = vec![std::usize::MAX, std::usize::MAX, std::usize::MAX];

            // Synchronize nonzero chunks for every rank
            for rank in 0..balancer.size {

                if rank == balancer.rank {

                     // If it is your turn, broadcast to all ranks `send_to_rank` that are not yourself
                    for send_to_rank in 0..balancer.size {

                        if send_to_rank != rank {
                            if my_nonzero_chunks.len() > 0 {
                                debug_print!("rank {rank} about to send {} nonzero chunks {:?} to {send_to_rank}", my_nonzero_chunks.len(), &my_nonzero_chunks);
                                balancer.world.process_at_rank(send_to_rank as i32).send(&my_nonzero_chunks);
                            } else {
                                // send empty
                                debug_print!("rank {rank} about to send empty to {send_to_rank}");
                                balancer.world.process_at_rank(send_to_rank as i32).send(&empty);
                            }

                        } 
                    }
        
                } else {
                    
                    // If it is not your turn to broadcast, receive other_nonzero_chunks and append
                    let (mut other_nonzero_chunks, status) = balancer.world
                        .any_process()
                        .receive_vec::<Index>();

                    debug_print!("rank {} received from rank {}: {:?}", balancer.rank, status.source_rank(), &other_nonzero_chunks);

                    if empty != other_nonzero_chunks {
                        all_nonzero_chunks.append(&mut other_nonzero_chunks);
                    }
                }

                // Barrier, for safety. Don't think we necessarily need it here and not after this scope
                balancer.barrier();
            }

            // After receiving all nonzero chunks, sort. Now, all ranks should have this same vector.
            all_nonzero_chunks.sort();

            // We first initialize a dashmap with T::NONZERO_ELEMENTS number of zeroed series
            // to which we will add nonzero chunks
            let nonzero_elements: HashSet<NonzeroElement> = T::get_nonzero_elements();

            // Iterate through sorted chunk index array and send/receive chunks
            for nonzero_chunk in all_nonzero_chunks {

                if my_nonzero_chunks.contains(&nonzero_chunk) {

                    // If you hold the data to this chunk, send it to other ranks
                    
                    // Get all nonzero elements for this theory for this chunk
                    let chunk_data_map = projections.get(&nonzero_chunk).unwrap();

                    // Iterate through and send all nonzero elements
                    for element in &nonzero_elements {

                        // Get data for this element from the chunk data map
                        let element_data = chunk_data_map.get(element).unwrap();
                        // This clone is theoretically unnecessary. It's done here to use a type (vec) that impls necessary traits to send
                        let element_data: Vec<f32> = element_data.clone().into_raw_vec();
                        for send_to_rank in 0..balancer.size {

                            if send_to_rank != balancer.rank {
                                debug_print!("Rank {} about to send {} f32s [{:?} .. ] for index {} to rank {}", balancer.rank, element_data.len(), &element_data[..3], nonzero_chunk, send_to_rank);

                                balancer.world
                                    .process_at_rank(send_to_rank as i32)
                                    .send(&element_data); 
                            }
                        }

                        // Wait for all other ranks to receive this nonzero element
                        balancer.barrier();
                    }
                } else {
         
                    // Otherwise, receive nonzero chunk
                    let chunk_map: DashMap<NonzeroElement, TimeSeries> = DashMap::new();

                    // Iterate through and receive all nonzero elements
                    for element in &nonzero_elements {

                        // let mut buffer_for_recieving: Vec<f32> = Vec::with_capacity(size); // This heap allocation is necessary, and ownership will be passed onto the dashmap

                        debug_print!("rank {} about to receive f32s", balancer.rank);
                        let (buffer_for_recieving, _status) = balancer.world
                            .any_process()
                            .receive_vec::<f32>();

                        // Convert buffer into array
                        let chunk_to_insert = TimeSeries::from_vec(buffer_for_recieving);

                        // Insert chunk into chunk_map
                        chunk_map.insert(element.clone(), chunk_to_insert);
                        
                        // Wait for all other ranks to receive this nonzero element
                        balancer.barrier();
                    }

                    // Insert chunk map into projections
                    projections
                        .insert(nonzero_chunk, chunk_map);
                }
            }

            // Now that all ranks have all of the data, need to find longest contiguous subset of chunks
            let (size, starting_value): (usize, usize) = {
            
                // To do this we first gather all chunks
                let set: Vec<usize> = projections
                    .par_iter_mut()
                    .map(|pair| *pair.key())
                    .collect();

                // Then, return largest contiguous subset
                get_largest_contiguous_subset(&set)
            };
            println!("rank {}: longest contiguous subset of chunks begins at {starting_value} and has length {size} chunks", balancer.rank);

            // TODO: generalize to yearly
            let secs_per_chunk = SECONDS_PER_DAY * days_per_chunk;

            let complete_series = Arc::new(DashMap::new());
            for element in &nonzero_elements {
                let empty_array = TimeSeries::zeros(size * secs_per_chunk);
                debug_print!("initializing map element {:?} with zeros array of len {}={}", element, size * secs_per_chunk, empty_array.len());
                complete_series.insert(element.clone(), empty_array);
            }

            projections
                .iter()
                .for_each(|chunk| {

                    // Get chunk and it's chunk_map
                    let (&current_chunk, chunk_map) = chunk.pair();

                    if !in_longest_subset(current_chunk, size, starting_value) {
                        return ()
                    }

                    chunk_map
                        .iter()
                        .for_each(|element| {
                        
                            // get element and its correpsonding series
                            let (element, series) = element.pair();

                            // Insert array into complete series
                            // TODO: THIS ASSUMES ALL CHUNKS ARE THE SAME LENGTH.
                            // NEED TO CHANGE FOR YEARLY STATIONARITY AND PERHAPS THE EDGE CHUNKS.
                            let start_index = (current_chunk-starting_value)*series.len();
                            let end_index = start_index + series.len();

                            complete_series
                                .get_mut(element)
                                .unwrap()
                                .slice_mut(s![start_index..end_index])
                                .assign(&series);
                        })
                });
            
            Arc::try_unwrap(complete_series).expect("An arc somehow survived")
        };

        // Flatten all chunks over which this largest contiguous subset spans
        // to get T::NONZERO_ELEMENTS number of large series. This essentially flattens
        // `auxiliary` from DashMap<_, AuxiliaryValue> to Auxiliary value (i.e. the complete,
        // stiched-together time series).
        let start_second;
        let auxiliary_complete: <T as Theory>::AuxiliaryValue = {

            // First communicate about which nonzero chunks were obtained
            let my_nonzero_chunks: Vec<Index> = auxiliary_values
                .iter()
                .fold(vec![], |mut v, pair| { v.push(*pair.key()); v});
            let mut all_nonzero_chunks = my_nonzero_chunks.clone();
            
            let empty: Vec<usize> = vec![std::usize::MAX, std::usize::MAX, std::usize::MAX];

            // Synchronize nonzero chunks for every rank
            for rank in 0..balancer.size {

                if rank == balancer.rank {

                     // If it is your turn, broadcast to all ranks `send_to_rank` that are not yourself
                    for send_to_rank in 0..balancer.size {

                        if send_to_rank != rank {
                            if my_nonzero_chunks.len() > 0 {
                                debug_print!("rank {rank} about to send {} nonzero chunks {:?} to {send_to_rank}", my_nonzero_chunks.len(), &my_nonzero_chunks);
                                balancer.world.process_at_rank(send_to_rank as i32).send(&my_nonzero_chunks);
                            } else {
                                // send empty
                                debug_print!("rank {rank} about to send empty to {send_to_rank}");
                                balancer.world.process_at_rank(send_to_rank as i32).send(&empty);
                            }

                        } 
                    }
        
                } else {
                    
                    // If it is not your turn to broadcast, receive other_nonzero_chunks and append
                    let (mut other_nonzero_chunks, status) = balancer.world
                        .any_process()
                        .receive_vec::<Index>();

                    debug_print!("rank {} received from rank {}: {:?}", balancer.rank, status.source_rank(), &other_nonzero_chunks);

                    if empty != other_nonzero_chunks {
                        all_nonzero_chunks.append(&mut other_nonzero_chunks);
                    }
                }

                // Barrier, for safety. Don't think we necessarily need it here and not after this scope
                balancer.barrier();
            }

            // After receiving all nonzero chunks, sort. Now, all ranks should have this same vector.
            all_nonzero_chunks.sort();

            // Iterate through sorted chunk index array and send/receive chunks
            for nonzero_chunk in all_nonzero_chunks {

                if my_nonzero_chunks.contains(&nonzero_chunk) {

                    // If you hold the data to this chunk, send it to other ranks
                    
                    // Get auxiliary values for this chunk and serialize
                    let chunk_auxiliary: Vec<u8> = serde_cbor::to_vec(&*auxiliary_values.get(&nonzero_chunk).unwrap())
                        .expect("failed to serialize serializable type");

                    for send_to_rank in 0..balancer.size {

                        if send_to_rank != balancer.rank {
                            debug_print!("Rank {} about to send {} auxiliary bytes [{:?} .. ] for index {} to rank {}", balancer.rank, chunk_auxiliary.len(), &chunk_auxiliary[..3], nonzero_chunk, send_to_rank);

                            balancer.world
                                .process_at_rank(send_to_rank as i32)
                                .send(&chunk_auxiliary); 
                        }
                    }

                    // Wait for all other ranks to receive this nonzero element
                    balancer.barrier();
                } else {

                    debug_print!("rank {} about to receive auxiliary f32s", balancer.rank);
                    let (buffer_for_recieving, _status) = balancer.world
                        .any_process()
                        .receive_vec::<u8>();

                    // Convert buffer into array
                    let auxiliary_chunk_to_insert: <T as Theory>::AuxiliaryValue = serde_cbor::from_slice(&buffer_for_recieving)
                        .expect("failed to deserialize deserializable auxilary value");
                    
                    // Wait for all other ranks to receive this nonzero element
                    balancer.barrier();

                    auxiliary_values.insert(nonzero_chunk, auxiliary_chunk_to_insert);
                }
            }

            // Now that all ranks have all of the data, need to find longest contiguous subset of chunks
            // This **should** be the same set as projections_complete but we will recompute and verify.
            let (size, starting_value): (usize, usize) = {
            
                // To do this we first gather all chunks
                let set: Vec<usize> = auxiliary_values
                    .par_iter_mut()
                    .map(|pair| *pair.key())
                    .collect();

                // Then, return largest contiguous subset
                get_largest_contiguous_subset(&set)
            };
            println!("rank {}: longest contiguous subset of chunks of auxilary values begins at {starting_value} and has length {size} chunks", balancer.rank);
            start_second = starting_value;

            // TODO: generalize to yearly
            let secs_per_chunk: usize = SECONDS_PER_DAY * days_per_chunk;

            let complete_series = <T as Theory>::AuxiliaryValue::from_chunk_map(
                &auxiliary_values,
                secs_per_chunk,
                starting_value,
                size,
            );
            complete_series
        };

        // Calculating the projections and auxiliary values are the only reason 
        // why we really needed multinode parallelism, since the dataset is almost 1 TB.
        // We can finish the calculation with a single node, so save results to disk.
        //
        // First, open file. Then, save serialized bytes.
        if balancer.rank == 0 {
            let (mut projections_file, mut auxiliary_file) = futures::join!(
                async { File::create("projections_complete").await.unwrap() },
                async { File::create("auxiliary_complete").await.unwrap() }
            );

            // Wrap with some metadata
            let projections_complete = ProjectionsComplete {
                projections_complete,
                start_second: start_second,
                days: days.clone().unwrap_or(DATA_DAYS),
            };

            // Define futures for saving these to disk
            let proj_future = async {
                let proj_serialized: Vec<u8> = serde_cbor::to_vec(&projections_complete).expect("failed to serialize projections_complete");
                projections_file
                    .write_all(&proj_serialized)
                    .await
                    .expect("failed to write projections to disk");
                println!("saved projections to disk");
            };
            let aux_future = async {
                let aux_serialized: Vec<u8> = serde_cbor::to_vec(&auxiliary_complete).expect("failed to serialize projections_complete");
                auxiliary_file
                    .write_all(&aux_serialized)
                    .await
                    .expect("failed to write projections to disk");
                println!("saved auxilary to disk");
            };

            // Save serialized bytes
            futures::join!(proj_future, aux_future);
        }

        Ok(())
    }


    pub fn analysis(
        stationarity: Stationarity,
        projections_complete_struct: Arc<ProjectionsComplete>,
        auxiliary_complete: Arc<<T as Theory>::AuxiliaryValue>,
        theory: &T,
        // balancer: &mut Manager<()>,
    ) -> Result<(), Box<dyn Error>> {

        // Retrieve series and days used from ProjectionsComplete
        let days = projections_complete_struct.days.clone();

        // Calculate coherence_times: a vector containing the integer number of seconds for every
        // coherence time we are considering.
        //
        // projections_complete is the map containing the stitched-together timeseries
        // so the length of one of the time series is the total amount of time we are analyzing.
        let total_secs: usize = projections_complete_struct.projections_complete.iter().next().unwrap().value().len();
        let total_time = total_secs as f64;
        let coherence_times: Vec<usize> = coherence_times(total_time, THRESHOLD);

        // Calculate frequency bins from coherence time
        let frequency_bins: Vec<FrequencyBin> = frequencies_from_coherence_times(&coherence_times);

        // Partition zipped(coherence_times, frequency_bins) over ranks
        // let local_set: Vec<(usize, FrequencyBin)> = balancer
        //     .local_set(
        //         &coherence_times
        //         .iter()
        //         .cloned()
        //         .zip(frequency_bins)
        //         .collect()
        //     );

        let set =  coherence_times
            .iter()
            .cloned()
            .zip(frequency_bins)
            .collect();

        // // Calculate data vector for this local set.
        // let data_vector_dashmap = theory
        //     // .calculate_data_vector(&projections_complete, &local_set);
        //     .calculate_data_vector(&projections_complete, &set);
        // log::debug!("finished data vector");
        
            // Calculate the theory mean
        let theory_mean = theory
            // .calculate_mean_theory(&local_set, total_secs, coherence_times.len(), auxiliary_complete);
            .calculate_mean_theory(&set, total_secs, coherence_times.len(), Arc::clone(&auxiliary_complete));
        log::debug!("finished mean");

        // Calculate the theory var
        let theory_var = theory
            // .calculate_var_theory(&local_set, total_secs, coherence_times.len(), auxiliary_complete);
            .calculate_var_theory(&set, &projections_complete_struct, coherence_times.len(), days, stationarity, auxiliary_complete);
        log::debug!("finished var");
        

        // println!("{} {} {} elements", data_vector_dashmap.len(), theory_mean.len(), theory_var.len());

        Ok(())
    }
}




#[derive(Copy, Clone, Debug)]
pub enum Stationarity {
    Yearly,
    Daily(usize),
}

impl Stationarity {

    /// This retrieves the start and end second index for the given year.
    /// 
    /// Note that these indices are relative to the first day there was data,
    /// not the first day for any subset of data being used for the analysis.
    pub fn get_year_indices(&self, year: usize) -> (usize, usize) {
        
        // First second of the year provided
        let start = day_since_first(0, year) * 24 * 60 * 60;

        // Last second of the year provided, calculated as
        // first second of the next year minus one
        let end = day_since_first(0, year+1) * 24 * 60 * 60 - 1;

        (start, end)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Coherence {
    Yearly,
    Days(usize)
}


/// Calculate the coherence times
fn coherence_times(total_time: f64, threshold: f64) -> Vec<usize> {

    // Initialize return value
    let mut times = vec![];

    // Find max n in eq (??)
    let max_n: usize = (-0.5*(1_000_000.0 / total_time).ln()/(1.0 + threshold).ln()).round() as usize;

    for n in 0..=max_n {

        // Find the raw coherence time
        let raw_coherence_time: f64 = total_time / (1.0 + threshold).powi(2*n as i32);
        let rounded = raw_coherence_time.round() as usize;

        // Find number in [rounded-10, .., rounded+10] with smallest max prime
        let mut number = 0;
        let mut max_prime = std::usize::MAX;
        for candidate in rounded-10..=rounded+10 {

            let x = maxprime(candidate);
            if x < max_prime {
                number = candidate;
                max_prime = x;
            }
        }

        times.push(number)
    }

    // Values should be in descending order.
    // Return values in descending order. 
    times
}

/// Given a complete set of `coherence_times`, sorted in descending order, calculates
/// the corresponding frequency bins
fn frequencies_from_coherence_times(coherence_times: &Vec<usize>) -> Vec<FrequencyBin> {

    assert!(
        coherence_times.iter().max().unwrap() > &I_MIN,
        "In order to have frequency multiple i_min be O(v_dm^-2), need max coherence time to be at least v_dm^-2 seconds"
    );
    
    // Calculate base frequencies, i.e. reciprocal of coherence times
    // Since coherence_times is in descending order, these will be in ascending order
    let base_frequencies: Vec<f64> = coherence_times
        .iter()
        .map(|&x| (x as f64).recip())
        .collect();

    // This constructs all frequency bins except for the highest one
    let mut frequency_bins: Vec<FrequencyBin> = base_frequencies
        .windows(2) // Get all neighboring pairs in `base_frequencies`
        .enumerate()
        .map(|(bin_index, pair)| {
            
            // Unpack pair
            let (lower, higher) = (pair[0], pair[1]);
            assert!(higher > lower, "frequencies are not in correct order");

            // Find start and end multiples of frequency
            let start = if bin_index == 0 { 0 } else { I_MIN };
            let mut end = (higher * I_MIN as f64 / lower) as usize;
            if end as f64 * lower >= higher * I_MIN as f64 { end -= 1; }

            assert!(end as f64 * lower < (higher * I_MIN as f64), "highest frequency in bin is higher than lowest in next bin");

            // Return bin containing all frequencies in [lower, higher), in ascending order
            return FrequencyBin { lower: lower, multiples: start..=end }

        }).collect();

    // Add highest frequency bin
    frequency_bins.push({
        
        let last_coherence_time_in_seconds: usize = *coherence_times.last().unwrap();
        let highest_frequency: usize = last_coherence_time_in_seconds - 1;
        let highest_frequency_bin_start = (last_coherence_time_in_seconds as f64).recip();

        FrequencyBin {
            lower: highest_frequency_bin_start,
            multiples: I_MIN..=highest_frequency,
        }
    });

    // Return bin
    frequency_bins
}

#[test]
fn test_frequencies_from_coherence_times() {

    // The function that generates these returns them in descending order. lets do the same
    let coherence_times = vec![10_000_000, 250_000];

    // Calculate frequency bins
    let frequency_bins = frequencies_from_coherence_times(&coherence_times);

    frequency_bins
        .iter()
        .zip(
        vec![
            FrequencyBin {
                lower: 1.0 / 10_000_000.0,
                multiples: 0..=(I_MIN as f64 * 10_000_000.0 / 250_000.0) as usize - 1 /* -1 because 1e-6 neatly divides 1/250_000 */,
            },
            FrequencyBin {
                lower: 1.0 / 250_000.0,
                multiples: I_MIN..=249_999,
            }
        ].iter()
    ).for_each(|(bin_a, bin_b)| {
        assert_eq!(bin_a, bin_b, "frequency bins are not the same");
    })
        
}

async fn calculate_weights_for_chunk(
    index: Index,
    local_hashmap_n: &DashMap<StationName, Weight>,
    local_hashmap_e: &DashMap<StationName, Weight>,
    local_wn: &mut TimeSeries,
    local_we: &mut TimeSeries,
    datasets: &DashMap<StationName, Dataset>,
    // local_valid_seconds: Arc<DashMap<usize /* index */, usize /* count */>>,
) {
    datasets
        .iter()
        .for_each(|dataset| {
        
            // Unpack value from (key, value) pair from DashMap
            let dataset = dataset.value();

            // Valid samples. THIS ASSUMES ALL FIELDS HAVE THE SAME VALID ENTRIES
            let num_samples: usize = dataset.field_1.fold(0, |acc, &x| if x != SUPERMAG_NAN { acc + 1 } else { acc } );

            // If there are no valid entries, abort. Do not clean, and do not modify dashmap
            if num_samples == 0 {
                println!("Station {} index {} aborting", &dataset.station_name, index);
                return ()
            }

            // Clean the fields and find valid entries
            let (valid_entries_1, clean_field_1): (Array1<bool>, TimeSeries) = {
                let (entries, field): (Vec<bool>, Vec<f32>) = dataset.field_1
                    .iter()
                    .map(|&x| if x != SUPERMAG_NAN { (true, x) } else { (false, 0.0) })
                    .unzip();
                (Array1::from_vec(entries), Array1::from_vec(field)) 
            };
            let (valid_entries_2, clean_field_2): (Array1<bool>, TimeSeries) = {
                let (entries, field): (Vec<bool>, Vec<f32>) = dataset.field_2
                    .iter()
                    .map(|&x| if x != SUPERMAG_NAN { (true, x) } else { (false, 0.0) })
                    .unzip();
                (Array1::from_vec(entries), Array1::from_vec(field)) 
            };

            // // Mark valid seconds
            // let num_seconds_per_chunk: usize = local_wn.len();
            // let first_second: usize = index*num_seconds_per_chunk;
            // valid_entries_1
            //     .iter()
            //     .fold(first_second, |sec, valid| {
            //         assert!(local_valid_seconds.insert(sec, *valid as usize).is_none(), "Error: duplicate entry in valid_seconds");
            //         sec + 1
            //     });

            if cfg!(debug_assertions) {
                println!(
                    "Station {} has {} null entries ({:.1}%) for chunk index {}",
                    &dataset.station_name,
                    dataset.field_1.len() - num_samples,
                    (dataset.field_1.len() - num_samples) as f64 / dataset.field_1.len() as f64,
                    index,
                );
                
                println!(
                    "Station {} has {:?} min/max entries for chunk index {}",
                    &dataset.station_name,
                    clean_field_1.fold((0.0, 0.0), |mut acc, &x| {
                        if acc.0 > x {
                            acc.0 = x; 
                        }
                        if acc.1 < x {
                            acc.1 = x;
                        }
                        acc
                    }),
                    index,
                );
            }

            // Calculate weights (NOTE: these were swapped at some point due to a typo idenitified in the paper)
            let n_weight: f32 = (clean_field_2.dot(&clean_field_1) / num_samples as f32).recip();
            let e_weight: f32 = (clean_field_1.dot(&clean_field_2) / num_samples as f32).recip();

            let wn_weight: TimeSeries = valid_entries_1.map(|&is_valid| if is_valid { n_weight } else { 0.0 });
            let we_weight: TimeSeries = valid_entries_2.map(|&is_valid| if is_valid { e_weight } else { 0.0 });

            // Add to local hashmaps and time series
            let clean_station_name: String = dataset.station_name.clone().split("/").collect::<Vec<_>>().get(2).unwrap().to_string();
            local_hashmap_n.insert(clean_station_name.clone(), n_weight);
            local_hashmap_e.insert(clean_station_name, e_weight);
            local_wn.add_assign(&wn_weight);
            local_we.add_assign(&we_weight);
        });
}


/// `starting_value` and `size` specify a range of integers.
/// This function checks if `chunk` is in that range.
pub(crate) fn in_longest_subset(chunk: usize, size: usize, starting_value: usize) -> bool {

    (chunk >= starting_value)
    && (chunk < (starting_value + size))
}


fn next_power_of_two(number: usize) -> usize {

    // Initialize result with 2
    let mut next_power = 2;

    // Find next power of two iteratively 
    while next_power < number {
        next_power *= 2;
    }

    // Return next power of two
    next_power
}