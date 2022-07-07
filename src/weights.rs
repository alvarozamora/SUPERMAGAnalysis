use crate::{
    constants::*,
    utils::*,
    utils::loader::*,
    utils::async_balancer::*,
    theory::*,
};
use mpi::{topology::Communicator, traits::Root};
use ndarray::{s, Array1, arr1, ArrayViewMut, Array2};
use dashmap::DashMap;
use std::{sync::Arc, ops::RangeInclusive};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::ops::AddAssign;
use dashmap::try_result::TryResult;
use std::ops::Range;
use rayon::iter::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use mpi::point_to_point::{Source, Destination};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

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
struct FrequencyBin {
    lower: f64,
    multiples: RangeInclusive<usize>,
}

/// This holds the weights (inverse white noise). These weights are a measurement 
/// intrinsic to the dataset and do not depend on the theory.
pub struct Weights {
    pub n: DashMap<Index, DashMap<StationName, Weight>>,
    pub e: DashMap<Index, DashMap<StationName, Weight>>,
    pub wn: DashMap<Index, TimeSeries>,
    pub we: DashMap<Index, TimeSeries>,
    pub stationarity: Stationarity,
}


pub struct Analysis<T: Theory + Send> {
    pub weights: Arc<Weights>,
    pub theory: T,
    pub projections: Arc<DashMap<Index, DashMap<NonzeroElement, TimeSeries>>>,
    pub data_vector: DashMap<Index, DashMap<NonzeroElement, ComplexSeries>>,
}

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
    pub async fn new(stationarity: Stationarity, theory: T, balancer: &mut Manager<()>) -> Arc<Self> {

        println!("Running Analysis");

        // Number of days per chunk
        let days: usize = match stationarity {
            Stationarity::Yearly => todo!(),
            Stationarity::Daily(days) => days,
        };

        // Grab dataset loader for the chunks
        let loader = Arc::new(DatasetLoader::new(days));

        // Initialize empty `Weights`
        let weights: Arc<Weights> = Arc::new(Weights {
                n: DashMap::new(),
                e: DashMap::new(),
                wn: DashMap::new(),
                we: DashMap::new(),
                stationarity,
        });

        // Initialize empty data vector, projections
        let data_vector = Arc::new(DashMap::new());
        let projections = Arc::new(DashMap::new());

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
            let local_data_vector: Arc<_> = data_vector.clone();
            let local_projections: Arc<_> = projections.clone();

            // Create a new task for each `chunk`
            balancer.
                task(Box::new( async move { tokio::task::spawn( async move {

                    // Get chunk index
                    let index: Index = local_loader.semivalid_chunks.entry(entry).into_key();
                    println!("{:?} working on index {index}", std::thread::current().id());

                    // Load datasets for this chunk
                    let datasets: DashMap<StationName, Dataset> = local_loader.load_chunk(index).await.unwrap();

                    // e.g. on a year boundary where all stations change
                    if datasets.len() == 0 {
                        println!("Empty chunk. Proceeding to next chunk");
                        return ()
                    }

                    // Local hashmaps and time series
                    let local_hashmap_n: DashMap<StationName, Weight> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_hashmap_e: DashMap<StationName, Weight> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_wn: RwLock<TimeSeries> = Arc::new(RwLock::new(Array1::from_vec(vec![0.0_f32; days * SECONDS_PER_DAY])));
                    let local_we: RwLock<TimeSeries> = Arc::new(RwLock::new(Array1::from_vec(vec![0.0_f32; days * SECONDS_PER_DAY])));
                    
                    let secs_per_chunk: usize = datasets.iter().next().unwrap().value().field_1.len();
                    let mut counter = Array1::<usize>::zeros(secs_per_chunk);

                    calculate_weights_for_chunk();

                    println!("Finished weights");
                    println!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    println!("local_hashmap_n has {} entries", local_hashmap_n.len());


                    for sec in 0..counter.len() {

                        // For this test, index = day
                        let day = index;
                        let secs_since = day * SECONDS_PER_DAY + sec;

                        local_valid_secs.insert(secs_since, counter[sec]);
                    }

                    // e.g. all stations have nans for all values for this chunk
                    if local_hashmap_n.len() == 0 { // && cfg!(debug_assertions) {
                        println!("Invalid chunk. Proceeding to next chunk");
                        return ()
                    }

                    // Calculate projections for this chunk
                    println!("calculating projections");
                    let chunk_projections: DashMap<NonzeroElement, TimeSeries> = local_theory.calculate_projections(
                        Arc::clone(&local_hashmap_n),
                        Arc::clone(&local_hashmap_e),
                        &*local_wn.read(),
                        &*local_we.read(),
                        datasets
                    );
                    assert!(local_projections.insert(index, chunk_projections).is_none(), "A duplicate entry was made");

                    // Add to dashmap
                    local_weights.n.insert(index, Arc::try_unwrap(local_hashmap_n).expect("An Arc survived"));
                    local_weights.e.insert(index, Arc::try_unwrap(local_hashmap_e).expect("An Arc survived"));
                    loop {
                        match local_weights.wn.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },    //array.add_assign(&wn_weight); break },
                            TryResult::Absent => { local_weights.wn.insert(index, Arc::try_unwrap(local_wn).expect("An Arc survived").into_inner()); break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }
                    loop {
                        match local_weights.we.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },    //array.add_assign(&we_weight); break },
                            TryResult::Absent => { local_weights.we.insert(index, Arc::try_unwrap(local_we).expect("An Arc survived").into_inner()); break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }
                }).await.unwrap()})
            );
           
        }

        // Wait for all tasks on this node to finish, and check that they all succeeded
        println!("About to buffer_await");
        let result: Vec<()> = balancer.buffer_await().await;
        assert!(result.iter().all(|&x| x == ()));

        // TODO: GLOBAL SYNC HERE


        // // After rechunking into coherence times, do FFT + noise + bayesian analysis + anything else for every 
        // {
        //     const COHERENCE_TIMES: usize = 1000;
        //     const THRESHOLD: f64 = 0.03;
        //     let coherence_times: [usize; COHERENCE_TIMES] = find_coherence_times(); // This must be an integer multiple of 1s, hence usize
        
        //     for coherence_time in coherence_times {
        
        //         // This is the chunked data for this particular coherence time
        //         let coherence_chunk = rechunk_into(
        //             projections, 
        //             stationarity,
        //             coherence,
        //         );

        //         // TODO: FFT

        //         // TODO: noise spectral analysis + bayesian analysis
        
        //     }
            
        

        // Calculate data vector
        // println!("calculating data vector");
        // let total_time = SECONDS_PER_DAY as f32 * days as f32;
        // let frequencies = &[300.0/total_time, 400.0/total_time, 500.0/total_time];
        //     let local_data_vector_value = local_theory.calculate_data_vector(
        //         projections,
        //     );

        //     // Insert data vector
        //     local_data_vector.insert(index, local_data_vector_value);
        // }

        // Unwrap data_vector and theory
        let data_vector = Arc::try_unwrap(data_vector).expect("Somehow an Arc survived");
        let theory = if let Ok(theory) = Arc::try_unwrap(theory) {
            theory
        } else {
            panic!()
        };

        // Return Analysis
        Arc::new(Analysis {
            weights,
            theory,
            projections,
            data_vector,
            valid_secs,
        })
    }
}




#[derive(Copy, Clone, Debug)]
pub enum Stationarity {
    Yearly,
    Daily(usize),
}

#[derive(Copy, Clone, Debug)]
pub enum Coherence {
    Days(usize),
    Seconds(usize),
}

const RESOLUTION: f32 = 1.0;


// Calculate the coherence 
fn coherence_times(total_time: f32, threshold: f32, min_periods: usize) -> Vec<usize> {

    // Initialize return value
    let mut times = vec![];

    // Find max in in eq (??) TODO: check rounding is safe
    let max_n: usize = (-0.5*(1_000_000.0 / total_time).ln()/(1.0 + threshold).ln()).round() as usize;

    for n in 0..=max_n {

        // Find the raw coherence time
        let raw_coherence_time: f32 = total_time as f32 / (1.0 + threshold as f32).powi(2*n as i32);
        let rounded = raw_coherence_time.round() as usize; //  TODO: check rounding is safe

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

    times
}


async fn calculate_weights_for_chunk(
    local_hashmap_n: Arc<DashMap<>>
) {
    


    datasets
        .iter()
        .for_each(|dataset| {
        
            // Unpack value from (key, value) pair from DashMap
            let dataset = dataset.value();

            // Valid samples. THIS ASSUMES ALL FIELDS HAVE THE SAME VALID ENTRIES
            let num_samples: usize = dataset.field_1.fold(0, |acc, &x| if x != SUPERMAG_NAN { acc + 1 } else { acc } );

            // If there are no valid entries, abort. Do not clean, and do not modify dashmap
            // if num_samples == 0 {
            //     println!("Station {} index {} aborting", &dataset.station_name, index);
            //     return ()
            // }

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
            // TODO: calculate indices of nans
            let wn_weight: TimeSeries = valid_entries_1.map(|&is_valid| if is_valid { n_weight } else { 0.0 });
            let we_weight: TimeSeries = valid_entries_2.map(|&is_valid| if is_valid { e_weight } else { 0.0 });

            // Add to local hashmaps and time series
            let clean_station_name: String = dataset.station_name.clone().split("/").collect::<Vec<_>>().get(2).unwrap().to_string();
            local_hashmap_n.insert(clean_station_name.clone(), n_weight);
            local_hashmap_e.insert(clean_station_name, e_weight);
            local_wn.write().add_assign(&wn_weight);
            local_we.write().add_assign(&we_weight);
        });
}


/// Given a number `n`, this function finds its largest prime factor
fn maxprime(n: usize) -> usize {

    // Deal with base case
    if n == 1 { return 1 }

    // Find upper_bound for checks
    let upper_bound = (n as f64).sqrt() as usize;

    // Iterate through all odd numbers between 2 and the upper_bound
    for i in (2..=2).chain((3..=upper_bound).step_by(2)) {
        if n % i == 0 {
            return maxprime(n/i) 
        }
    }

    // Because we are iterating up, this will return the largest prime factor
    return n
}


#[test]
fn test_maxprime_small_prime() {

    assert_eq!(maxprime(7), 7);
}

#[test]
fn test_maxprime_bigger_prime() {

    assert_eq!(maxprime(53), 53);
}


#[test]
fn test_maxprime_prime_squared() {

    assert_eq!(maxprime(49), 7);
}


#[test]
fn test_maxprime_multiple_primes() {

    assert_eq!(maxprime(24), 3);
}


#[test]
fn test_maxprime_multiple_primes_prime_bigger_number() {

    assert_eq!(maxprime(150), 5);
}
