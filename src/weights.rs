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
                    debug_print!("{:?} working on index {index}", std::thread::current().id());

                    // Load datasets for this chunk
                    let datasets: DashMap<StationName, Dataset> = local_loader.load_chunk(index).await.unwrap();

                    // e.g. on a year boundary where all stations change
                    if datasets.len() == 0 {
                        debug_print!("Empty chunk. Proceeding to next chunk");
                        return ()
                    }

                    // Local hashmaps and time series
                    let local_hashmap_n: DashMap<StationName, Weight> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_hashmap_e: DashMap<StationName, Weight> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_wn: RwLock<TimeSeries> = Arc::new(RwLock::new(Array1::from_vec(vec![0.0_f32; days * SECONDS_PER_DAY])));
                    let local_we: RwLock<TimeSeries> = Arc::new(RwLock::new(Array1::from_vec(vec![0.0_f32; days * SECONDS_PER_DAY])));
                    
                    debug_print!("Finished weights for index {index}");
                    debug_print!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    debug_print!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    // e.g. all stations have nans for all values for this chunk
                        println!("Invalid chunk, as there are less than {} stations with data in this chunk. Proceeding to next chunk", T::MIN_STATIONS);
                        return ()
                    } else if local_wn.iter().any(|&x| x == 0.0_f32) {
                        println!("Invalid chunk, as there is at least one time slot with a normalization weight of 0.0");
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


/// Calculate the coherence times
fn coherence_times(total_time: f64, threshold: f64) -> Vec<usize> {

    // Initialize return value
    let mut times = vec![];

    // Find max in in eq (??)
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
            let start = if bin_index == 0 { 0 } else { 1 };
            let mut end = (higher / lower) as usize;
            if end as f64 * lower == higher { end -= 1; }

            assert!(end as f64 * lower < higher, "highest frequency in bin is higher than lowest in next bin");

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
            multiples: 1..=highest_frequency,
        }
    });

    // Return bin
    frequency_bins
}

#[test]
fn test_frequencies_from_coherence_times() {

    // The function that generates these returns them in descending order. lets do the same
    let coherence_times = vec![100, 4];

    // Calculate frequency bins
    let frequency_bins = frequencies_from_coherence_times(&coherence_times);

    frequency_bins
        .iter()
        .zip(
        vec![
            FrequencyBin {
                lower: 1.0 / 100.0,
                multiples: 0..=24,
            },
            FrequencyBin {
                lower: 1.0 / 4.0,
                multiples: 1..=3,
            }
        ].iter()
    ).for_each(|(bin_a, bin_b)| {
        assert_eq!(*bin_a, *bin_b, "frequency bins are not the same");
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
            // TODO: calculate indices of nans
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



fn in_longest_subset(chunk: usize, size: usize, starting_value: usize) -> bool {

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