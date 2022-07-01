use crate::{
    constants::*,
    utils::loader::*,
    utils::async_balancer::*,
    theory::*,
};
use ndarray::{Array1, arr1};
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use std::ops::AddAssign;
use dashmap::try_result::TryResult;
use std::ops::Range;
// use rayon::iter::ParallelIterator;


type Index = usize;
type Weight = f32;
type StationName = String;
type TimeSeries = Array1<f32>;
type ComplexSeries = Array1<ndrustfft::Complex<f32>>;

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
    pub async fn new(stationarity: Stationarity, theory: T, balancer: &mut Manager<()>) -> Arc<Self>
    {

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

        // Initialize empty data vector
        let data_vector = Arc::new(DashMap::new());

        // Wrap theory in an Arc
        let theory = Arc::new(theory);

        // Calculate local set of tasks
        let set: Range<usize> = 0..loader.semivalid_chunks.len();
        let local_set: Vec<usize> = balancer.local_set(&set.collect());

        for entry in local_set {

            // Clone all relevant Arcs
            let local_weights = weights.clone();
            let local_loader = loader.clone();
            let local_theory = theory.clone();
            let local_data_vector = data_vector.clone();

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
                    let local_hashmap_n: Arc<DashMap<StationName, Weight>> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_hashmap_e: Arc<DashMap<StationName, Weight>> = Arc::new(DashMap::with_capacity(datasets.len()));
                    let local_wn: Arc<RwLock<TimeSeries>> = Arc::new(RwLock::new(arr1(&vec![0.0_f32; days * SECONDS_PER_DAY])));
                    let local_we: Arc<RwLock<TimeSeries>> = Arc::new(RwLock::new(arr1(&vec![0.0_f32; days * SECONDS_PER_DAY])));
                    
                    datasets
                        .iter()
                        .for_each(|dataset| {
                            // for mut dataset in datasets.iter_mut() {
                        
                            // Unpack value from (key, value) pair from DashMap
                            let dataset = dataset.value();

                            // Valid samples
                            let num_samples: usize = dataset.field_1.fold(0, |acc, &x| if x != SUPERMAG_NAN { acc + 1 } else { acc } );

                            // If there are no valid entries, abort. Do not clean, and do not modify dashmap
                            if num_samples == 0 {
                                println!("Station {} index {} aborting", &dataset.station_name, index);
                                return ()
                            }
        

                            // Clean the fields
                            let clean_field_1 = dataset.field_1.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });
                            let clean_field_2 = dataset.field_2.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });

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
                            let wn_weight: TimeSeries = clean_field_2.map(|&x| if x != 0.0 { n_weight } else { 0.0 });
                            let we_weight: TimeSeries = clean_field_1.map(|&x| if x != 0.0 { e_weight } else { 0.0 });

                            // Add to local hashmaps and time series
                            let clean_station_name: String = dataset.station_name.clone().split("/").collect::<Vec<_>>().get(2).unwrap().to_string();
                            local_hashmap_n.insert(clean_station_name.clone(), n_weight);
                            local_hashmap_e.insert(clean_station_name, e_weight);
                            local_wn.write().unwrap().add_assign(&wn_weight);
                            local_we.write().unwrap().add_assign(&we_weight);
                        });

                    println!("Finished weights");
                    println!("local_hashmap_n has {} entries", local_hashmap_n.len());
                    println!("local_hashmap_n has {} entries", local_hashmap_n.len());


                    // e.g. all stations have nans for all values for this chunk
                    if local_hashmap_n.len() == 0 {
                        println!("Invalid chunk. Proceeding to next chunk");
                        return ()
                    }
                    // Calculate projections
                    println!("calculating projections");
                    let projections = local_theory.calculate_projections(
                        Arc::clone(&local_hashmap_n),
                        Arc::clone(&local_hashmap_e),
                        &*local_wn.read().unwrap(),
                        &*local_we.read().unwrap(),
                        datasets
                    );

                    // Add to dashmap
                    local_weights.n.insert(index, Arc::try_unwrap(local_hashmap_n).expect("Somehow an Arc survived"));
                    local_weights.e.insert(index, Arc::try_unwrap(local_hashmap_e).expect("Somehow an Arc survived"));
                    loop {
                        match local_weights.wn.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },    //array.add_assign(&wn_weight); break },
                            TryResult::Absent => { local_weights.wn.insert(index, Arc::try_unwrap(local_wn).expect("Somehow an Arc survived").into_inner().unwrap()); break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }
                    loop {
                        match local_weights.we.try_get_mut(&index) {
                            TryResult::Present(_) => { panic!("some other thread somehow worked on this chunk") },    //array.add_assign(&we_weight); break },
                            TryResult::Absent => { local_weights.we.insert(index, Arc::try_unwrap(local_we).expect("Somehow an Arc survived").into_inner().unwrap()); break },
                            TryResult::Locked => { tokio::task::yield_now().await },
                        }
                    }

                    // Calculate data vector
                    println!("calculating data vector");
                    // let total_time = SECONDS_PER_DAY as f32 * days as f32;
                    // let frequencies = &[300.0/total_time, 400.0/total_time, 500.0/total_time];
                    let local_data_vector_value = local_theory.calculate_data_vector(
                        projections,
                    );

                    // Insert data vector
                    local_data_vector.insert(index, local_data_vector_value);

                }).await.unwrap()})
            );
           
        }

        // Wait for all tasks on this node to finish
        println!("About to buffer_await");
        let result: Vec<()> = balancer.buffer_await().await;
        assert!(result.iter().all(|&x| x == ()));

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
            data_vector,
        })
    }
}




#[derive(Copy, Clone)]
pub enum Stationarity {
    Yearly,
    Daily(usize),
}