use crate::{
    constants::*,
    utils::loader::*,
    utils::balancer::*,
};
use ndarray::Array1;//, ScalarOperand};
use dashmap::DashMap;
use std::thread::spawn;
use std::sync::Arc;
use std::rc::Rc;
use std::ops::Add;
use std::ops::AddAssign;

type StationName = String;
type TimeSeries<T> = Array1<T>;

/// This holds the weights (inverse white noise). These weights are a measurement 
/// intrinsic to the dataset and do not depend on the theory.
pub struct Weights {
    pub n: DashMap<StationName, f32>,
    pub e: DashMap<StationName, f32>,
    pub wn: DashMap<StationName, TimeSeries<f32>>,
    pub we: DashMap<StationName, TimeSeries<f32>>,
}

impl Weights {

    /// This calculates the inverse white noise (i.e. weights) for a given dataset,
    /// as per eq 13 and 14 in the paper. 
    pub fn new_from_year(year: usize, balancer: &Balancer) -> Arc<Self> {

        // Grab dataset loader for the year
        let mut loader: DatasetLoader = DatasetLoader::new_from_year(year);

        // Initialize empty `Weights`
        let weights: Arc<Weights> = Arc::new(Weights {
                n: DashMap::new(),
                e: DashMap::new(),
                wn: DashMap::new(),
                we: DashMap::new(),
        });


        // Initialize Handles
        let mut handles: Handles<_> = Handles::new();

        while loader.station_files.len() > 0 {

            // Grab station file
            let station_file: String = loader.station_files.pop().unwrap();

            // Get Arc to dashmap/hashmap
            let local_weights = weights.clone();
            let coordinate_map = loader.coordinate_map.clone();



            // Spawn a new thread for each `station_file`
            handles.add(
                spawn(move || {
                
                    // Load dataset
                    let dataset: Dataset = DatasetLoader::load(year, station_file, coordinate_map);

                    // Valid samples
                    let num_samples: usize = dataset.field_1.fold(0, |acc, &x| if x != SUPERMAG_NAN { acc + 1 } else { acc } );
                    println!(
                        "Station {} has {} null entries ({:.1}%) for year {}",
                        &dataset.station_name,
                        dataset.field_1.len() - num_samples,
                        (dataset.field_1.len() - num_samples) as f64 / dataset.field_1.len() as f64,
                        year,
                    );

                    // Clean the fields
                    let clean_field_1 = dataset.field_1.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });
                    let clean_field_2 = dataset.field_2.map(|&x| if x != SUPERMAG_NAN { x } else { 0.0 });

                    if cfg!(debug_assertions) {
                        println!(
                            "Station {} has {:?} min/max entries for year {}",
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
                            year,
                        );
                    }

                    // Calculate weights
                    let n_weight: f32 = (clean_field_1.dot(&clean_field_1) / num_samples as f32).recip();
                    let e_weight: f32 = (clean_field_2.dot(&clean_field_2) / num_samples as f32).recip();
                    let wn_weight: TimeSeries<f32> = clean_field_1.map(|&x| if x != 0.0 { n_weight } else { 0.0 });
                    let we_weight: TimeSeries<f32> = clean_field_2.map(|&x| if x != 0.0 { e_weight } else { 0.0 });

                    // Add to dashmap
                    local_weights.n.insert(dataset.station_name.clone(), n_weight);
                    local_weights.e.insert(dataset.station_name.clone(), e_weight);
                    if local_weights.wn.contains_key(&dataset.station_name) {
                        let mut array = local_weights.wn.get_mut(&dataset.station_name).unwrap();
                        array.add_assign(&wn_weight);
                    } else {
                        local_weights.wn.insert(dataset.station_name.clone(), wn_weight);
                    }
                    if local_weights.we.contains_key(&dataset.station_name) {
                        let mut array = local_weights.we.get_mut(&dataset.station_name).unwrap();
                        array.add_assign(&we_weight);
                    } else {
                        local_weights.we.insert(dataset.station_name.clone(), we_weight);
                    }
                    
            }));
           
        }

        // Wait for threads to finish
        handles.wait();

        weights
    }
}