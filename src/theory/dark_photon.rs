use super::*;
use std::collections::HashMap;
use dashmap::DashMap;
use ndarray::{Array1, ScalarOperand};
use goertzel_filter::dft;
use std::slice::Iter;
use std::sync::Arc;
use rayon::iter::*;
use crate::utils::{
    loader::Dataset,
    // fft::find_nearest_frequency_1s,
};
use std::ops::{Mul, Div, Add};

/// Contains all necessary things
#[derive(Clone)]
pub struct DarkPhoton {
    kinetic_mixing: f64,
    vec_sph_fns: Arc<DashMap<Mode, VecSphFn>>,
}

lazy_static! {
    static ref DARK_PHOTON_MODES: Vec<Mode> = vec![
        Mode(1,  0),
        Mode(1, -1),
        Mode(1,  1),
    ];

    static ref DARK_PHOTON_NONZERO_ELEMENTS: Vec<NonzeroElement> = vec![
        NonzeroElement {
            index: 1,
            name: Some(String::from("X1")),
            assc_mode: (Mode(1, -1), Component::PolarReal)
        },
        NonzeroElement {
            index: 2,
            name: Some(String::from("X2")),
            assc_mode: (Mode(1, -1), Component::PolarImag)
        },
        NonzeroElement {
            index: 3,
            name: Some(String::from("X3")),
            assc_mode: (Mode(1, -1), Component::AzimuthReal)
        },
        NonzeroElement {
            index: 4,
            name: Some(String::from("X4")),
            assc_mode: (Mode(1, -1), Component::AzimuthImag)
        },
        NonzeroElement {
            index: 5,
            name: Some(String::from("X5")),
            assc_mode: (Mode(1,  0), Component::AzimuthReal)
        },
    ];
}


impl DarkPhoton
{

    /// This initilaizes a `DarkPhoton` struct. This struct is to be used during an analysis to produce 
    /// data vectors and signals after implementing `Theory`.
    pub fn initialize(kinetic_mixing: f64) -> Self {

        // Calculate vec_sphs at each station
        let vec_sph_fns = Arc::new(vector_spherical_harmonics(DARK_PHOTON_MODES.clone().into_boxed_slice()));

        DarkPhoton {
            kinetic_mixing,
            vec_sph_fns,
        }

    }
}


impl Theory for DarkPhoton
{

    // const MODES: Modes = DARK_PHOTON_MODES;
    // const NONZERO_ELEMENTS: NonzeroElements = DARK_PHOTON_NONZERO_ELEMENTS;

    fn calculate_projections(
        &self,
        weights_n: Arc<DashMap<StationName, f32>>,
        weights_e: Arc<DashMap<StationName, f32>>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        chunk_dataset: DashMap<StationName, Dataset>,
    ) -> DashMap<NonzeroElement, TimeSeries> {

        // Initialize projection table
        let projection_table = DashMap::new();



        for nonzero_element in DARK_PHOTON_NONZERO_ELEMENTS.iter() {

            let (tx, rx) = std::sync::mpsc::channel();

            let combined_time_series: TimeSeries = chunk_dataset
                .iter()
                .map(|key_value| {

                    // Unpack (key, value) pair
                    // Here, key is StationName and value = dataset
                    let (station_name, dataset) = key_value.pair();

                    tx.send(dataset.field_1.len()).unwrap();

                    // Get product of relevant component of vector spherical harmonics and of the magnetic field. 
                    let relevant_product = match nonzero_element.assc_mode {
                        (mode, component) => {
                            
                            // Get relevant vec_sph_fn
                            let vec_sph_fn = self.vec_sph_fns.get(&mode).unwrap();

                            let relevant_vec_sph = match component {
                                Component::PolarReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].re,
                                Component::PolarImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].im,
                                Component::AzimuthReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].re,
                                Component::AzimuthImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].im,
                                _ => panic!("not included in dark photon"),
                            };

                            // Note that this multiplies the magnetic field by the appropriate weights, so it's not quite the measured magnetic field
                            let relevant_mag_field = match component {
                                Component::PolarReal => dataset.field_1.clone().mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::PolarImag => dataset.field_1.clone().mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::AzimuthReal => dataset.field_2.clone().mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                Component::AzimuthImag => dataset.field_2.clone().mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                _ => panic!("not included in dark photon"),
                            };

                            relevant_vec_sph * relevant_mag_field
                        }
                    };

                    (station_name.clone(), relevant_product)
                })
                .collect::<HashMap<StationName, TimeSeries>>()
                .iter()
                .fold(TimeSeries::default(rx.recv().unwrap()), |acc, (_key, series)| acc.add(series));

            assert!(projection_table.insert(nonzero_element.clone(), combined_time_series).is_none(), "Somehow made a duplicate entry");
        }

        projection_table
    }  


    fn calculate_data_vector(
        &self,
        projections: DashMap<NonzeroElement, TimeSeries>,
        frequencies: &[Frequency],
        total_time: f32,
    ) -> DashMap<(FrequencyIndex, NonzeroElement), DFTValue> {


        // // Find closest frequencies for given frequencies
        // let closest_frequencies = frequencies
        //     .iter()
        //     .map(|&freq| find_nearest_frequency_1s(freq, total_time))
        //     .collect::<Vec<Frequency>>();

        // For each of these frequencies, and for each element, calculate DFT at that frequency
        let result = DashMap::with_capacity(frequencies.len() * projections.len());

        // For every element in `projections`, do the DFT for every frequency in `closest_frequencies`
        projections
            .iter()
            .for_each(|key_value| {

                // Unpack (key, value) pair from projections. value here is the nonzero projections element X
                let (key, value) = key_value.pair();

                
                // for (frequency_index, frequency) in closest_frequencies.iter().enumerate() {
                for (frequency_index, frequency) in frequencies.iter().enumerate() {

                    // Find the key and value to insert
                    let key: (FrequencyIndex, NonzeroElement) = (frequency_index, key.clone());

                    // value is the dft at the given frequency
                    // TODO: ensure that the dft function below actually computes the closest frequency we want.
                    let value = dft(&value.map(|&x| x as f64).to_vec(), *frequency as f64);
                    let value: DFTValue = Complex::new(value.re as f32, value.im as f32);

                    // Insert (key, value)
                    assert!(result.insert(key, value).is_none(), "Somehow made a duplicate entry");
                }
            
            });

        result
    }
}