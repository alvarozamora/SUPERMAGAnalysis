use super::*;
use std::collections::HashMap;
use dashmap::DashMap;
// use goertzel_filter::dft;
use std::sync::Arc;
use crate::utils::{
    loader::Dataset,
};
use std::ops::{Mul, Div, Add};
use ndrustfft::{ndfft_r2c, R2cFftHandler};


type DarkPhotonVecSphFn = Arc<dyn Fn(f32, f32) -> f32 + Send + 'static + Sync>;
/// Contains all necessary things
#[derive(Clone)]
pub struct DarkPhoton {
    kinetic_mixing: f64,
    vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>>,
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


impl DarkPhoton {

    /// This initilaizes a `DarkPhoton` struct. This struct is to be used during an analysis to produce
    /// data vectors and signals after implementing `Theory`.
    pub fn initialize(kinetic_mixing: f64) -> Self {

        // Calculate vec_sphs at each station
        // let vec_sph_fns = Arc::new(vector_spherical_harmonics(DARK_PHOTON_MODES.clone().into_boxed_slice()));
        // Manual override to remove prefactors
        let mut vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>> = Arc::new(DashMap::new());

        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[0].clone(),
            Arc::new(|theta: f32, phi: f32| -> f32 {
                phi.sin()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[1].clone(),
            Arc::new(|theta: f32, phi: f32| -> f32 {
                phi.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[2].clone(),
            Arc::new(|theta: f32, phi: f32| -> f32 {
                phi.cos() * theta.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[3].clone(),
            Arc::new(|theta: f32, phi: f32| -> f32 {
                - phi.sin() * theta.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[4].clone(),
            Arc::new(|theta: f32, phi: f32| -> f32 {
                theta.sin()
            }));

        // Sanity check on our own work
        assert_eq!(vec_sph_fns.len(), Self::NONZERO_ELEMENTS);

        DarkPhoton {
            kinetic_mixing,
            vec_sph_fns,
        }

    }
}


impl Theory for DarkPhoton {

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

            // size of dataset
            let size = chunk_dataset.iter().next().unwrap().value().field_1.len();

            // Here we iterate thrhough weights_n and not chunk_dataset because
            // stations in weight_n are a subset (filtered) of those in chunk_dataset.
            // Could perhaps save memory by dropping coressponding invalid datasets in chunk_dataset.
            let combined_time_series: TimeSeries = weights_n
                .iter()
                .map(|key_value| {

                    // Unpack (key, value) pair
                    // Here, key is StationName and value = dataset
                    let station_name = key_value.key();
                    let dataset = chunk_dataset.get(station_name).unwrap();

                    // Get product of relevant component of vector spherical harmonics and of the magnetic field. 
                    let relevant_product = match nonzero_element.assc_mode {
                        (_, component) => {
                            
                            // Get relevant vec_sph_fn
                            let vec_sph_fn = self.vec_sph_fns.get(&nonzero_element).unwrap();

                            // TODO: Change these to match definitions in the paper
                            // let relevant_vec_sph = match component {
                            //     Component::PolarReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].re,
                            //     Component::PolarImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].im,
                            //     Component::AzimuthReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].re,
                            //     Component::AzimuthImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].im,
                            //     _ => panic!("not included in dark photon"),
                            // };
                            // Manual Override
                            let relevant_vec_sph = vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32);

                            // Note that this multiplies the magnetic field by the appropriate weights, so it's not quite the measured magnetic field
                            let relevant_mag_field = match component {
                                Component::PolarReal => (&dataset.field_1).mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::PolarImag => (&dataset.field_1).mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::AzimuthReal => (&dataset.field_2).mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                Component::AzimuthImag => (&dataset.field_2).mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                _ => panic!("not included in dark photon"),
                            };

                            relevant_vec_sph * relevant_mag_field
                        }
                    };

                    (station_name.clone(), relevant_product)
                })
                .collect::<HashMap<StationName, TimeSeries>>()
                .iter()
                .fold(TimeSeries::default(size), |acc, (_key, series)| acc.add(series));
            
            // Insert combined time series for this nonzero element for this chunk, ensuring no duplicate entry
            assert!(projection_table.insert(nonzero_element.clone(), combined_time_series).is_none(), "Somehow made a duplicate entry");
        }

        projection_table
    }  


    fn calculate_data_vector(
        &self,
        projections: DashMap<NonzeroElement, TimeSeries>,
    ) -> DashMap<NonzeroElement, ComplexSeries> {

        // Initialize fft planner
        let mut handler = None;

        // For every element in `projections`, do the FFT over the TimeSeries
        projections
            .iter()
            .map(|key_value| {

                // Unpack (key, value) pair from projections. value here is the nonzero projections element X
                let (key, value) = key_value.pair();

                // Initialize fft_planner if needed. This is done here because the length is not known in advance.
                if handler.is_none() { 
                    handler = Some(R2cFftHandler::new(value.len())); 
                }

                // Calculate fft
                let mut fft_value = ComplexSeries::zeros(value.len()/2 + 1);
                ndfft_r2c(&value, &mut fft_value, handler.as_mut().unwrap(), 0);
                
                (key.clone(), fft_value)
            
            })
            .collect()
    }
}