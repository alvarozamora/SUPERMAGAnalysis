use super::*;
use std::collections::HashMap;
use dashmap::DashMap;
// use goertzel_filter::dft;
use std::sync::Arc;
use crate::{utils::{
    loader::{Dataset, Index},
    approximate_sidereal, coordinates::Coordinates,
}, constants::{SECONDS_PER_DAY, SIDEREAL_DAY_SECONDS}, weights::Weights};
use std::ops::{Mul, Div, Add};
use ndrustfft::{ndfft_r2c, R2cFftHandler, FftHandler, ndfft};
use rayon::prelude::*;
use ndarray::s;
use ndrustfft::Complex;
use std::f64::consts::PI;
use std::f32::consts::PI as SINGLE_PI;

type DarkPhotonVecSphFn = Arc<dyn Fn(f32, f32) -> f32 + Send + 'static + Sync>;
/// Contains all necessary things
#[derive(Clone)]
pub struct DarkPhoton {
    kinetic_mixing: f64,
    vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>>,
}

impl Debug for DarkPhoton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("{}", self.kinetic_mixing).as_str())
    }
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

    const MIN_STATIONS: usize = 3;
    const NONZERO_ELEMENTS: usize = 5;

    fn get_nonzero_elements() -> HashSet<NonzeroElement> {

        let mut nonzero_elements = HashSet::new();

        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[0].clone());
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[1].clone());
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[2].clone());
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[3].clone());
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[4].clone());

        nonzero_elements
    }
    fn calculate_projections(
        &self,
        weights_n: &DashMap<StationName, f32>,
        weights_e: &DashMap<StationName, f32>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        chunk_dataset: &DashMap<StationName, Dataset>,
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
        projections_complete: &DashMap<NonzeroElement, TimeSeries>,
        local_set: &Vec<(usize, FrequencyBin)>,
    ) -> DashMap<usize, DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>>> {


        // Parallelize local_set on this rank
        let data_vector_dashmap: DashMap<usize, DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>>> = DashMap::new();

        local_set
            .par_iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* FrequencyBin */)| {
                
                let inner_dashmap: DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>> = DashMap::new();

                // Parallelize over all nonzero elements in the theory
                projections_complete
                    .par_iter()
                    .for_each_with(&inner_dashmap, |inner_map, series| {
                        
                        // Unpack element, series
                        let (element, series) = series.pair();

                        // Calculate number of exact chunks, and the total size of all exact chunks
                        let exact_chunks: usize = series.len() / coherence_time;
                        let exact_chunks_size: usize = exact_chunks * coherence_time;

                        // Chunk series
                        let two_dim_series = series
                            .slice(s!(0..exact_chunks_size))
                            .into_shape((*coherence_time, exact_chunks))
                            .expect("This shouldn't fail ever. If anything we should get an early panic from .slice()")
                            .map(|x| Complex { re: *x as f64, im: 0.0});

                        // Do ffts
                        let num_fft_elements = *coherence_time;
                        let mut fft_handler = FftHandler::<f64>::new(*coherence_time);
                        let mut fft_result = ndarray::Array2::<Complex<f64>>::zeros((num_fft_elements, exact_chunks));
                        ndfft(&two_dim_series, &mut fft_result, &mut fft_handler, 0);

                        // Get values at relevant frequencies
                        let approx_sidereal: usize = approximate_sidereal(frequency_bin);
                        if num_fft_elements < 2*approx_sidereal + 1 {
                            println!("no triplets exist");
                            return // Err("no triplets exist")
                        }

                        // let start_relevant: usize = *frequency_bin.multiples.start()-approx_sidereal;
                        let start_relevant: usize = frequency_bin.multiples.start().saturating_sub(approx_sidereal);
                        let end_relevant: usize = (*frequency_bin.multiples.end()+approx_sidereal).min(num_fft_elements-1);
                        let relevant_range = start_relevant..=end_relevant;
                        println!("relevant_range is {start_relevant}..={end_relevant}");
                        let relevant_values = fft_result.slice_axis(ndarray::Axis(0), ndarray::Slice::from(relevant_range));

                        // Get all relevant triplets
                        let relevant_triplets: Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)> = relevant_values
                            .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                            .into_iter()
                            .map(|window| 
                                (
                                    window.slice(s![0_usize, ..]).to_owned(),
                                    window.slice(s![approx_sidereal, ..]).to_owned(),
                                    window.slice(s![2*approx_sidereal, ..]).to_owned(),
                                )).collect();

                        // Insert relevant triplets into the inner dashmap
                        assert!(inner_map
                            .insert(element.clone(), relevant_triplets).is_none(), "Somehow a duplicate entry was made");
                    });

                assert!(data_vector_dashmap
                    .insert(*coherence_time, inner_dashmap).is_none(), "Somehow a duplicate entry was made");
            });

        data_vector_dashmap
    }

    fn calculate_mean_theory(
        &self,
        local_set: &Vec<(usize, FrequencyBin)>,
        len_data: usize,
    ) -> DashMap<usize, DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>>> {

        let mu: DashMap<
            usize /* coherence time */, 
            DashMap<
                Index /* chunk for this coherence time */,
                DashMap<
                    NonzeroElement,
                    f64
                >
            >
        > = DashMap::new();

        
        local_set
            .par_iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* &FrequencyBin */)| {

                // For the processed, cleaned dataset, this is 
                // the number of chunks for this coherence time
                let num_chunks = len_data / coherence_time;

                // Calculate the sidereal day frequency
                let fd = SIDEREAL_DAY_SECONDS.recip();

                // Calculate c_i. Unlike the original implementation, this is done using euler's 
                // exp(ix) = cos(x) + i sin(x)
                let c_i = Array1::range(0.0, len_data as f64, 1.0)
                    .map(|x| Complex { re: *x as f64, im: 0.0 })
                    .mul(
                        Complex {
                            re: 0.0, 
                            im: 2.0 * PI * approximate_sidereal(frequency_bin) as f64 / *coherence_time as f64 - fd,
                        }
                    )
                    .mapv(Complex::exp);

                // Calculate all the trig fds and pads
                let cosfd = Array1::range(0.0, len_data as f64, 1.0)
                    .mul(2.0 * PI * fd)
                    .mapv(f64::cos);
                let sinfd = Array1::range(0.0, len_data as f64, 1.0)
                    .mul(2.0 * PI * fd)
                    .mapv(f64::sin);
                let cospad = Array1::range(0.0, len_data as f64, 1.0)
                    .mul(2.0 * PI * approximate_sidereal(frequency_bin) as f64 / *coherence_time as f64)
                    .mapv(f64::cos);
                let sinpad = Array1::range(0.0, len_data as f64, 1.0)
                    .mul(2.0 * PI * approximate_sidereal(frequency_bin) as f64 / *coherence_time as f64)
                    .mapv(f64::sin);


            });
        DashMap::new()
    }
}

/// This function takes in the weights w_i along with the station coordinates and calculates H_i(t)
fn calculate_auxilary_values(
    weights_n: &DashMap<StationName, f32>,
    weights_e: &DashMap<StationName, f32>,
    weights_wn: &TimeSeries,
    weights_we: &TimeSeries,
    // chunk_dataset: &DashMap<StationName, Dataset>,
) -> DashMap<usize, TimeSeries> {

    // Initialize auxilary_value table
    let auxilary_value_series_plural = DashMap::new();

    // Gather coordinate table
    let coordinates = construct_coordinate_map();

    // Get size of series
    let size = weights_wn.len();

    for i in 1..=7 {

        // Here we iterate thrhough weights_n and not chunk_dataset because
        // stations in weight_n are a subset (filtered) of those in chunk_dataset.
        // Could perhaps save memory by dropping coressponding invalid datasets in chunk_dataset.
        let auxilary_value_series_unnormalized: TimeSeries = weights_n
            .iter()
            .map(|key_value| {

                // Unpack (key, value) pair
                // Here, key is StationName and value = dataset
                let station_name = key_value.key();

                // Get station coordinate
                let sc: &Coordinates = coordinates
                    .get(station_name)
                    .expect("station coordinates should exist");

                // Get product of relevant component of vector spherical harmonics and of the magnetic field.
                let auxilary_value = match i {
                    
                    // H1 summand = wn * cos(phi)^2
                    1 => (sc.longitude.cos().powi(2) as f32).mul(*weights_n.get(station_name).unwrap()),

                    // H2 summand = wn * sin(phi) * cos(phi)
                    2 => ((sc.longitude.sin() * sc.longitude.cos()) as f32).mul(*weights_n.get(station_name).unwrap()),

                    // H3 summand = we * cos(polar)^2
                    3 => (sc.polar.cos().powi(2) as f32).mul(*weights_e.get(station_name).unwrap()),

                    // H4 summand = we * cos(phi)^2 * cos(polar)^2
                    4 => ((sc.longitude.cos().powi(2) * sc.polar.cos().powi(2)) as f32).mul(*weights_n.get(station_name).unwrap()),

                    // H5 summand = we * sin(phi) * cos(phi) * cos(polar)^2
                    5 => ((sc.longitude.sin() * sc.longitude.cos() * sc.polar.cos().powi(2)) as f32)
                        .mul(*weights_n.get(station_name).unwrap()),

                    // H6 summand = we * cos(phi) * sin(polar) * cos(polar)
                    6 => ((sc.longitude.cos() * sc.polar.sin() * sc.polar.cos()) as f32).mul(*weights_n.get(station_name).unwrap()),

                    // H7 summand = we * sin(phi) * sin(polar) * cos(polar)
                    7 => ((sc.longitude.sin() * sc.polar.sin() * sc.polar.cos()) as f32).mul(*weights_n.get(station_name).unwrap()),
                    
                    _ => unreachable!("hardcoded to iterate from 1 to 7"),
                };

                auxilary_value
            })
            .fold(TimeSeries::default(size), |acc, series| acc.add(series));

            // Divide by correct Wi
            let auxilary_value_series_normalized = match i {
                1 => auxilary_value_series_unnormalized.div(weights_wn),
                2 => auxilary_value_series_unnormalized.div(weights_wn),
                3 => auxilary_value_series_unnormalized.div(weights_we),
                4 => auxilary_value_series_unnormalized.div(weights_we),
                5 => auxilary_value_series_unnormalized.div(weights_we),
                6 => auxilary_value_series_unnormalized.div(weights_we),
                7 => auxilary_value_series_unnormalized.div(weights_we),
                _ => unreachable!("hardcoded to iterate from 1 to 7")
            }; 

        // Insert combined time series for this nonzero element for this chunk, ensuring no duplicate entry
        assert!(auxilary_value_series_plural.insert(i, auxilary_value_series_normalized).is_none(), "Somehow made a duplicate entry");
    }

    auxilary_value_series_plural
}


/// This calculates the fft of a tophat function starting at t=0 and going to t=T, 
/// the length of the longest contiguous subset of the dataset Xi(t). 
/// 
/// To reiterate/clarify, this function assumes there are no gaps in data availability.
/// 
/// At k = 0, this results in 0/0, requiring the use of the L'Hospital rule. As such,
/// this case is dealt separately.
fn one_tilde(
    coherence_time: f64,
    chunk_index: usize,
    number_of_chunks: usize,
    k: f64,
) -> Complex<f64> {
    match k {
        
        // Deal with L'Hospital limit separately
        0.0 => Complex::new(coherence_time, 0.0),

        // Otherwise, evaluate function
        _ => (Complex::new(0.0, (1.0 - ((chunk_index + number_of_chunks) as f64 * coherence_time)/number_of_chunks as f64) * k).exp())*(-1.0 + Complex::new(0.0, coherence_time * k).exp())
                / (-1.0 + Complex::new(0.0, k).exp())
    }
}