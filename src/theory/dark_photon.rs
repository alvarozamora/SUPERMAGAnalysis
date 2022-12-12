use super::*;
use std::{collections::HashMap, sync::{MutexGuard, TryLockError}};
use dashmap::{DashMap, ReadOnlyView};
use interp1d::Interp1d;
use itertools::Itertools;
use rustfft::FftPlanner;
use serde_derive::{Serialize, Deserialize};
use tokio::sync::Semaphore;
use std::sync::Arc;
use crate::{utils::{
    loader::Dataset,
    approximate_sidereal, coordinates::Coordinates, fft::get_frequency_range_1s,
}, constants::{SIDEREAL_DAY_SECONDS}, weights::in_longest_subset};
use std::{
    ops::{Mul, Div, Add, Range, Sub},
};
use ndrustfft::{FftHandler, ndfft, ndfft_par};
use rayon::prelude::*;
use ndarray::{s, ScalarOperand, Array2, Axis};
use num_traits::{ToPrimitive, Float, Num};
use ndrustfft::Complex;
use std::sync::{Mutex, atomic::{Ordering, AtomicU32}};
use std::f32::consts::PI as SINGLE_PI;
use indicatif::{ProgressBar, MultiProgress, ProgressStyle};
use ndarray_linalg::{Cholesky, solve::Inverse, UPLO, SVD};


const ZERO: Complex<f32> = Complex::new(0.0, 0.0);
const ONE: Complex<f32> = Complex::new(1.0, 0.0);

/// Size of chunks (used for noise spectra ffts)
const TAU: usize = 16384 * 64;

type CoherenceTime = usize;
type Window = usize;
type Chunk = usize;
pub type InnerVarChunkWindowMap = DashMap<Window, Triplet>;
pub type InnerAChunkWindowMap = DashMap<Window, DashMap<Triplet, (Array2<Complex<f32>>,  Array2<Complex<f32>>)>>;


type DarkPhotonVecSphFn = Arc<dyn Fn(f32, f32) -> f32 + Send + 'static + Sync>;
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
        let vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>> = Arc::new(DashMap::new());
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[0].clone(),
            Arc::new(|_theta: f32, phi: f32| -> f32 {
                phi.sin()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[1].clone(),
            Arc::new(|_theta: f32, phi: f32| -> f32 {
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
            Arc::new(|theta: f32, _phi: f32| -> f32 {
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

    type AuxiliaryValue = DarkPhotonAuxiliary;
    type Mu = DarkPhotonMu;
    type Var = InnerVarChunkWindowMap;
    // type DataVector = DashMap<usize, DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>>>;
    // type DataVector = DashMap<usize /* Tc */, DashMap<usize /* chunk */, DashMap<usize /* window/triplet */, DarkPhotonVec<f32>>>>;
    type DataVector = ReadOnlyView<(CoherenceTime, Chunk, Window), DarkPhotonVec<f32>>;

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
                .into_iter()
                .fold(TimeSeries::default(size), |acc, (_key, series)| acc.add(series));

            // Insert combined time series for this nonzero element for this chunk, ensuring no duplicate entry
            assert!(projection_table.insert(nonzero_element.clone(), combined_time_series).is_none(), "Somehow made a duplicate entry");
        }

        projection_table
    }

    /// This calculates the auxiliary values for a chunk.
    fn calculate_auxiliary_values(
        &self,
        weights_n: &DashMap<StationName, f32>,
        weights_e: &DashMap<StationName, f32>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        _chunk_dataset: &DashMap<StationName, Dataset>,
    ) -> Self::AuxiliaryValue {
        DarkPhotonAuxiliary {
            h: dark_photon_auxiliary_values(weights_n, weights_e, weights_wn, weights_we),
        }
    }

    fn calculate_data_vector(
        &self,
        projections_complete: &ProjectionsComplete,
        local_set: &Vec<(usize, FrequencyBin)>,
    ) -> Self::DataVector {

        // This holds all (15 x 1) data vectors
        let data_vector_dashmap = DashMap::with_capacity_and_shard_amount(local_set.len().next_power_of_two() * 4, 128);

        // Local references
        let dp1 = DARK_PHOTON_NONZERO_ELEMENTS[0].clone();
        let dp2 = DARK_PHOTON_NONZERO_ELEMENTS[1].clone();
        let dp3 = DARK_PHOTON_NONZERO_ELEMENTS[2].clone();
        let dp4 = DARK_PHOTON_NONZERO_ELEMENTS[3].clone();
        let dp5 = DARK_PHOTON_NONZERO_ELEMENTS[4].clone();

        log::debug!("starting data vector");
        let multi = MultiProgress::new();
        let sty = ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap();
        let ch_pb = multi.add(ProgressBar::new(local_set.len() as u64));
        ch_pb.set_style(sty.clone());
        ch_pb.inc(0);


        let fft_semaphore = Semaphore::const_new(100); // essentially, no limit
        let total_num = local_set.len();
        let mut counter = 0;
        local_set
            .iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* FrequencyBin */)| {

                counter += 1;
                log::info!("coherence time {counter} of {total_num}");

                let pb = multi.add(ProgressBar::new(5));
                pb.set_style(sty.clone());
                pb.inc(0);

                // Parallelize over all nonzero elements in the theory
                projections_complete
                    .projections_complete
                    .par_iter()
                    .for_each_with(&data_vector_dashmap, |data_vector_map, series| {
                        
                        // Unpack element, series
                        let (element, series) = series.pair();

                        // Calculate number of exact chunks, and the total size of all exact chunks
                        // TODO: include last chunk via f64 --> ceiling
                        let exact_chunks: usize = series.len() / coherence_time;
                        let exact_chunks_size: usize = exact_chunks * coherence_time;

                        // Chunk series
                        let two_dim_series: Array2<Complex<f32>> = series
                            .slice(s!(0..exact_chunks_size))
                            .into_shape((*coherence_time, exact_chunks))
                            .expect("This shouldn't fail ever. If anything we should get an early panic from .slice()")
                            .map(|x| x.into());
                        drop(series); // to save memory

                        // Do ffts
                        let num_fft_elements = *coherence_time;
                        let mut fft_handler = FftHandler::<f32>::new(*coherence_time);
                        let permit = fft_semaphore.acquire();
                        let mut fft_result = ndarray::Array2::<Complex<f32>>::zeros((num_fft_elements, exact_chunks));
                        ndfft_par(&two_dim_series, &mut fft_result, &mut fft_handler, 0);
                        drop(two_dim_series); // to save memory
                        drop(fft_handler);

                        // Get values at relevant frequencies
                        let approx_sidereal: usize = approximate_sidereal(frequency_bin);
                        if num_fft_elements < 2*approx_sidereal + 1 {
                            println!("no triplets exist");
                            return // Err("no triplets exist")
                        }

                        // Get the start and end of the range of relevant frequencies from this bin
                        let start_relevant: usize = frequency_bin.multiples.start().saturating_sub(approx_sidereal);
                        let end_relevant: usize = (*frequency_bin.multiples.end()+approx_sidereal).min(num_fft_elements-1);
                        let relevant_range = start_relevant..=end_relevant;
                        // log::debug!("relevant_range is {start_relevant}..={end_relevant}");
                        // let relevant_values = fft_result.slice_axis(ndarray::Axis(0), ndarray::Slice::from(relevant_range));
                        // Overwrite to save memory!
                        fft_result = fft_result.slice_axis(ndarray::Axis(0), ndarray::Slice::from(relevant_range)).to_owned();
                        drop(permit);

                        // Get all relevant triplets -- KEEP THIS HERE FOR NOW
                        // let relevant_triplets: Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)> = relevant_values
                        //     .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                        //     .into_iter()
                        //     .map(|window| 
                        //         (
                        //             // NOTE: technically this might be an expensive clone, but it is likely okay.
                        //             window.slice(s![0_usize, ..]).to_owned(),
                        //             window.slice(s![approx_sidereal, ..]).to_owned(),
                        //             window.slice(s![2*approx_sidereal, ..]).to_owned(),
                        //         )).collect();

                        // relevant_values
                        fft_result
                            .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                            .into_iter()
                            .enumerate()
                            .for_each(|(window_index, window)|{
                                
                                // These have shape 1 x num_chunks
                                let low = window.slice(s![0_usize, ..]);
                                let mid = window.slice(s![approx_sidereal, ..]);
                                let high = window.slice(s![2*approx_sidereal, ..]);
                                
                                (0..low.len())
                                    .into_par_iter()
                                    .for_each_with(&data_vector_map, |map, chunk_index| {
                                    pb.set_message(format!("chunk {} of {}", chunk_index, low.len()));
                                    // data_vector_map
                                    map
                                        .entry((*coherence_time, chunk_index, window_index))
                                        .and_modify(|inner_data_vector: &mut DarkPhotonVec<f32>| {
                                            match element {
                                                e if e == &dp1 => {
                                                    inner_data_vector.low[0] += low[chunk_index];
                                                    inner_data_vector.mid[0] += mid[chunk_index];
                                                    inner_data_vector.high[0] += high[chunk_index];
                                                },
                                                e if e == &dp2 => {
                                                    inner_data_vector.low[1] += low[chunk_index];
                                                    inner_data_vector.mid[1] += mid[chunk_index];
                                                    inner_data_vector.high[1] += high[chunk_index];
                                                },
                                                e if e == &dp3 => {
                                                    inner_data_vector.low[2] += low[chunk_index];
                                                    inner_data_vector.mid[2] += mid[chunk_index];
                                                    inner_data_vector.high[2] += high[chunk_index];
                                                },
                                                e if e == &dp4 => {
                                                    inner_data_vector.low[3] += low[chunk_index];
                                                    inner_data_vector.mid[3] += mid[chunk_index];
                                                    inner_data_vector.high[3] += high[chunk_index];
                                                },
                                                e if e == &dp5 => {
                                                    inner_data_vector.low[4] += low[chunk_index];
                                                    inner_data_vector.mid[4] += mid[chunk_index];
                                                    inner_data_vector.high[4] += high[chunk_index];
                                                },
                                                _ => unreachable!("dark photon only has these 5 nonzero elements"),
                                            };
                                        })
                                        .or_insert_with(|| {
                                            // Initialize with zeros and metadata
                                            let mut inner_data_vector = DarkPhotonVec {
                                                low: [0.0.into(); 5].into_iter().collect(),
                                                mid: [0.0.into(); 5].into_iter().collect(),
                                                high: [0.0.into(); 5].into_iter().collect(),
                                                // coh_time: *coherence_time,
                                                // window: window_index,
                                                // chunk: chunk_index,
                                            };

                                            match element {
                                                e if e == &dp1 => {
                                                    inner_data_vector.low[0] += low[chunk_index];
                                                    inner_data_vector.mid[0] += mid[chunk_index];
                                                    inner_data_vector.high[0] += high[chunk_index];
                                                },
                                                e if e == &dp2 => {
                                                    inner_data_vector.low[1] += low[chunk_index];
                                                    inner_data_vector.mid[1] += mid[chunk_index];
                                                    inner_data_vector.high[1] += high[chunk_index];
                                                },
                                                e if e == &dp3 => {
                                                    inner_data_vector.low[2] += low[chunk_index];
                                                    inner_data_vector.mid[2] += mid[chunk_index];
                                                    inner_data_vector.high[2] += high[chunk_index];
                                                },
                                                e if e == &dp4 => {
                                                    inner_data_vector.low[3] += low[chunk_index];
                                                    inner_data_vector.mid[3] += mid[chunk_index];
                                                    inner_data_vector.high[3] += high[chunk_index];
                                                },
                                                e if e == &dp5 => {
                                                    inner_data_vector.low[4] += low[chunk_index];
                                                    inner_data_vector.mid[4] += mid[chunk_index];
                                                    inner_data_vector.high[4] += high[chunk_index];
                                                },
                                                _ => unreachable!("dark photon only has these 5 nonzero elements"),
                                            };
                                            inner_data_vector
                                        });
                                });
                            });
                        pb.inc(1);
                    });
                ch_pb.inc(1);
            });

        data_vector_dashmap.into_read_only()
    }

    /// NOTE: this implementation assumes that the time series is fully contiguous (with no null values).
    /// As such, it will produce incorrect results if used with a dataset that contains null values.
    /// 
    /// 
    fn calculate_mean_theory(
        &self,
        set: &Vec<(usize, FrequencyBin)>,
        len_data: usize,
        coherence_times: usize,
        auxiliary_values: Arc<Self::AuxiliaryValue>,
    ) -> DashMap<usize, DashMap<usize, DarkPhotonMu>> {

        // Calculate the sidereal day frequency
        const FD: f64 = 1.0 / SIDEREAL_DAY_SECONDS;

        // Map of Map of mus
        // First key is coherence time
        // Second key is chunk index
        let result = DashMap::with_capacity(coherence_times);

        log::trace!("starting loop for {} coherence times", set.len());
        set
            .into_par_iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* &FrequencyBin */)| {

                // For the processed, cleaned dataset, this is 
                // the number of chunks for this coherence time
                let num_chunks = len_data / coherence_time;
                
                // TODO: refactor elsewhere to be user input or part of some fit
                const RHO: f32 = 6.04e7;
                const R: f32 = 0.0212751;
                let mux_prefactor: f32 = SINGLE_PI * R * (2.0 * RHO).sqrt() / 4.0;

                // This is the inner chunk map from chunk to coherence time
                let inner_chunk_map = DashMap::new();

                for chunk in 0..num_chunks {

                    // Beginning and end index for this chunk in the total series
                    // NOTE: `end` is exclusive
                    let start: usize  = chunk * coherence_time;
                    let end: usize = ((chunk + 1) * coherence_time).min(len_data);
                    log::trace!("calculated start and end indices");

                    // Calculate cos + isin. Unlike the original implementation, this is done using euler's 
                    // exp(ix) = cos(x) + i sin(x)
                    //
                    // Note: when you encounter a chunk that has total time < coherence time, the s![start..end] below will truncate it.
                    // TODO: check phase
                    // TODO: check f32 precision
                    // TODO: shorthand Complex with const I
                    // let cis_fh_f = Array1::range(0.0, *coherence_time as f32, 1.0)
                    let cis_fh_f = (start..end)
                        .map(|x| Complex { re: x as f32, im: 0.0 })
                        .collect::<Array1<Complex<f32>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                im: 2.0 * SINGLE_PI * ((approximate_sidereal(frequency_bin).to_f64().expect("usize to double failed") * frequency_bin.lower).to_f32().expect("double to single failed") - FD as f32),
                            }
                        )
                        .mapv(Complex::exp);
                    // let cis_f = Array1::range(0.0, *coherence_time as f32, 1.0)
                    let cis_f = (start..end)
                        .map(|x| Complex { re: x as f32, im: 0.0 })
                        .collect::<Array1<Complex<f32>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                im: 2.0 * SINGLE_PI * FD as f32,
                            }
                        )
                        .mapv(Complex::exp);
                    // let cis_f_fh = Array1::range(0.0, *coherence_time as f32, 1.0)
                    let cis_f_fh = (start..end)
                        .map(|x| Complex { re: x as f32, im: 0.0 })
                        .collect::<Array1<Complex<f32>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                // This minus sign flips (fdhat-fd) --> (fd-fdhat)
                                im: -2.0 * SINGLE_PI * ((approximate_sidereal(frequency_bin).to_f64().expect("usize to double failed") * frequency_bin.lower).to_f32().expect("double to single failed") - FD as f32),
                            }
                        )
                        .mapv(Complex::exp);
                    log::trace!("calculated cis, which have length {}", cis_fh_f.len());
                    
                    // Get references to auxiliary values for this chunk for better readability
                    let h1 = auxiliary_values.h[0].slice(s![start..end]);
                    let h2 = auxiliary_values.h[1].slice(s![start..end]);
                    let h3 = auxiliary_values.h[2].slice(s![start..end]);
                    let h4 = auxiliary_values.h[3].slice(s![start..end]);
                    let h5 = auxiliary_values.h[4].slice(s![start..end]);
                    let h6 = auxiliary_values.h[5].slice(s![start..end]);
                    let h7 = auxiliary_values.h[6].slice(s![start..end]);
                    log::trace!("obtained auxiliary values for chunk {chunk}, which have length {}", h1.len());

                    // Start of f = fd-fdhat components

                    // mux0 is FT of (1 - H1 + iH2) at f=fd-fdhat
                    let mux0 = (&cis_f_fh)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(1.0 - h1_, h2_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux1 is FT of (H2 + iH1) at f=fd-fdhat
                    let mux1 = (&cis_f_fh)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h2_, h1_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux2 is FT of (H4 - iH5) at f=fd-fdhat
                    let mux2 = (&cis_f_fh)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4, &h5)| Complex::new(h4, -h5))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux3 is FT of (-H5 + i(H3-H4)) at f=fd-fdhat
                    let mux3 = (&cis_f_fh)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3, &h4), &h5)| Complex::new(-h5, h3-h4))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux4 is FT of (H6 - iH7) at f=fd-fdhat
                    let mux4 = (&cis_f_fh)
                        .mul(h6
                            .iter()
                            .zip(h7)
                            .map(|(&h6, &h7)| Complex::new(h6, -h7))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // start of f=fd components

                    // mux5 is Real(FT of 2*(1-H1)) = -2*Real(FT of H1-1)
                    //         + Im(FT of 2*H2)  = 2 * Im(FT fo H2)
                    // at f = fd
                    let mux5: Complex<f32> = {

                        // Real(FT of 2*(1-H1)) = -2*Real(FT of (H1-1))
                        let first_term = -2.0*(&cis_f)
                            .mul(&h1.sub(1.0))
                            .mul(mux_prefactor)
                            .sum()
                            .re;

                        // Im(FT of 2*H2)  = 2 * Im(FT fo H2)
                        let second_term = 2.0*(&cis_f)
                            .mul(&h2)
                            .mul(mux_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };


                    // mux6 is Real(FT of 2*H2) + Im(FT of 2*H1)
                    // at f = fd
                    let mux6: Complex<f32> = {

                        // Real(FT of 2*H2)
                        let first_term = 2.0*(&cis_f)
                            .mul(&h2)
                            .mul(mux_prefactor)
                            .sum()
                            .re;
                            
                        // Im(FT of 2*H1)
                        let second_term = 2.0*(&cis_f)
                            .mul(&h1)
                            .mul(mux_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };


                    // mux7 is Real(FT of 2*H4) - Im(FT of 2*H5)
                    // at f = fd
                    let mux7: Complex<f32> = {

                        // Real(FT of 2*H4)
                        let first_term = 2.0*(&cis_f)
                            .mul(&h4)
                            .mul(mux_prefactor)
                            .sum()
                            .re;
                            
                        // Im(FT of -2*H5)
                        let second_term = -2.0*(&cis_f)
                            .mul(&h5)
                            .mul(mux_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // mux8 is Real(FT of -2*H5) + Im(FT of 2*(H3-H4))
                    // at f = fd
                    let mux8: Complex<f32> = {

                        // Real(FT of -2*H5)
                        let first_term = -2.0*(&cis_f)
                            .mul(&h5)
                            .mul(mux_prefactor)
                            .sum()
                            .re;
                            
                        // Im(FT of 2*(H3-H4))
                        let second_term = 2.0*(&cis_f)
                            .mul(&h3.sub(&h4))
                            .mul(mux_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // mux9 is Real(FT of 2*H6) - Im(FT of 2*H7)
                    // at f = fd
                    let mux9: Complex<f32> = {

                        // Real(FT of 2*H6)
                        let first_term = 2.0*(&cis_f)
                            .mul(&h6)
                            .mul(mux_prefactor)
                            .sum()
                            .re;
                            
                        // Im(FT of -2*H7)
                        let second_term = -2.0*(&cis_f)
                            .mul(&h7)
                            .mul(mux_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // start of f = fdhat-fd components

                    // mux10 is FT of (1 - H1 - iH2) at f = fdhat-fd
                    let mux10: Complex<f32> = (&cis_fh_f)
                        .mul(h1
                                .iter()
                                .zip(h2)
                                .map(|(&h1_, &h2_)| Complex::new(1.0-h1_, -h2_))
                                .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux11 is FT of (H2 - iH1) at f = fdhat-fd
                    let mux11: Complex<f32> = (&cis_fh_f)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h2_, -h1_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux12 is FT of (H4 + iH5) at f = fdhat-fd
                    let mux12: Complex<f32> = (&cis_fh_f)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4_, &h5_)| Complex::new(h4_, h5_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux13 is FT of (-H5 + i*(H4 - H3)) at f = fdhat-fd
                    let mux13: Complex<f32> = (&cis_fh_f)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3_, &h4_), &h5_)| Complex::new(-h5_, h4_-h3_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux14 is FT of (H6 + iH7) at f = fdhat-fd
                    let mux14: Complex<f32> = (&cis_fh_f)
                        .mul(h6
                            .iter()
                            .zip(h7)
                            .map(|(&h6_, &h7_)| Complex::new(h6_, h7_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // start of muy
                    let muy_prefactor: f32 = -mux_prefactor;

                    // Start of f = fd-fdhat components

                    // muy0 is FT of (H2 + i*(H1-1)) at f=fd-fdhat
                    let muy0 = (&cis_f_fh)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h2_, -1.0 + h1_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy1 is FT of (H1 - iH2) at f=fd-fdhat
                    let muy1 = (&cis_f_fh)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h1_, -h2_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy2 is FT of (-H5 - iH4) at f=fd-fdhat
                    let muy2 = (&cis_f_fh)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4, &h5)| Complex::new(-h5, -h4))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy3 is FT of (H3 - H4 + iH5) at f=fd-fdhat
                    let muy3 = (&cis_f_fh)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3, &h4), &h5)| Complex::new(h3-h4, h5))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy4 is FT of (-H7 - iH6) at f=fd-fdhat
                    let muy4 = (&cis_f_fh)
                        .mul(h6
                            .iter()
                            .zip(h7)
                            .map(|(&h6, &h7)| Complex::new(-h7, -h6))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // start of f=fd components

                    // muy5 is 2*Re(FT(H2)) + 2*Im(FT(H1-1)) at f = fd
                    let muy5: Complex<f32> = {

                        //  2*Re(FT(H2))
                        let first_term = 2.0*(&cis_f)
                            .mul(&h2)
                            .mul(muy_prefactor)
                            .sum()
                            .re;

                        // 2*Im(FT(H1-1))
                        let second_term = 2.0*(&cis_f)
                            .mul(&h1.sub(1.0))
                            .mul(muy_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };


                    // muy6 is 2*Re(FT(H1)) - Im(FT(H2)) at f = fd
                    let muy6: Complex<f32> = {

                        // 2*Re(FT(H1))
                        let first_term = 2.0*(&cis_f)
                            .mul(&h1)
                            .mul(muy_prefactor)
                            .sum()
                            .re;
                            
                        // -2*Im(FT(H2))
                        let second_term = -2.0*(&cis_f)
                            .mul(&h2)
                            .mul(muy_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };


                    // muy7 is -2*Re(FT(H5)) - 2*Im(FT(H4)) at f = fd
                    let muy7: Complex<f32> = {

                        // -2*Re(FT(H5))
                        let first_term = -2.0*(&cis_f)
                            .mul(&h5)
                            .mul(muy_prefactor)
                            .sum()
                            .re;
                        
                        // -2*Im(FT(H4))
                        let second_term = -2.0*(&cis_f)
                            .mul(&h4)
                            .mul(muy_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // muy8 is 2*Re(FT(H3-H4)) + 2*Im(FT(H5)) at f = fd
                    let muy8: Complex<f32> = {

                        // 2*Re(FT(H3-H4))
                        let first_term = 2.0*(&cis_f)
                            .mul(&h3.sub(&h4))
                            .mul(muy_prefactor)
                            .sum()
                            .re;
                            
                        // 2*Im(TF(H5))
                        let second_term = 2.0*(&cis_f)
                            .mul(&h5)
                            .mul(muy_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // muy9 is -2*Re(FT(H7)) - 2*Im(FT(H6))
                    let muy9: Complex<f32> = {

                        // -2*Re(FT(H7))
                        let first_term = -2.0*(&cis_f)
                            .mul(&h7)
                            .mul(muy_prefactor)
                            .sum()
                            .re;
                            
                        // -2*Im(FT(H6))
                        let second_term = -2.0*(&cis_f)
                            .mul(&h6)
                            .mul(muy_prefactor)
                            .sum()
                            .im;
                        
                        (first_term + second_term).into()
                    };

                    // start of f = fdhat-fd components

                    // muy10 is FT(H2 + i*(1-H1)) at f = fdhat - fd
                    let muy10: Complex<f32> = (&cis_fh_f)
                        .mul(Complex::<f32>::new(1.0, 0.0)
                            .add(h1
                                .iter()
                                .zip(h2)
                                .map(|(&h1_, &h2_)| Complex::new(h2_, 1.0-h1_))
                                .collect::<Array1<_>>()))
                        .mul(muy_prefactor)
                        .sum();

                    // muy11 is FT(H1 + iH2) at f = fdhat - fd
                    let muy11: Complex<f32> = (&cis_fh_f)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h1_, h2_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy12 is FT(-H5 + iH4) at f = fdhat-fd
                    let muy12: Complex<f32> = (&cis_fh_f)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4_, &h5_)| Complex::new(-h5_, h4_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy13 is FT(H3-H4-iH5) at f = fdhat-fd
                    let muy13: Complex<f32> = (&cis_fh_f)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3_, &h4_), &h5_)| Complex::new(h3_ - h4_, -h5_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy14 is FT of (-H7 + iH6) at f = fdhat-fd
                    let muy14: Complex<f32> = (&cis_fh_f)
                        .mul(h6
                            .iter()
                            .zip(h7)
                            .map(|(&h6_, &h7_)| Complex::new(-h7_, h6_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // start of muz

                    // lets fill in zero components first
                    let [muz0, muz1, muz5, muz6, muz10, muz11] = [ZERO; 6];

                    // Now the nonzero components mu2, mu3, mu4, mu7, mu8, mu9, mu12, mu13, mu14
                    let muz_prefactor: f32 = 2.0 * mux_prefactor;
                    
                    // muz 2, 3, 4 are all at f = -fdhat
                    let fdhat = (approximate_sidereal(frequency_bin).to_f64().expect("usize to double failed") * frequency_bin.lower).to_f32().expect("double to single failed");
                    // TODO: check phase
                    // let cis_mfh = Array1::range(0.0, *coherence_time as f32, 1.0)
                    let cis_mfh = (start..end)
                        .map(|x| Complex { re: x as f32, im: 0.0 })
                        .collect::<Array1<Complex<f32>>>()
                        .mul(Complex::new(0.0, 2.0 * SINGLE_PI * -fdhat))
                        .mapv(Complex::exp);
                    // let cis_fh = Array1::range(0.0, *coherence_time as f32, 1.0)
                    let cis_fh = (start..end)
                        .map(|x| Complex { re: x as f32, im: 0.0 })
                        .collect::<Array1<Complex<f32>>>()
                        .mul(Complex::new(0.0, 2.0 * SINGLE_PI * fdhat))
                        .mapv(Complex::exp);

                    // muz2 is FT(H6) at f = -fdhat
                    let muz2: Complex<f32> = (&cis_mfh).mul(&h6).mul(muz_prefactor).sum();
                    // muz3 is FT(-H7) at f = -fdhat
                    let muz3: Complex<f32> = -(&cis_mfh).mul(&h7).mul(muz_prefactor).sum();
                    // muz4 is -FT(H3-1) at f = -fdhat (notice negative)
                    let muz4: Complex<f32> = -(&cis_mfh).mul(&h3.sub(1.0)).mul(muz_prefactor).sum();

                    // Note: These terms need the - along with prefactor for the f32 -> Complex<f32> into() to work
                    // muz7 is FT(H6) at f = 0
                    let muz7: Complex<f32> = h6.mul(muz_prefactor).sum().into();
                    // muz8 is FT(-H7) at f = 0
                    let muz8: Complex<f32> = h7.mul(-muz_prefactor).sum().into();
                    // muz9 is -FT(H3-1) at f = 0 (notice negative in front of FT)
                    let muz9: Complex<f32> = h3.sub(1.0).mul(-muz_prefactor).sum().into();

                    // muz12 is FT(H6) at f = fdhat
                    let muz12: Complex<f32> = (&cis_fh).mul(&h6).mul(muz_prefactor).sum();
                    // muz13 is FT(-H7) at f = fdhat
                    let muz13: Complex<f32> = -(&cis_fh).mul(&h7).mul(muz_prefactor).sum();
                    // muz14 is -FT(H3-1) at f = fdhat (notice negative in front of FT)
                    let muz14: Complex<f32> = -(&cis_fh).mul(&h3.sub(1.0)).mul(muz_prefactor).sum();

                    let chunk_mu = DarkPhotonMu {
                        x: [mux0, mux1, mux2, mux3, mux4, mux5, mux6, mux7, mux8, mux9, mux10, mux11, mux12, mux13, mux14],
                        y: [muy0, muy1, muy2, muy3, muy4, muy5, muy6, muy7, muy8, muy9, muy10, muy11, muy12, muy13, muy14],
                        z: [muz0, muz1, muz2, muz3, muz4, muz5, muz6, muz7, muz8, muz9, muz10, muz11, muz12, muz13, muz14],
                    };
                    log::trace!("calculated mus for chunk {chunk}");

                    // Insert chunk mu into chunk map
                    inner_chunk_map.insert(
                        chunk,
                        chunk_mu,
                    );
                }
                
                // Insert chunk map into coherence time map
                result.insert(
                    *coherence_time,
                    inner_chunk_map,
                );
            });
        result
    }


    fn calculate_var_theory(
        &self,
        set: &Vec<(usize, FrequencyBin)>,
        projections_complete: &ProjectionsComplete,
        coherence_times: usize,
        days: Range<usize>,
        stationarity: Stationarity,
        auxiliary_values: Arc<Self::AuxiliaryValue>,
    ) -> Result<DiskDB> {
        // Map of Map of Vars
        // key is stationary time chunk (e.g. year)
        let spectra = DashMap::with_capacity(coherence_times);

        let downtime = 0;

        // Initialize progress bar
        let initial_spectra_pb = ProgressBar::new(2021-2003);
        log::info!("Calculating power spectra for stationarity times");

        // NOTE: This is hardcoded for stationarity = 1 year
        (2003..2021).into_par_iter().for_each(|year| {
        // (2007..=2007).into_par_iter().for_each(|year| {

            // Get stationarity period indices (place within entire SUPERMAG dataset)
            // NOTE: this definition varies from original implementation. The original
            // python implementation defines the `end` index to be the first index of the
            // next chunk, since start:end is not end inclusive. This means the size of 
            // the chunks are (end - start + 1)
            let (start_stationarity, end_stationarity) = stationarity.get_year_indices(year);
            
            // Now convert these indices to the indices within the subset used
            let secs: Range<usize> = projections_complete.secs();
            let (start_in_series, end_in_series) = (
                // TODO: make sure these fallback indices are correct
                secs.clone().position(|i| i == start_stationarity).unwrap_or(secs.start),
                    // .expect("sec index is out of bounds"), 
                secs.clone().position(|i| i == end_stationarity).unwrap_or(secs.end-1),
                    // .expect("sec index is out of bounds"),
            );
            assert_ne!(start_in_series, end_in_series, "zero size");

            // Get the subseries for this year
            let projections_subset: DashMap<NonzeroElement, Vec<f32>> = projections_complete
                .iter()
                .map(|kv| {
                    // Get element and the complete longest contiguous series
                    let (element, complete_series) = kv.pair();
                    // println!("getting yearly subset {}..={} of complete series with length {}", start_in_series, end_in_series, complete_series.len());
                    let pair = (element.clone(), complete_series.slice(s![start_in_series..=end_in_series]).to_vec());
                    // println!("got yearly subset of complete series");
                    pair
                }).collect();
        
            // Get chunk indices
            let num_chunks: usize = ((end_in_series - start_in_series + 1) / TAU).max(1);
            let chunk_size: usize = (end_in_series - start_in_series + 1) / num_chunks;
            let chunk_mod: usize = (end_in_series - start_in_series + 1) % num_chunks;
            let stationarity_chunks: Vec<[usize; 2]> = (0..num_chunks)
                .map(|k| {
                    [k * (chunk_size + downtime) + k.min(chunk_mod), k * (chunk_size + downtime) + (k + 1).min(chunk_mod) + chunk_size]
                }).collect_vec();
            // dbg!(&stationarity_chunks);

            // Set up an fft planner to reuse for all ffts in this year
            let mut planner = FftPlanner::new();
            let fft_handler = planner.plan_fft_forward(2*TAU);
            let mut scratch = vec![0.0.into(); 2*TAU];
        
            let mut chunk_collection = Vec::with_capacity(stationarity_chunks.len());
            for stationarity_chunk in &stationarity_chunks {

                // Get the data for this chunk, pad it to have length equal to a power of 2, and take its fft
                let chunk_ffts: Vec<(NonzeroElement, Power<f32>)> = projections_subset
                    .iter()
                    .map(|kv| {

                        // Get element and series
                        let (element, year_series) = kv.pair();

                        // Get chunk from year
                        log::trace!("getting chunk from year");
                        let chunk_from_year = &year_series[stationarity_chunk[0]..stationarity_chunk[1]];
                        log::trace!("got chunk from year");

                        // Padded series
                        let chunk_size = stationarity_chunk[1] - stationarity_chunk[0];
                        let mut padded_series = Array1::<Complex<f32>>::zeros(2*TAU);
                        log::trace!(
                            "size of chunk is {}, being placed in a {} series of zeros",
                            chunk_size, padded_series.len(),
                        );

                        log::trace!("modifying series");
                        padded_series
                            .slice_mut(s![0..chunk_size])
                            .iter_mut()
                            .zip(chunk_from_year)
                            .for_each(|(ps, s)| { *ps = s.into(); });
                        log::trace!("modified series");


                        // FFT of padded series
                        fft_handler
                            .process_with_scratch(padded_series.as_slice_mut().expect("should be in contiguous order"), &mut scratch);

                        // Package power with metadata
                        let power = Power {
                            power: padded_series,
                            start_sec: start_in_series,
                            end_sec: end_in_series,
                        };

                        (element.clone(), power)
                    }).collect();

                // Initialize dashmap for correlation
                let chunk_ffts_squared: DashMap<(NonzeroElement, NonzeroElement), Power<f32>> = DashMap::new();
                let nancount = 0;
                for (e1, fft1) in chunk_ffts.iter() {
                    for (e2, fft2) in chunk_ffts.iter() {
                        chunk_ffts_squared.insert(
                            (e1.clone(), e2.clone()),
                             // NOTE: in this scope, fft1 and fft2 should have the same start/end
                             // so its okay to use fft2.power and discard start/end and inherit
                             // start/end from fft1
                             (fft1.mul(&fft2.power.map(Complex::conj)))
                                .mul_scalar(2.0 / (/* original had 60* */ (stationarity_chunk[1] - stationarity_chunk[0] - nancount) as f32))
                        );
                    }
                }

                chunk_collection.push(chunk_ffts_squared);
            }

            // Take the average. 
            // First, initialize zero arrays 
            let avg_power: DashMap<(NonzeroElement, NonzeroElement), Power<f32>> = DARK_PHOTON_NONZERO_ELEMENTS
                .iter()
                .cartesian_product(DARK_PHOTON_NONZERO_ELEMENTS.iter())
                .map(|(e1, e2)| ((e1.clone(), e2.clone()), Power { power: Array1::zeros(2*TAU), start_sec: start_in_series, end_sec: end_in_series }))
                .collect();
            // Then, sum (get denominator before consuming iterator)
            let denominator = chunk_collection.len() as f32;
            chunk_collection
                .into_iter()
                .for_each(|chunk| {
                    chunk
                        .into_iter()
                        .for_each(|(element_pair, power)| {
                            avg_power
                                .get_mut(&element_pair)
                                .expect("nonzero element_pair should exist")
                                .add_assign(&power.power);
                        })
                });

            // Finally, divide
            avg_power
                .iter_mut()
                .for_each(|mut kv| kv.value_mut().div_assign_scalar(denominator * ONE));

            
            // Add along with the rest of the stationarity times
            spectra.insert(year, avg_power);
            // log::info!("Finished calculating power for stationarity_time {year}");
            initial_spectra_pb.inc(1)
        });


        // Initialize progress bar
        let interpolators_pb = ProgressBar::new(spectra.len() as u64);
        log::info!("Generating interpolators");
        
        let power_interpolators: DashMap<_, DashMap<_,_>> = spectra
            .par_iter()
            .map(|kv| {
                let (stationarity_time, power_map) = kv.pair();
                (
                    *stationarity_time,
                    { 
                        let result = power_map
                            .par_iter()
                            .map(|kv_inner| {
                                let (element_pair, power) = kv_inner.pair();
                                let power_frequencies: Vec<f32> = power.frequencies();
                                let power_vec = power.power.to_vec();
                                (element_pair.clone(), Interp1d::new_unsorted(power_frequencies, power_vec).expect("failed to construct power interpolator"))
                            }).collect();
                        interpolators_pb.inc(1);
                        result
                    }
                )
            }).collect();
        
        // Initialize progress bar
        let stitch_pb = ProgressBar::new(set.len() as u64);
        log::debug!("Stitching spectra; there are {coherence_times} coherence times");

        // Stitch spectra together according to coherence times
        let node_id: usize = std::env::var("SLURM_NODEID").unwrap_or("0".to_string()).parse().unwrap();
        let disk_db = DiskDB::connect(format!("./sigma_{node_id}/"))
            .expect("failed to open sigma db");
        let triplet_counter = AtomicU32::new(0);
        set
            .into_iter()
            .for_each(|(coherence_time, frequency_bin)| {

                // First get the domain for the data used
                let secs = projections_complete.secs();
                let len_data = secs.len();

                // Then, get each of the coherence chunks
                let num_chunks = len_data / coherence_time;
                
                
                // This is the inner chunk map from chunk to coherence time
                // let inner_chunk_map: DashMap<(NonzeroElement, NonzeroElement), DashMap<usize, Triplet<f32>>> = DashMap::new();
                // NOTE: work in progress: restructuring to get the 5x5 2d arrays contiuous in memory
                let inner_chunk_map: InnerVarChunkWindowMap = DashMap::new();

                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk| {

                    // Beginning and end index for this chunk in the total series
                    // NOTE: `end` is exclusive
                    let chunk_start: usize  = chunk * coherence_time;
                    let chunk_end: usize = ((chunk + 1) * coherence_time)
                        .min(len_data);
                    log::trace!("calculated start and end indices");

                    // Check all spectra (over all stationarity periods) for any overlap
                    spectra
                        .par_iter()
                        .for_each(|kv| {

                            // Unpack key value pair
                            let (stationarity_time, power_map) = kv.pair();

                            // Iterate through every element in the 5x5 map
                            power_map
                                .into_par_iter()
                                .for_each(|kv_inner| {

                                    // Get element pair and corresponding power
                                    let (element_pair, power) = kv_inner.pair();
                                    let (e1, e2) = element_pair;

                                    // Get start and end seconds for this power
                                    let (power_start, power_end) = (power.start_sec, power.end_sec);

                                    // Calculate overlap (in number of seconds)
                                    // TODO: verify that these two ends are both inclusive ends,
                                    // as that is what the function assumes.
                                    let overlap: usize = calculate_overlap(
                                        chunk_start,
                                        chunk_end,
                                        power_start,
                                        power_end
                                    );

                                    // Add contribution to chunk if there is overlap
                                    if overlap > 0 {

                                        // Get values at relevant frequencies
                                        let approx_sidereal: usize = approximate_sidereal(frequency_bin);
                                        let num_fft_elements = *coherence_time;
                                        if num_fft_elements < 2*approx_sidereal + 1 {
                                            log::warn!("no triplets exist for {coherence_time}");
                                            return // Err("no triplets exist")
                                        }

                                        // Get the start and end of the range of relevant frequencies from this bin
                                        let start_relevant: usize = frequency_bin.multiples.start().saturating_sub(approx_sidereal);
                                        let end_relevant: usize = (*frequency_bin.multiples.end()+approx_sidereal).min(num_fft_elements-1);
                                        let relevant_range = start_relevant..=end_relevant;
                                        log::trace!("relevant_range is {start_relevant}..={end_relevant}");

                                        // Interpolate power to appropriate frequencies
                                        let frequencies_to_interpolate_to: Vec<f32> = relevant_range
                                            .map(|i| i as f32 * frequency_bin.lower as f32)
                                            .collect();
                                        let interpolated_power: Array1<Complex<f32>> = frequencies_to_interpolate_to
                                            .iter()
                                            .map(|f| {
                                                power_interpolators
                                                    .get(&stationarity_time)
                                                    .expect("interpolator should exist for this stationarity time")
                                                    .get(element_pair)
                                                    .expect("interpolator should exist for this element pair")
                                                    .interpolate_checked(*f)
                                                    .unwrap(/* unwrapping for now if out of bounds */)
                                            }).collect();

                                        // Get all relevant triplets and multiply them by their overlap weight
                                        interpolated_power
                                            .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                                            .into_iter()
                                            .enumerate()
                                            .for_each(|(window_index, window)| {

                                                // Get triplet
                                                let low: Complex<f32> = window[0_usize].mul(overlap as f32);
                                                let mid: Complex<f32> = window[approx_sidereal].mul(overlap as f32);
                                                let high: Complex<f32> =  window[2*approx_sidereal].mul(overlap as f32);

                                                // get frequencies
                                                let lowf: f32 = frequencies_to_interpolate_to[window_index];
                                                let midf: f32 = frequencies_to_interpolate_to[window_index + approx_sidereal];
                                                let hif: f32 = frequencies_to_interpolate_to[window_index + 2*approx_sidereal];
                                                
                                                // The element indices start at 1 so subtract 1
                                                // i.e. (X1, X2, X3, X4, X5) -> (0, 1, 2, 3, 4)
                                                let (ix, iy) = (e1.index-1, e2.index-1);

                                                // Add/store triplet
                                                inner_chunk_map
                                                    .entry(window_index)
                                                    .and_modify(|triplet| {

                                                        // First, add to low triplet matrix (fa-fdhat)
                                                        triplet.low[[ix, iy]] += low;

                                                        // Then, add to mid triplet matrix (fa)
                                                        triplet.mid[[ix, iy]] += mid;

                                                        // Finally, add to hi triplet matrix (fa+fdhat)
                                                        triplet.high[[ix, iy]] += high;
                                                    }).or_insert_with(|| {

                                                        // TODO: remove after debugging
                                                        triplet_counter.fetch_add(1, Ordering::Relaxed);

                                                        // Calculate the triplet of arrays with the first entry
                                                        let low_array =  {
                                                            let mut zero_array = Array2::zeros((5,5));
                                                                zero_array[[ix, iy]] = low;
                                                                zero_array
                                                        };
                                                        let mid_array =  {
                                                            let mut zero_array = Array2::zeros((5,5));
                                                                zero_array[[ix, iy]] = mid;
                                                                zero_array
                                                        };
                                                        let high_array =  {
                                                            let mut zero_array = Array2::zeros((5,5));
                                                                zero_array[[ix, iy]] = high;
                                                                zero_array
                                                        };

                                                        // Package and initialize triplet
                                                        Triplet {
                                                            low: low_array,
                                                            mid: mid_array,
                                                            high: high_array,
                                                            // lowf: lowf,
                                                            midf: midf,
                                                            // hif: hif,
                                                            // coh_time: *coherence_time,
                                                            // chunk: chunk,
                                                            // window: Some(window_index),
                                                        }
                                                    });
                                            });
                                    }
                            });
                        });
                });
                // Insert inner chunk containing this coherence time's power spectrum
                // result
                //     .insert(*coherence_time, inner_chunk_map);
                // now we are just inserting into db
                disk_db
                    .insert_windows(*coherence_time, &inner_chunk_map)
                    .expect("sigma insertion failed");

                // log::info!("Finished calculating power for coherence_time {coherence_time}");
                stitch_pb.inc(1)
            });

        let triplet_count = triplet_counter.load(std::sync::atomic::Ordering::Acquire);
        log::debug!("initialized {} triplets -> {} total arrays", triplet_count, 3 * triplet_count);

        Ok(disk_db)
    }

    /// The process is
    /// 1) Carry out Cholesky decomoposition on Sigma_k = A_k * Adag_k, obtaining A_k
    /// 2) Invert A_k, obtaining Ainv_k
    /// 3) Calculate Y_k = Ainv_k * X_k
    /// 4) Calculate nu_ik = Ainv_k * mu_ik
    /// 5) SVD into Nk = nu_ik -> U_k * S_k * Vdag_k, obtaining U_k
    /// 6) Calculate Zk = Udag_k * Y_k
    /// 7) Calculate likelihood -ln Lk = |Z_k - eps * S_k * d_k|^2
    fn calculate_likelihood(
        &self,
        set: &Vec<(usize, FrequencyBin)>,
        projections_complete: &ProjectionsComplete,
        data_vector: &Self::DataVector,
        theory_mean: &DashMap<usize, DashMap<usize, DarkPhotonMu>>,
        coherence_times: usize,
        days: Range<usize>,
        stationarity: Stationarity,
    ) -> Vec<(f32, f32)> {

        // Load sigma
        let node_id: usize = std::env::var("SLURM_NODEID").unwrap_or("0".to_string()).parse().unwrap();
        let sigma_db = DiskDB::connect(format!("./sigma_{node_id}/"))
            .expect("database should exist by this stage");

        // Initialize database where A_k and Ainv_k is stored
        // let a_db = DiskDB::connect("./a/").unwrap();
        // let ak_db = DiskDB::connect("./ak/").unwrap();

        // Get number of seconds in the continguous subset of dataset
        let num_secs = projections_complete.num_secs();
        
        let sz2_coherence: DashMap<_, _> = set
            .into_iter()
            .map(|(coherence_time, frequency_bin)| {

                // Get number of chunks for this coherence time
                let num_chunks = num_secs / coherence_time;

                // Get theory mean for this coherence time
                let theory_mean_ct = theory_mean
                    .get(&coherence_time)
                    .expect("theory mean should exist for every coherence time here");

                // // Get data vector component for coherence time
                // let data_vector_ct = data_vector
                //     .get(&coherence_time)
                //     .expect("data vector should exist for every coherence time here");

                // For this coherence time, get the map to every triplet window (for sigma)
                let window_map = sigma_db
                    .get_windows(*coherence_time)
                    .expect("failed to get window map")
                    .expect("window map not present in db");

                // // Step 1 and 2: calculate inverse of A_k
                let ainv = window_map
                    .par_iter()
                    .map(|kv2| {
                        let (window, triplet) = kv2.pair();
                        (*window, triplet.block_cholesky().block_inv())
                        }).collect::<DashMap<_,_>>();

                let sz2_map = DashMap::new();
                for chunk in 0..num_chunks {

                    // // Get data vector for this chunk,
                    // let data_vector_chunk = data_vector_ct
                    // .get(&chunk)
                    // .expect("data vector should exist for every chunk");

                    // Step 3: Calculate Y_k = Ainv_k * X_k for all windows in this chunk
                    let y: DashMap<Window, DarkPhotonVec<f32>> = ainv
                        .par_iter()
                        .map(|kv2| {
                            // Get triplet window and corresponding map
                            let (window, ainv_triplet) = kv2.pair();
                            
                            // Get data vector for this coh time, chunk, window
                            let data_vector_window = data_vector
                                .get(&(*coherence_time, chunk, *window))
                                .expect("data vector should exist for every window");
                            
                            // (window, Ainv * Xk_)
                            (*window, ainv_triplet.dot_vec_f32(&*data_vector_window))
                            }).collect::<DashMap<_,DarkPhotonVec<f32>>>();

                    // Step 4: Calculate nu_ik = Ainv_k * mu_ik
                    let nu = ainv
                        .par_iter()
                        .map(|kv2| {
                            // Get triplet window and corresponding map
                            let (window, ainv_triplet) = kv2.pair();

                            // Get theory mean for this chunk,
                            // scaled by the window's frequency
                            let scale_frequency = ainv_triplet.midf;
                            let theory_mean_chunk_scaled: DarkPhotonMuBlock = theory_mean_ct
                                .get(&chunk)
                                .expect("this coherence time should exist")
                                .scale(scale_frequency)
                                .to_blocks();
                            
                            // ainv is 15x15 and consists of 5x5 blocks.
                            // Mean should be 15x3, split into blocks here.
                            // matmul should result in a 15x3 matrix, but they are blocked into 5x3 chunks.
                            let three_blocks = ainv_triplet.mat_mul([theory_mean_chunk_scaled.block1, theory_mean_chunk_scaled.block2, theory_mean_chunk_scaled.block3]);
                            assert_eq!(three_blocks.low.shape(), &[5, 3]);
                            assert_eq!(three_blocks.mid.shape(), &[5, 3]);
                            assert_eq!(three_blocks.high.shape(), &[5, 3]);

                            // We take these 5x3 blocks and assemble a 15x3 matrix
                            let nu: Array2<Complex<f32>> = Array2::from_shape_fn((15, 3), |(i, j)| { 
                                match i {
                                    _i if _i < 5 => three_blocks.low[[i, j]],
                                    _i if _i < 10 => three_blocks.mid[[i-5, j]],
                                    _i if _i < 15 => three_blocks.high[[i-10, j]],
                                    _ => unreachable!("should not reach this"),
                                }
                            });
                            assert_eq!(nu.shape(), &[15, 3]);

                            // Return nu with window as the key
                            (*window, nu)
                        }).collect::<DashMap<_, Array2<Complex<f32>>>>();

                    // Step 5: Calculate svd of Nik
                    // Step 6: Calculate Zk = Udag_k * Y_k
                    // This gets us s and z2
                    let sz2 = nu
                        .into_par_iter()
                        .map(|(window, nu_window)| {
                            // Step 5: Carry out svd
                            // nu is a 15x3 matrix
                            // u should be 15x3, S should be 3x3, and v should be 3x3.
                            let (Some(u), s, Some(v)) = nu_window.svd(true, true).expect("svd failed") else {
                                panic!("u and v are being requested but were not given ")
                            };
                            assert_eq!(u.shape(), &[15, 3], "svd: u does not have correct shape");
                            assert_eq!(s.shape(), &[3], "svd: s does not have correct shape");
                            assert_eq!(v.shape(), &[3, 3], "svd: v does not have correct shape");

                            // Conjugate and transpose v and u
                            let vdag = v.t().map(|c| c.conj());
                            let udag = u.t().map(|c| c.conj());
                            assert_eq!(vdag.shape(), &[3, 3], "congj t: vdag does not have correct shape");
                            assert_eq!(udag.shape(), &[3, 15], "congj t: udag does not have correct shape");
                            
                            // Step 6: Zk = Udag_k * Y_k
                            let y_k = y.get(&window).expect("yk should exist for this window").to_vec();
                            assert_eq!(y_k.shape(), &[15]);
                            let z = udag.dot(&y_k);
                            assert_eq!(z.shape(), &[3]);

                            (window, (s, z))
                        }).collect::<DashMap<_, _>>();

                    // Insert into sz map
                    sz2_map.insert(chunk, sz2.into_read_only());
                }
                // Return sz map
                (coherence_time, (frequency_bin, sz2_map.into_read_only()))
            }).collect();

        let sz2_coherence = sz2_coherence.into_read_only();

        // Use SZ to calculate bounds
        let freqs_and_bounds: Vec<(f32, f32)> = sz2_coherence
            .into_par_iter()
            .map(|(coherence_time, (frequency_bin, sz_chunk_map))| {

                // Total number of chunks for this coherence time
                let num_chunks = num_secs / coherence_time;

                // let bounds = vec![];
                let approx_sidereal: usize = approximate_sidereal(frequency_bin);
                let num_fft_elements = *coherence_time;
                if num_fft_elements < 2*approx_sidereal + 1 {
                    println!("no triplets exist");
                }
                let start_relevant: usize = frequency_bin.multiples.start().saturating_sub(approx_sidereal);
                let end_relevant: usize = (*frequency_bin.multiples.end()+approx_sidereal).min(num_fft_elements-1);
                let relevant_range = start_relevant..=end_relevant;
                // log::debug!("relevant_range is {start_relevant}..={end_relevant}");
                let window_indices: Vec<usize> = relevant_range
                    .into_iter()
                    .enumerate()
                    .map(|(window_index, _)| window_index)
                    .collect();
                
                // Now that we have all the window indices,
                // for each of them collect all chunks and calculate bound
                let mut bounds = Vec::with_capacity(window_indices.len());
                for window in window_indices {
                    // Initialize vec to hold references to sz pairs for this frequency/window
                    let mut sz_references = Vec::<&(Array1<f32>, Array1<Complex<f32>>)>::with_capacity(num_chunks);
                    for chunk in 0..num_chunks {
                        sz_references.push(sz_chunk_map
                            .get(&chunk)
                            .expect("chunk should exist")
                            .get(&window)
                            .expect("window should exist")
                        );
                    }

                    let frequency = window as f64 * frequency_bin.lower;
                    bounds.push((frequency as f32, bound(&sz_references)));
                }

                // TODO
                bounds
            }).flatten().collect();

        freqs_and_bounds
    }
}


/// Calculates the bound for a particular (coherence_time, chunk, window)
fn bound(
    // This collects all z and s that have the same frequency
    sz: &[&(Array1<f32>, Array1<Complex<f32>>)],
) -> f32 {

    // The pdf is of the form N * sqrt(sum(...)) * prod(a exp(b))
    // so we will break down logpdf into summands 
    // 1) log N
    // 2) log sqrt(sum(...))
    // 3) log sum(a)
    // 4) log sum(b)
    let logpdf = |norm: f32, eps: f32| {
        // Term 1: logarithm of normalization factor
        let term_1: f32 = norm.ln();

        // Term 2: logarithm of square root of sum, from jeffery's prior
        let term_2: f32 = sz.iter().map(|(si, _zi)| {
            si.iter().map(|sik| {
                (4.0 * eps.powi(2) * sik.powi(4)) / (3.0 + eps.powi(2) * sik.powi(2)).powi(2)
            }).sum::<f32>()
        }).sum::<f32>().sqrt().ln();

        // Term 3: log sum(a)
        let term_3: f32 = sz.iter().map(|(si, _zi)| {
            si.iter().map(|sik| {
                ((3.0 + eps.powi(2) * sik.powi(2)).powi(2)).ln()
            }).sum::<f32>()
        }).sum();


        // Term 4: log(a)
        let term_4: f32 = {
            let mut acc = 0.0_f32;
            for j in 0..sz.len() {
                let (sj, zj) = sz[j];
                for i in 0..3 {
                    acc += 3.0 * zj[i].norm_sqr() / (3.0 + eps.powi(2) * sj[i].powi(2));
                }
            }
            acc
        };

        // Add all terms
        term_1 + term_2 + term_3 + term_4
    };

    // Find where logeps is maximum in -20, 20
    // TODO: refactor into DarkPhoton: Theory
    // TODO: change for this dataset
    let min_log_eps = -20.0;
    let max_log_eps = 20.0;
    let num_eps = 1000;
    let log_eps_grid: Vec<f32> = (0..num_eps).map(|i| min_log_eps + i as f32 * (max_log_eps-min_log_eps) / num_eps.sub(1) as f32).collect();
    let (max_logp, max_logp_eps) = log_eps_grid
        .iter()
        .map(|&log_eps| (log_eps, logpdf(1.0, 10_f32.powf(log_eps)))) // (logeps, logpdf)
        .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap()) // find max logpdf
        .unwrap();
    log::debug!("found max_logp {max_logp:.2e} at logeps {max_logp_eps:.2e}");

    // Transform pdf(x) to pdf(y) via 
    // pdf(y) = pdf(x(y)) |dx(y)/dy| 
    //        = pdf(exp(logeps)) * exp(logeps) 
    // i.e. for 
    // x = eps
    // y = logeps
    // x(y) = exp(y)  ---> eps(logeps) = exp(logeps)
    // dx(y)/dy = d/dy exp(y) = exp(y) ---> exp(logeps)
    //
    // The integration library used below expects double precision...
    let transformed_unnormalized_pdf = |logeps: f64| {
        logpdf(1.0, logeps.exp() as f32) as f64 * logeps.exp()
    };

    // Now, integrate function from -20 to 20, letting integrator figure it out
    let normalization = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, -20.0, 20.0, 1e-6).integral;
    let mut dlogeps = 1.0;
    let mut upper_bound = -10.0;
    let mut converged = false;
    let mut last_status = 0u8;
    const ABOVE: u8 = 1;
    const BELOW: u8 = 2;
    while !converged {
        let current_integral = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, -20.0, upper_bound, 1e-6).integral;
        let i = current_integral / normalization;
        if i.abs() < 1e-6 {
            // If within tolerance we are converged
            converged = true;
        } else if i > 0.95 {
            // Else, if above target decrease upper_bound
            // First check if we crossed target
            if last_status == BELOW {
                // we have skipped over target, so reduce delta
                dlogeps /= 10.0;

            }
            upper_bound -= dlogeps;
            last_status = ABOVE;
        } else if i < 0.95 {
            // Else, if below target increase upper_bound
            // First check if we crossed target
            if last_status == BELOW {
                // we have skipped over target, so reduce delta
                dlogeps /= 10.0;

            }
            upper_bound += dlogeps;
            last_status = BELOW;
        }
    }

    // Upper bound is for logeps so exponentiate to get exp bound
    upper_bound.exp() as f32
}


#[derive(Default, Serialize, Deserialize)]
pub struct DarkPhotonMu {
    // #[serde(with = "ComplexDef")]
    pub x: [Complex<f32>; 15],
    // #[serde(with = "ComplexDef")]
    pub y: [Complex<f32>; 15],
    // #[serde(with = "ComplexDef")]
    pub z: [Complex<f32>; 15],
}

#[derive(Default, Serialize, Deserialize)]
/// Note: these should all have length = 5
pub struct DarkPhotonVec<T: Num + Clone> {
    pub low: Array1<Complex<T>>,
    pub mid:  Array1<Complex<T>>,
    pub high:  Array1<Complex<T>>,
    // utility variables for assertions
    // coh_time: usize,
    // chunk: usize,
    // window: usize,
}

impl<T: Num + Clone> DarkPhotonVec<T> {
    fn to_vec(&self) -> Array1<Complex<T>> {
        ndarray::concatenate![Axis(0), self.low, self.mid, self.high]
    }
}

#[derive(Default, Serialize, Deserialize, Debug, PartialEq)]
struct DarkPhotonMuBlock {
    block1: Array2<Complex<f32>>,
    block2: Array2<Complex<f32>>,
    block3: Array2<Complex<f32>>,
}

// const DARK_PHOTON_MU_SHAPE: (usize, usize) = (15, 3);

impl DarkPhotonMu {
    // /// produces a 15 x 3 matrix out of the x, y, z components
    // fn to_matrix(&self) -> Array2<Complex<f32>> {
    //     Array2::from_shape_fn(
    //         DARK_PHOTON_MU_SHAPE,
    //         |(j, i)| {
    //             match i {
    //                 0 => self.x[j],
    //                 1 => self.y[j],
    //                 2 => self.z[j],
    //                 _ => unreachable!("only three components")
    //             }
    //         }
    //     )
        
    // }
    /// produces three 5 x 3 matrices out of the x, y, z components
    fn to_blocks(&self) -> DarkPhotonMuBlock {
        let block1 = Array2::from_shape_fn(
            (5, 3),
            |(j, i)| {
                match i {
                    0 => self.x[j],
                    1 => self.y[j],
                    2 => self.z[j],
                    _ => unreachable!("only three components")
                }
            }
        );
        let block2 = Array2::from_shape_fn(
            (5, 3),
            |(j, i)| {
                match i {
                    0 => self.x[j+5],
                    1 => self.y[j+5],
                    2 => self.z[j+5],
                    _ => unreachable!("only three components")
                }
            }
        );
        let block3 = Array2::from_shape_fn(
            (5, 3),
            |(j, i)| {
                match i {
                    0 => self.x[j+10],
                    1 => self.y[j+10],
                    2 => self.z[j+10],
                    _ => unreachable!("only three components")
                }
            }
        );
        DarkPhotonMuBlock { block1, block2, block3 }
    }

    fn scale(&self, factor: f32) -> Self {
        Self {
            x: (&self.x).map(|entry| entry * factor),
            y: (&self.y).map(|entry| entry * factor),
            z: (&self.z).map(|entry| entry * factor),
        }
    }
}

// #[test]
// fn test_to_matrix() {
//     use ndarray::array;
//     let dpmu = DarkPhotonMu {
//         x: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| x.into()),
//         y: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| (-x).into()),
//         z: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x|  I * x),
//     };
//     let matrix: Array2<Complex<f32>> = array![
//         [1.0_f32 * ONE, 2.0 * ONE, 3.0 * ONE, 4.0 * ONE, 5.0 * ONE, 6.0 * ONE, 7.0 * ONE, 8.0 * ONE, 9.0 * ONE, 10.0 * ONE, 11.0 * ONE, 12.0 * ONE, 13.0 * ONE, 14.0 * ONE, 15.0 * ONE],
//         [-1.0_f32 * ONE, -2.0 * ONE, -3.0 * ONE, -4.0 * ONE, -5.0 * ONE, -6.0 * ONE, -7.0 * ONE, -8.0 * ONE, -9.0 * ONE, -10.0 * ONE, -11.0 * ONE, -12.0 * ONE, -13.0 * ONE, -14.0 * ONE, -15.0 * ONE],
//         [1.0_f32 * I, 2.0 * I, 3.0 * I, 4.0 * I, 5.0 * I, 6.0 * I, 7.0 * I, 8.0 * I, 9.0 * I, 10.0 * I, 11.0 * I, 12.0 * I, 13.0 * I, 14.0 * I, 15.0 * I]
//     ];
//     assert_eq!(
//         dpmu.to_matrix(),
//         matrix,
//     );
// }

#[test]
fn test_to_blocks() {
    use ndarray::array;
    let dpmu = DarkPhotonMu {
        x: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| x.into()),
        y: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| (-x).into()),
        z: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x|  I * x),
    };
    let block1: Array2<Complex<f32>> = array![
        [1.0_f32 * ONE, 2.0 * ONE, 3.0 * ONE, 4.0 * ONE, 5.0 * ONE],
        [-1.0_f32 * ONE, -2.0 * ONE, -3.0 * ONE, -4.0 * ONE, -5.0 * ONE],
        [1.0_f32 * I, 2.0 * I, 3.0 * I, 4.0 * I, 5.0 * I]
    ];
    let block2: Array2<Complex<f32>> = array![
        [6.0 * ONE, 7.0 * ONE, 8.0 * ONE, 9.0 * ONE, 10.0 * ONE],
        [-6.0 * ONE, -7.0 * ONE, -8.0 * ONE, -9.0 * ONE, -10.0 * ONE],
        [6.0 * I, 7.0 * I, 8.0 * I, 9.0 * I, 10.0 * I]
    ];
    let block3: Array2<Complex<f32>> = array![
        [11.0 * ONE, 12.0 * ONE, 13.0 * ONE, 14.0 * ONE, 15.0 * ONE],
        [-11.0 * ONE, -12.0 * ONE, -13.0 * ONE, -14.0 * ONE, -15.0 * ONE],
        [11.0 * I, 12.0 * I, 13.0 * I, 14.0 * I, 15.0 * I]
    ];
    assert_eq!(
        dpmu.to_blocks(),
        DarkPhotonMuBlock { block1, block2, block3 },
    );
}

/// Assumes these are inclusive indices!
fn calculate_overlap(
    start1: usize,
    end1: usize,
    start2: usize,
    end2: usize
) -> usize {
    start1.max(start2)
        .checked_sub(end1.min(end2) + 1)
        .unwrap_or(0)
}

/// This function takes in the weights w_i along with the station coordinates and calculates H_i(t)
/// This doesn't necessarily need to be parallelized because this is done per coherence chunk, which is parallelized.
/// Thus, no further delegation is necessary (likely).
fn dark_photon_auxiliary_values(
    weights_n: &DashMap<StationName, f32>,
    weights_e: &DashMap<StationName, f32>,
    weights_wn: &TimeSeries,
    weights_we: &TimeSeries,
    // chunk_dataset: &DashMap<StationName, Dataset>,
) -> [TimeSeries; 7] {

    // Gather coordinate table
    let coordinates = construct_coordinate_map();

    // Get size of series
    let size = weights_wn.len();

    let auxiliary_values = [1, 2, 3, 4, 5, 6, 7].map(|i| {

        // Here we iterate thrhough weights_n and not chunk_dataset because
        // stations in weight_n are a subset (filtered) of those in chunk_dataset.
        // Could perhaps save memory by dropping coressponding invalid datasets in chunk_dataset.
        let auxiliary_value_series_unnormalized: TimeSeries = weights_n
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
                let auxiliary_value = match i {
                    
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

                auxiliary_value
            })
            .fold(TimeSeries::default(size), |acc, series| acc.add(series));

            // Divide by correct Wi
            let auxiliary_value_series_normalized = match i {
                1 => auxiliary_value_series_unnormalized.div(weights_wn),
                2 => auxiliary_value_series_unnormalized.div(weights_wn),
                3 => auxiliary_value_series_unnormalized.div(weights_we),
                4 => auxiliary_value_series_unnormalized.div(weights_we),
                5 => auxiliary_value_series_unnormalized.div(weights_we),
                6 => auxiliary_value_series_unnormalized.div(weights_we),
                7 => auxiliary_value_series_unnormalized.div(weights_we),
                _ => unreachable!("hardcoded to iterate from 1 to 7")
            }; 

        auxiliary_value_series_normalized
    });

    auxiliary_values
}

#[derive(Serialize, Deserialize)]
pub struct Power<T: Num + Clone> {
    power: Array1<Complex<T>>,
    start_sec: usize,
    end_sec: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
/// Note: these should all be 5x5 arrays
pub struct Triplet {
    pub low: Array2<Complex<f32>>,
    pub mid: Array2<Complex<f32>>,
    pub high: Array2<Complex<f32>>,
    // TODO: add something like this to ensure we are only multiplying the correct frequencies
    // pub lowf: f32,
    pub midf: f32,
    // pub hif: f32,
    // pub coh_time: usize,
    // pub chunk: usize,
    // pub window: Option<usize>
}

impl Triplet {
    fn mat_mul(&self, arrays: [Array2<Complex<f32>>; 3]) -> Triplet {
        Triplet {
            low: self.low.dot(&arrays[0]),
            mid: self.mid.dot(&arrays[1]),
            high: self.high.dot(&arrays[2]),
            // lowf: self.lowf,
            midf: self.midf,
            // hif: self.hif,
            // coh_time: self.coh_time,
            // chunk: self.chunk,
            // window: self.window,
        }
    }
    // /// Applies the map to every entry in every block
    // fn map<F>(&self, f: F) -> Triplet
    // where
    //     F: Fn(&Complex<f32>) -> Complex<f32>
    // {
    //     Triplet {
    //         low: self.low.map(&f),
    //         mid: self.mid.map(&f),
    //         high: self.high.map(&f),
    //         // lowf: self.lowf,
    //         midf: self.midf,
    //         // hif: self.hif,
    //         // coh_time: self.coh_time,
    //         // chunk: self.chunk,
    //         // window: self.window,
    //     }
    // }
    // /// This takes the [Triplet] self and multipled a [DarkPhotonVec] to produce another [DarkPhotonVec]
    // /// via matrix multiplication, i.e. A.x = y
    // fn dot_vec(&self, vecs: &DarkPhotonVec<f64>) -> DarkPhotonVec<f32> {
    //     DarkPhotonVec {
    //         low: self.low.dot(&vecs.low.map(|x| Complex { re: x.re.to_f32().unwrap(), im: x.im.to_f32().unwrap() })),
    //         mid: self.mid.dot(&vecs.mid.map(|x| Complex { re: x.re.to_f32().unwrap(), im: x.im.to_f32().unwrap() })),
    //         high: self.high.dot(&vecs.high.map(|x| Complex { re: x.re.to_f32().unwrap(), im: x.im.to_f32().unwrap() })),
    //         // coh_time: vecs.coh_time,
    //         // window: vecs.window,
    //         // chunk: vecs.chunk,
    //     }
    // }

    /// This takes the [Triplet] self and multipled a [DarkPhotonVec] to produce another [DarkPhotonVec]
    /// via matrix multiplication, i.e. A.x = y
    fn dot_vec_f32(&self, vecs: &DarkPhotonVec<f32>) -> DarkPhotonVec<f32> {
        DarkPhotonVec {
            low: self.low.dot(&vecs.low),
            mid: self.mid.dot(&vecs.mid),
            high: self.high.dot(&vecs.high),
            // coh_time: vecs.coh_time,
            // window: vecs.window,
            // chunk: vecs.chunk,
        }
    }

    /// Inverts every block in the triplet, panicking if it failed
    /// NOTE: The inverse of a block diagonal matrix is equal to a block diagonal 
    /// matrix with the inverse of the respective blocks. As such, we can do this
    /// for every triplet element independently.
    fn block_inv(&self) -> Triplet {
        Triplet {
            low: self.low.inv().expect("failed matrix inversion"),
            mid: self.mid.inv().expect("failed matrix inversion"),
            high: self.high.inv().expect("failed matrix inversion"),
            // lowf: self.lowf,
            midf: self.midf,
            // hif: self.hif,
            // coh_time: self.coh_time,
            // chunk: self.chunk,
            // window: self.window,
        }
    }
    /// Performs lower cholesky decomposition on every block in the triplet,
    /// panicking if it failed.
    /// NOTE: The cholesky decomposition of a block diagonal matrix is equal to a 
    /// block diagonal matrix with the cholesky decomposition of the blocks. As such,
    /// we can do this for every triplet element independently.
    fn block_cholesky(&self) -> Triplet {
        Triplet {
            low: self.low.cholesky(UPLO::Lower).expect("failed cholesky decomposition"),
            mid: self.mid.cholesky(UPLO::Lower).expect("failed cholesky decomposition"),
            high: self.high.cholesky(UPLO::Lower).expect("failed cholesky decomposition"),
            // lowf: self.lowf,
            midf: self.midf,
            // hif: self.hif,
            // coh_time: self.coh_time,
            // chunk: self.chunk,
            // window: self.window,
        }
    }

    // /// Performs singular value decomposition on every block in the triplet,
    // /// panicking if it failed.
    // /// NOTE: The singular value decomposition of a block diagonal matrix is equal to a 
    // /// block diagonal matrix with the singular value decomposition of the blocks. As such,
    // /// we can do this for every triplet element independently.
    // /// 
    // /// TODO: ensure the middle array is correct (i.e. not transposed)
    // fn block_svd(&self) -> (Triplet, Array2<f32>, Triplet) {
    //     let (ulow, slow, vlow) = self.low.svd(true, true).expect("svd failed");
    //     let (umid, smid, vmid) = self.mid.svd(true, true).expect("svd failed");
    //     let (uhigh, shigh, vhigh) = self.high.svd(true, true).expect("svd failed");
    //     (Triplet {
    //         low: ulow.expect("requested u"),
    //         mid: umid.expect("requested u"),
    //         high: uhigh.expect("requested u"),
    //         // lowf: self.lowf,
    //         midf: self.midf,
    //         // hif: self.hif,
    //         // coh_time: self.coh_time,
    //         // chunk: self.chunk,
    //         // window: self.window,
    //     },
    //     Array2::from_shape_vec((3, 3), slow.into_iter().chain(smid.into_iter()).chain(shigh.into_iter()).collect()).expect("should be correct shape"),
    //     Triplet {
    //         low: vlow.expect("requested v"),
    //         mid: vmid.expect("requested v"),
    //         high: vhigh.expect("requested v"),
    //         // lowf: self.lowf,
    //         midf: self.midf,
    //         // hif: self.hif,
    //         // coh_time: self.coh_time,
    //         // chunk: self.chunk,
    //         // window: self.window,
    //     })
    // }

    // /// Tranposes and conjugates every block
    // fn dagger(&self) -> Self {
    //     Triplet {
    //         low: self.low.t().map(|c| c.conj()),
    //         mid: self.mid.t().map(|c| c.conj()),
    //         high: self.high.t().map(|c| c.conj()),
    //         // lowf: self.lowf,
    //         midf: self.midf,
    //         // hif: self.hif,
    //         // coh_time: self.coh_time,
    //         // chunk: self.chunk,
    //         // window: self.window,
    //     }
    // }

    // /// Scale all blocks by the same scalar factor
    // fn scale(&self, factor: f32) -> Self {
    //     Self {
    //         low: (&self.low).mul(factor),
    //         mid: (&self.mid).mul(factor),
    //         high: (&self.high).mul(factor),
    //         // lowf: self.lowf,
    //         midf: self.midf,
    //         // hif: self.hif,
    //         // coh_time: self.coh_time,
    //         // chunk: self.chunk,
    //         // window: self.window,
    //     }
    // }
}

// #[derive(Serialize, Deserialize, Clone, Copy)]
// pub struct Triplet<T: Num + Clone> {
//     low: SinglePower<T>,
//     mid: SinglePower<T>,
//     high: SinglePower<T>, 
//     window: usize,
// }
// impl<T: Num + Clone + Debug> std::ops::AddAssign for Triplet<T>
// where
//     Complex<T>: std::ops::AddAssign
// {
//     fn add_assign(&mut self, rhs: Self) {
//         assert_eq!(self.window, rhs.window, "invalid window for Triplet add_assign");
//         self.low += rhs.low;
//         self.mid += rhs.mid;
//         self.high += rhs.high;
//     }
// }
// impl<T: Num + Clone + Debug> std::ops::AddAssign for SinglePower<T>
// where
//     Complex<T>: std::ops::AddAssign
// {
//     fn add_assign(&mut self, rhs: Self) {
//         assert_eq!(self.frequency, rhs.frequency, "invalid frequency for Triplet add_assign");
//         self.power += rhs.power;
//     }
// }

// #[derive(Serialize, Deserialize, Clone, Copy)]
// pub struct SinglePower<T: Num + Clone> {
//     power: Complex<T>,
//     frequency: T,
// }

impl<T: Float> Power<T> {
    pub fn map<'a, B, F>(&'a self, f: F) -> Power<T>
    where
        F: FnMut(&'a Complex<T>) -> Complex<T>,
        T: 'a,
    {
        Power {
            power: self.power.map(f),
            ..*self
        }
    }
    pub fn mul<'a>(&'a self, rhs: &Array1<Complex<T>>) -> Power<T> {
        Power {
            power: (&self.power).mul(rhs),
            ..*self
        }
    }
    pub fn add<'a>(&'a self, rhs: &Array1<Complex<T>>) -> Power<T> {
        Power {
            power: (&self.power).add(rhs),
            ..*self
        }
    }
    /// NOTE: these assign operations are not optimal. Perhaps a future TODO.
    pub fn add_assign<'a>(&'a mut self, rhs: &Array1<Complex<T>>) {
        self.power = (&self.power).add(rhs);
    }
    pub fn add_assign_scalar<'a, S>(&'a mut self, rhs: S)
    where
        S: Into<Complex<T>> + Copy,
    {
        self.power = (&self.power).map(|x| x + rhs.into());
    }
    pub fn div_assign<'a>(&'a mut self, rhs: &Array1<Complex<T>>) {
        self.power = (&self.power).div(rhs);
    }
    pub fn div_assign_scalar<'a, S>(&'a mut self, rhs: S)
    where
        S: Into<Complex<T>> + Copy
    {
        self.power = (&self.power).map(|x| x / rhs.into());
    }
    pub fn mul_scalar<'a, S: ScalarOperand>(&'a self, rhs: S) -> Power<T>
    where
        T: Mul<S, Output=T>,
        Complex<T>: Mul<S, Output=Complex<T>>,
    {
        Power {
            power: (&self.power).mul(rhs),
            ..*self
        }
    }
    // TODO: ensure this is returning the right frequencies
    fn frequencies(&self) ->  Vec<f32> {
        get_frequency_range_1s(TAU.try_into().expect("usize to i64 failed"))
            .into_iter()
            .map(|double| double.to_f32().expect("double to single precision failed"))
            .collect()
    }
}

/// This calculates the fft of a tophat function starting at t=0 and going to t=T, 
/// the length of the longest contiguous subset of the dataset Xi(t). 
/// 
/// To reiterate/clarify, this function assumes there are no gaps in data availability.
/// 
/// At k = 0, this results in 0/0, requiring the use of the L'Hospital rule. As such,
/// this case is dealt separately.
// fn one_tilde(
//     coherence_time: f64,
//     chunk_index: usize,
//     number_of_chunks: usize,
//     k: f64,
// ) -> Complex<f64> {
//     match k {
        
//         // Deal with L'Hospital limit separately
//         _k if k.is_zero() => Complex::new(coherence_time, 0.0),

//         // Otherwise, evaluate function
//         _ => (Complex::new(0.0, (1.0 - ((chunk_index + number_of_chunks) as f64 * coherence_time)/number_of_chunks as f64) * k).exp())*(-1.0 + Complex::new(0.0, coherence_time * k).exp())
//                 / (-1.0 + Complex::new(0.0, k).exp())
//     }
// }

#[derive(Serialize, Deserialize, Debug)]
pub struct DarkPhotonAuxiliary {
    pub h: [TimeSeries; 7]
}

impl FromChunkMap for DarkPhotonAuxiliary {
    fn from_chunk_map(
        auxiliary_values_chunked: &DashMap<usize, Self>,
        secs_per_chunk: usize,
        starting_value: usize,
        size: usize,
    ) -> Self {

        // Initialie auxiliary array to zeros
        let mut auxiliary_values = [(); 7]
            .map(|_| TimeSeries::zeros(size * secs_per_chunk));

        auxiliary_values_chunked
            .iter()
            .for_each(|chunk| {

                // Get chunk and it's auxiliary_value
                let (&current_chunk, chunk_auxiliary) = chunk.pair();

                if !in_longest_subset(current_chunk, size, starting_value) {
                    return ()
                }

                // Insert array into complete series
                // TODO: THIS ASSUMES ALL CHUNKS ARE THE SAME LENGTH.
                // NEED TO CHANGE FOR YEARLY STATIONARITY AND PERHAPS THE EDGE CHUNKS.
                let start_index = (current_chunk-starting_value)*chunk_auxiliary.h[0].len();
                let end_index = start_index + chunk_auxiliary.h[0].len();

                for i in 0..7 {
                    auxiliary_values
                    .get_mut(i)
                    .unwrap()
                    .slice_mut(s![start_index..end_index])
                    .assign(&chunk_auxiliary.h[i]);
                }
            });
                    
        
        DarkPhotonAuxiliary { h: auxiliary_values }
    }
}