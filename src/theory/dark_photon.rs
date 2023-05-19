use super::*;
use std::{collections::HashMap, io::{Write, Read}, sync::atomic::AtomicUsize, ops::AddAssign};
use crossbeam_channel::{bounded, unbounded};
use dashmap::{DashMap};
use interp1d::Interp1d;
use itertools::Itertools;
use rocksdb::{DBWithThreadMode, SingleThreaded, MultiThreaded};
use rustfft::FftPlanner;
use serde_derive::{Serialize, Deserialize};
use tokio::sync::Semaphore;
use std::sync::Arc;
use crate::{
    utils::{
    loader::Dataset,
    approximate_sidereal, coordinates::Coordinates, fft::get_frequency_range_1s, svd::compact_svd, sec_to_year,
    },
    constants::SIDEREAL_DAY_SECONDS,
};
use std::{
    ops::{Mul, Div, Add, Range, Sub},
};
use ndrustfft::{FftHandler, ndfft_par, Zero};
use rayon::prelude::*;
use ndarray::{s, ScalarOperand, Array2, Axis};
use num_traits::{ToPrimitive, Float, Num};
use ndrustfft::Complex;
use std::sync::{Mutex, atomic::{Ordering, AtomicU32}};
use std::f64::consts::PI as DOUBLE_PI;
use indicatif::{ProgressBar, MultiProgress, ProgressStyle};
use ndarray_linalg::{Cholesky, solve::Inverse, UPLO};


const ZERO: Complex<FloatType> = Complex::new(0.0, 0.0);
const ONE: Complex<FloatType> = Complex::new(1.0, 0.0);

/// Size of chunks (used for noise spectra ffts)
// const TAU: usize = 16384 * 64;
// TODO: revert
// const TAU_REDUCTION_FACTOR_TEST: usize = 8;
// const TAU: usize = 16384 * 64 / TAU_REDUCTION_FACTOR_TEST;
// const TAU: usize = 16384 * 64 / 16;
const TAU: usize = 65_536;
const MAX_FREQUENCY: FloatType = (TAU - 1) as FloatType / TAU as FloatType;

type Chunk = usize;
type Window = usize;
pub type InnerVarChunkWindowMap = DashMap<(Chunk, Window), Triplet>;
pub type InnerAChunkWindowMap = DashMap<Window, DashMap<Triplet, (Array2<Complex<f32>>,  Array2<Complex<f32>>)>>;


type DarkPhotonVecSphFn = Arc<dyn Fn(FloatType, FloatType) -> FloatType + Send + 'static + Sync>;

#[derive(Clone)]
pub struct DarkPhoton {
    pub vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>>,
    // The RwLock guarantees only one thread is every modifying the database, making SingleThreaded safe
    pub data_vector: Arc<DBWithThreadMode<MultiThreaded>>,
    pub theory_mean: Arc<DBWithThreadMode<MultiThreaded>>,
    pub theory_var: Arc<DiskDB>,
}


lazy_static! {
    static ref DARK_PHOTON_MODES: Vec<Mode> = vec![
        Mode(1,  0),
        Mode(1, -1),
        Mode(1,  1),
    ];

    pub static ref DARK_PHOTON_NONZERO_ELEMENTS: Vec<NonzeroElement> = vec![
        // "X1",
        NonzeroElement {
            index: 1,
            assc_mode: (Mode(1, -1), Component::PolarReal)
        },
        // "X2"
        NonzeroElement {
            index: 2,
            assc_mode: (Mode(1, -1), Component::PolarImag)
        },
        // X3
        NonzeroElement {
            index: 3,
            assc_mode: (Mode(1, -1), Component::AzimuthReal)
        },
        // X4
        NonzeroElement {
            index: 4,
            assc_mode: (Mode(1, -1), Component::AzimuthImag)
        },
        // X5
        NonzeroElement {
            index: 5,
            assc_mode: (Mode(1,  0), Component::AzimuthReal)
        },
    ];
}


impl DarkPhoton {

    /// This initilaizes a `DarkPhoton` struct. This struct is to be used during an analysis to produce
    /// data vectors and signals after implementing `Theory`.
    pub fn new() -> Self {

        // Calculate vec_sphs at each station
        // let vec_sph_fns = Arc::new(vector_spherical_harmonics(DARK_PHOTON_MODES.clone().into_boxed_slice()));


        // Manual override to remove prefactors
        let vec_sph_fns: Arc<DashMap<NonzeroElement, DarkPhotonVecSphFn>> = Arc::new(DashMap::new());
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[0],
            Arc::new(|_theta: FloatType, phi: FloatType| -> FloatType {
                phi.sin()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[1],
            Arc::new(|_theta: FloatType, phi: FloatType| -> FloatType {
                phi.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[2],
            Arc::new(|theta: FloatType, phi: FloatType| -> FloatType {
                phi.cos() * theta.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[3],
            Arc::new(|theta: FloatType, phi: FloatType| -> FloatType {
                - phi.sin() * theta.cos()
            }));
        vec_sph_fns.insert(
            DARK_PHOTON_NONZERO_ELEMENTS[4],
            Arc::new(|theta: FloatType, _phi: FloatType| -> FloatType {
                theta.sin()
            }));

        // Sanity check on our own work
        assert_eq!(vec_sph_fns.len(), Self::NONZERO_ELEMENTS);

        DarkPhoton {
            vec_sph_fns,
            data_vector: Arc::new(DBWithThreadMode::<MultiThreaded>::open_default("dark_photon_data_vector").expect("data vector db failed to open")),
            theory_mean: Arc::new(DBWithThreadMode::<MultiThreaded>::open_default("dark_photon_theory_mean").expect("theory mean db failed to open")),
            theory_var: Arc::new(DiskDB::connect("dark_photon_theory_var").expect("theory var db")),
        }

    }

}

impl Theory for DarkPhoton {

    const MIN_STATIONS: usize = 3;
    const NONZERO_ELEMENTS: usize = 5;

    type AuxiliaryValue = DarkPhotonAuxiliary;
    // type Mu = DarkPhotonMu;
    // type Var = InnerVarChunkWindowMap;
    // type DataVector = DashMap<usize, DashMap<NonzeroElement, Vec<(Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<Complex<f64>>)>>>;
    // type DataVector = DashMap<usize /* Tc */, DashMap<usize /* chunk */, DashMap<usize /* window/triplet */, DarkPhotonVec<f32>>>>;
    // type DataVector = ReadOnlyView<(CoherenceTime, Chunk, Window), DarkPhotonVec<f32>>;

    fn get_nonzero_elements() -> HashSet<NonzeroElement> {

        let mut nonzero_elements = HashSet::new();

        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[0]);
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[1]);
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[2]);
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[3]);
        nonzero_elements.insert(DARK_PHOTON_NONZERO_ELEMENTS[4]);

        nonzero_elements
    }

    fn calculate_projections(
        &self,
        weights_n: &DashMap<StationName, FloatType>,
        weights_e: &DashMap<StationName, FloatType>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        chunk_dataset: &DashMap<StationName, Dataset>,
    ) -> DashMap<NonzeroElement, TimeSeries> {

        // Initialize projection table
        let projection_table = DashMap::new();

        // for nonzero_element in DARK_PHOTON_NONZERO_ELEMENTS.iter() {
        DARK_PHOTON_NONZERO_ELEMENTS.par_iter().for_each(|nonzero_element| {

            // size of dataset
            let size = chunk_dataset.iter().next().unwrap().value().field_1.len();

            // Here we iterate thrhough weights_n and not chunk_dataset because
            // stations in weight_n are a subset (filtered) of those in chunk_dataset.
            // Could perhaps save memory by dropping coressponding invalid datasets in chunk_dataset.
            let combined_time_series: TimeSeries = weights_n
                .par_iter()
                .map(|key_value| {

                    // Unpack (key, value) pair
                    // Here, key is StationName and value = dataset
                    let station_name = key_value.key();
                    let dataset = chunk_dataset.get(station_name).unwrap();

                    log::trace!("calculating projections X{} for {station_name} on index {}", nonzero_element.index, dataset.index);

                    // Get product of relevant component of vector spherical harmonics and of the magnetic field.
                    let relevant_product = match nonzero_element.assc_mode {
                        (_, component) => {

                            // Get relevant vec_sph_fn
                            let vec_sph_fn = self.vec_sph_fns.get(&nonzero_element).unwrap();

                            // These were commented out to match definitions in the paper
                            // let relevant_vec_sph = match component {
                            //     Component::PolarReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].re,
                            //     Component::PolarImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[0].im,
                            //     Component::AzimuthReal =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].re,
                            //     Component::AzimuthImag =>  vec_sph_fn(dataset.coordinates.polar as f32, dataset.coordinates.longitude as f32).phi[1].im,
                            //     _ => panic!("not included in dark photon"),
                            // };

                            // Manual Override
                            let relevant_vec_sph = vec_sph_fn(dataset.coordinates.polar as FloatType, dataset.coordinates.longitude as FloatType);
                            assert!(!relevant_vec_sph.is_nan(), "spherical harmonic is nan");

                            // Note that this multiplies the magnetic field by the appropriate weights, so it's not quite the measured magnetic field
                            let relevant_mag_field = match component {
                                Component::PolarReal => (&dataset.field_1).mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::PolarImag => (&dataset.field_1).mul(*weights_n.get(station_name).unwrap()).div(weights_wn),
                                Component::AzimuthReal => (&dataset.field_2).mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                Component::AzimuthImag => (&dataset.field_2).mul(*weights_e.get(station_name).unwrap()).div(weights_we),
                                _ => panic!("not included in dark photon"),
                            };
                            assert!(relevant_mag_field.iter().all(|m| !m.is_nan()), "mag_field contains nan");


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
        });

        projection_table
    }

    /// This calculates the auxiliary values for a chunk.
    fn calculate_auxiliary_values(
        &self,
        weights_n: &DashMap<StationName, FloatType>,
        weights_e: &DashMap<StationName, FloatType>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        _chunk_dataset: &DashMap<StationName, Dataset>,
        valid_entry_map: &DashMap<StationName, Array1<bool>>,
    ) -> Self::AuxiliaryValue {
        DarkPhotonAuxiliary {
            h: dark_photon_auxiliary_values(weights_n, weights_e, weights_wn, weights_we, valid_entry_map),
        }
    }

    // Returns true if there are no nans
    fn check_aux_for_nan(auxiliary_values: &Self::AuxiliaryValue) -> bool {
        auxiliary_values.h.par_iter().all(|h| h.par_iter().all(|h_| !h_.is_nan()))
    }

    fn calculate_data_vector(
        &self,
        projections_complete: &ProjectionsComplete,
        local_set: &Vec<(usize, FrequencyBin)>,
    ) {
        // Clear database
        self.data_vector.iterator(rocksdb::IteratorMode::Start).map(|x| x.unwrap()).par_bridge().for_each(|(key, _)| {
            drop(self.data_vector.delete(key));
        });

        // Local references
        let dp1 = &DARK_PHOTON_NONZERO_ELEMENTS[0];
        let dp2 = &DARK_PHOTON_NONZERO_ELEMENTS[1];
        let dp3 = &DARK_PHOTON_NONZERO_ELEMENTS[2];
        let dp4 = &DARK_PHOTON_NONZERO_ELEMENTS[3];
        let dp5 = &DARK_PHOTON_NONZERO_ELEMENTS[4];

        // KeySet
        let keyset = DashMap::new(); 

        log::debug!("starting data vector");
        let multi = MultiProgress::new();
        let sty = ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap();
        let ch_pb = multi.add(ProgressBar::new(local_set.len() as u64));
        ch_pb.set_style(sty.clone());
        ch_pb.set_message("coherence times");
        ch_pb.inc(0);


        let fft_semaphore = Semaphore::const_new(1);
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || {
            let mut file = std::fs::File::create("chunk_window.txt").unwrap();
            while let Ok((chunk_index, window_index)) = rx.recv() {
                file.write(format!("{chunk_index} {window_index}\n").as_bytes()).unwrap();
            }
        });

        local_set
            .par_iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* FrequencyBin */)| {

                let series_pb = multi.add(ProgressBar::new(5));
                series_pb.set_style(sty.clone());
                series_pb.set_message("projection element X_i");
                series_pb.inc(0);

                // Calculate number of exact chunks, and the total size of all exact chunks
                let exact_chunks: usize = projections_complete.num_secs() / coherence_time;
                let exact_chunks_size: usize = exact_chunks * coherence_time;
                let last_chunk_size = projections_complete.num_secs() % exact_chunks_size;
                let total_chunks = exact_chunks + if last_chunk_size > 0 { 1 } else { 0 };
                log::info!("data vector: {coherence_time} has {total_chunks} chunks");

                // Parallelize over all nonzero elements in the theory
                projections_complete
                    .projections_complete
                    .iter()
                    .for_each(|series| {

                        // Unpack element, series
                        let (element, series) = series.pair();

                        // Chunk series
                        log::trace!("constructing 2d series");
                        let mut two_dim_series: Array2<Complex<FloatType>> = series
                            .slice(s!(0..exact_chunks_size))
                            .into_shape((exact_chunks, *coherence_time))
                            .expect("This shouldn't fail ever. If anything we should get an early panic from .slice()")
                            .map(|x| x.into())
                            .t()
                            .to_owned();
                        
                            log::trace!("appending padded chunk");
                            if last_chunk_size > 0 {
                            let mut last_chunk_pad = Array1::<Complex<FloatType>>::zeros(*coherence_time);
                            last_chunk_pad
                                .slice_mut(s![0..last_chunk_size])
                                .assign(&series
                                    .slice(s![exact_chunks_size..])
                                    .map(|x| x.into())
                                );
                            two_dim_series.push_column(last_chunk_pad.view()).expect("failed to push row");
                        }
                        drop(series); // to save memory

                        // Do ffts
                        let num_fft_elements = *coherence_time;
                        let permit = fft_semaphore.acquire();
                        // let mut fft_handler = FftHandler::<f32>::new(*coherence_time);
                        // let mut fft_result = ndarray::Array2::<Complex<f32>>::zeros((num_fft_elements, total_chunks));
                        // ndfft_par(&two_dim_series, &mut fft_result, &mut fft_handler, 0);
                        let mut planner = FftPlanner::new();
                        log::trace!("performing ffts");
                        for mut column in two_dim_series.columns_mut() {
                            let fft_handler = planner.plan_fft_forward(*coherence_time);
                            fft_handler.process(column.as_slice_mut().unwrap());
                        }
                        let mut fft_result = two_dim_series;


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
                        // Reduce to save memory!
                        fft_result.slice_axis_inplace(ndarray::Axis(0), ndarray::Slice::from(relevant_range));//.slice_axis(ndarray::Axis(0), ndarray::Slice::from(relevant_range)).to_owned();
                        drop(permit);
                        
                        let num_windows = (fft_result.shape()[0] - (2*approx_sidereal + 1) + 1) as u64;
                        let window_pb = multi.add(ProgressBar::new(num_windows));
                        window_pb.set_style(sty.clone());
                        window_pb.set_message("windows");
                        log::info!("data vector: {coherence_time} has {num_windows} windows");


                        fft_result
                            .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                            .into_iter()
                            .enumerate()
                            .par_bridge()
                            .for_each(|(window_index, window)|{
                                
                                assert_eq!(window.shape(), &[2*approx_sidereal + 1, total_chunks]);
                                let low = window.slice(s![0_usize, ..]);
                                let mid = window.slice(s![approx_sidereal, ..]);
                                let high = window.slice(s![2*approx_sidereal, ..]);
                                assert_eq!(low.shape(), &[total_chunks]);
                                assert_eq!(mid.shape(), &[total_chunks]);
                                assert_eq!(high.shape(), &[total_chunks]);

                                (0..total_chunks)
                                    .into_par_iter()
                                    .for_each(|chunk_index| {

                                        tx.send((chunk_index, window_index)).unwrap();

                                        // data_vector_map
                                        // Insert lock if nonexistent
                                        let key = [*coherence_time, chunk_index, window_index].map(usize::to_le_bytes).concat();
                                        keyset.insert(key.clone(), ());

                                        // From here until the key is dropped there exists only one thread
                                        // which will load/modify/store this key/value
                                        let key_lock = keyset.get_mut(&key);

                                        // Try to get element
                                        let entry: Option<Vec<u8>> = self.data_vector.get(&key).expect("rocksdb failure");

                                        self.data_vector.put(&key, entry.map_or_else(
                                            // Create data vector element if it doesn't exist
                                            || {
                                                // Initialize with zeros and metadata
                                                let mut inner_data_vector = DarkPhotonVec {
                                                    low: Array1::<Complex<FloatType>>::zeros(5),
                                                    mid: Array1::<Complex<FloatType>>::zeros(5),
                                                    high: Array1::<Complex<FloatType>>::zeros(5),
                                                };

                                                match element {
                                                    e if e == dp1 => {
                                                        assert!(inner_data_vector.low[0].is_zero());
                                                        assert!(inner_data_vector.mid[0].is_zero());
                                                        assert!(inner_data_vector.high[0].is_zero());
                                                        inner_data_vector.low[0] = low[chunk_index];
                                                        inner_data_vector.mid[0] = mid[chunk_index];
                                                        inner_data_vector.high[0] = high[chunk_index];
                                                    },
                                                    e if e == dp2 => {
                                                        assert!(inner_data_vector.low[1].is_zero());
                                                        assert!(inner_data_vector.mid[1].is_zero());
                                                        assert!(inner_data_vector.high[1].is_zero());
                                                        inner_data_vector.low[1] = low[chunk_index];
                                                        inner_data_vector.mid[1] = mid[chunk_index];
                                                        inner_data_vector.high[1] = high[chunk_index];
                                                    },
                                                    e if e == dp3 => {
                                                        assert!(inner_data_vector.low[2].is_zero());
                                                        assert!(inner_data_vector.mid[2].is_zero());
                                                        assert!(inner_data_vector.high[2].is_zero());
                                                        inner_data_vector.low[2] = low[chunk_index];
                                                        inner_data_vector.mid[2] = mid[chunk_index];
                                                        inner_data_vector.high[2] = high[chunk_index];
                                                    },
                                                    e if e == dp4 => {
                                                        assert!(inner_data_vector.low[3].is_zero());
                                                        assert!(inner_data_vector.mid[3].is_zero());
                                                        assert!(inner_data_vector.high[3].is_zero());
                                                        inner_data_vector.low[3] = low[chunk_index];
                                                        inner_data_vector.mid[3] = mid[chunk_index];
                                                        inner_data_vector.high[3] = high[chunk_index];
                                                    },
                                                    e if e == dp5 => {
                                                        assert!(inner_data_vector.low[4].is_zero());
                                                        assert!(inner_data_vector.mid[4].is_zero());
                                                        assert!(inner_data_vector.high[4].is_zero());
                                                        inner_data_vector.low[4] = low[chunk_index];
                                                        inner_data_vector.mid[4] = mid[chunk_index];
                                                        inner_data_vector.high[4] = high[chunk_index];
                                                    },
                                                    _ => unreachable!("dark photon only has these 5 nonzero elements"),
                                                };
                                                bincode::serialize(&inner_data_vector).unwrap()

                                            // Otherwise update existing element
                                            },|inner_data_vector_bytes: Vec<u8>| {
                                                let mut inner_data_vector: DarkPhotonVec<FloatType> = bincode::deserialize(&inner_data_vector_bytes).unwrap();
                                                match element {
                                                    e if e == dp1 => {
                                                        assert!(inner_data_vector.low[0].is_zero());
                                                        assert!(inner_data_vector.mid[0].is_zero());
                                                        assert!(inner_data_vector.high[0].is_zero());
                                                        inner_data_vector.low[0] = low[chunk_index];
                                                        inner_data_vector.mid[0] = mid[chunk_index];
                                                        inner_data_vector.high[0] = high[chunk_index];
                                                    },
                                                    e if e == dp2 => {
                                                        assert!(inner_data_vector.low[1].is_zero());
                                                        assert!(inner_data_vector.mid[1].is_zero());
                                                        assert!(inner_data_vector.high[1].is_zero());
                                                        inner_data_vector.low[1] = low[chunk_index];
                                                        inner_data_vector.mid[1] = mid[chunk_index];
                                                        inner_data_vector.high[1] = high[chunk_index];
                                                    },
                                                    e if e == dp3 => {
                                                        assert!(inner_data_vector.low[2].is_zero());
                                                        assert!(inner_data_vector.mid[2].is_zero());
                                                        assert!(inner_data_vector.high[2].is_zero());
                                                        inner_data_vector.low[2] = low[chunk_index];
                                                        inner_data_vector.mid[2] = mid[chunk_index];
                                                        inner_data_vector.high[2] = high[chunk_index];
                                                    },
                                                    e if e == dp4 => {
                                                        assert!(inner_data_vector.low[3].is_zero());
                                                        assert!(inner_data_vector.mid[3].is_zero());
                                                        assert!(inner_data_vector.high[3].is_zero());
                                                        inner_data_vector.low[3] = low[chunk_index];
                                                        inner_data_vector.mid[3] = mid[chunk_index];
                                                        inner_data_vector.high[3] = high[chunk_index];
                                                    },
                                                    e if e == dp5 => {
                                                        assert!(inner_data_vector.low[4].is_zero());
                                                        assert!(inner_data_vector.mid[4].is_zero());
                                                        assert!(inner_data_vector.high[4].is_zero());
                                                        inner_data_vector.low[4] = low[chunk_index];
                                                        inner_data_vector.mid[4] = mid[chunk_index];
                                                        inner_data_vector.high[4] = high[chunk_index];
                                                    },
                                                    _ => unreachable!("dark photon only has these 5 nonzero elements"),
                                                };
                                                bincode::serialize(&inner_data_vector).unwrap()
                                            })).expect("rocksdb failure");

                                        // drop lock explicity for clarity
                                        drop(key_lock);
                                });
                                window_pb.inc(1);
                            });
                        window_pb.finish_and_clear();
                        series_pb.inc(1);
                    });
                series_pb.finish_and_clear();
                ch_pb.inc(1);
            });
        ch_pb.finish_and_clear();

        drop(tx);
        handle.join().unwrap();
    }

    /// NOTE: this implementation assumes that the time series is fully contiguous (with no null values).
    /// As such, it will produce incorrect results if used with a dataset that contains null values.
    fn calculate_theory_mean(
        &self,
        set: &Vec<(usize, FrequencyBin)>,
        len_data: usize,
        _coherence_times: usize,
        auxiliary_values: Arc<Self::AuxiliaryValue>,
    ) {

        // Calculate the sidereal day frequency
        const FD: f64 = 1.0 / SIDEREAL_DAY_SECONDS;

        // Map of Map of mus
        // First key is coherence time
        // Second key is chunk index
        log::trace!("theory mean: starting loop for {} coherence times", set.len());
        set
            // .into_iter()
            .into_par_iter()
            .for_each(|(coherence_time /* usize */, frequency_bin /* &FrequencyBin */)| {

                // For the processed, cleaned dataset, this is 
                // the number of chunks for this coherence time
                let num_chunks = len_data / coherence_time;
                let last_chunk = len_data % coherence_time > 0;
                let total_chunks = num_chunks + if last_chunk { 1 } else { 0 };
                log::info!("theory mean: coherence_time {coherence_time} has {total_chunks} chunks");
                
                // TODO: refactor elsewhere to be user input or part of some fit
                const RHO: FloatType = 6.04e7;
                const R: FloatType = 0.0212751;
                let mux_prefactor: FloatType = DOUBLE_PI as FloatType * R * (2.0 * RHO).sqrt() / 4.0;

                // This is the inner chunk map from chunk to coherence time
                let inner_chunk_map = DashMap::new();

                (0..total_chunks).into_par_iter().for_each(|chunk| {

                    // Beginning and end index for this chunk in the total series
                    // NOTE: `end` is exclusive
                    let start: usize  = chunk * coherence_time;
                    let end: usize = ((chunk + 1) * coherence_time).min(len_data);

                    // Calculate cos + isin. Unlike the original implementation, this is done using euler's 
                    // exp(ix) = cos(x) + i sin(x)
                    //
                    // Note: when you encounter a chunk that has total time < coherence time, the s![start..end] below will truncate it.
                    // let cis_fh_f = Array1::range(0.0, *coherence_time as FloatType, 1.0)
                    let cis_fh_f = (start..end)
                        .map(|x| Complex { re: x as FloatType, im: 0.0 })
                        .collect::<Array1<Complex<FloatType>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                // 2 * PI * (fdhat - fd)
                                im: 2.0 * DOUBLE_PI as FloatType * ((approximate_sidereal(frequency_bin) as FloatType * frequency_bin.lower as FloatType) - FD as FloatType),
                            }
                        )
                        .mapv(Complex::exp);
                    // let cis_f = Array1::range(0.0, *coherence_time as FloatType, 1.0)
                    let cis_f = (start..end)
                        .map(|x| Complex { re: x as FloatType, im: 0.0 })
                        .collect::<Array1<Complex<FloatType>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                im: 2.0 * DOUBLE_PI as FloatType * FD as FloatType,
                            }
                        )
                        .mapv(Complex::exp);
                    // let cis_f_fh = Array1::range(0.0, *coherence_time as FloatType, 1.0)
                    let cis_f_fh = (start..end)
                        .map(|x| Complex { re: x as FloatType, im: 0.0 })
                        .collect::<Array1<Complex<FloatType>>>()
                        .mul(
                            Complex {
                                re: 0.0, 
                                // This minus sign flips (fdhat-fd) --> (fd-fdhat)
                                im: -2.0 * DOUBLE_PI as FloatType * ((approximate_sidereal(frequency_bin) as FloatType * frequency_bin.lower as FloatType)- FD as FloatType),
                            }
                        )
                        .mapv(Complex::exp);
                    
                    // Get references to auxiliary values for this chunk for better readability
                    let h1 = auxiliary_values.h[0].slice(s![start..end]);
                    let h2 = auxiliary_values.h[1].slice(s![start..end]);
                    let h3 = auxiliary_values.h[2].slice(s![start..end]);
                    let h4 = auxiliary_values.h[3].slice(s![start..end]);
                    let h5 = auxiliary_values.h[4].slice(s![start..end]);
                    let h6 = auxiliary_values.h[5].slice(s![start..end]);
                    let h7 = auxiliary_values.h[6].slice(s![start..end]);

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
                    let mux5: Complex<FloatType> = {

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
                    let mux6: Complex<FloatType> = {

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
                    let mux7: Complex<FloatType> = {

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
                    let mux8: Complex<FloatType> = {

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
                    let mux9: Complex<FloatType> = {

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
                    let mux10: Complex<FloatType> = (&cis_fh_f)
                        .mul(h1
                                .iter()
                                .zip(h2)
                                .map(|(&h1_, &h2_)| Complex::new(1.0-h1_, -h2_))
                                .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux11 is FT of (H2 - iH1) at f = fdhat-fd
                    let mux11: Complex<FloatType> = (&cis_fh_f)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h2_, -h1_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux12 is FT of (H4 + iH5) at f = fdhat-fd
                    let mux12: Complex<FloatType> = (&cis_fh_f)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4_, &h5_)| Complex::new(h4_, h5_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux13 is FT of (-H5 + i*(H4 - H3)) at f = fdhat-fd
                    let mux13: Complex<FloatType> = (&cis_fh_f)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3_, &h4_), &h5_)| Complex::new(-h5_, h4_-h3_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // mux14 is FT of (H6 + iH7) at f = fdhat-fd
                    let mux14: Complex<FloatType> = (&cis_fh_f)
                        .mul(h6
                            .iter()
                            .zip(h7)
                            .map(|(&h6_, &h7_)| Complex::new(h6_, h7_))
                            .collect::<Array1<_>>())
                        .mul(mux_prefactor)
                        .sum();

                    // start of muy
                    let muy_prefactor: FloatType = -mux_prefactor;

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
                    let muy5: Complex<FloatType> = {

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
                    let muy6: Complex<FloatType> = {

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
                    let muy7: Complex<FloatType> = {

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
                    let muy8: Complex<FloatType> = {

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
                    let muy9: Complex<FloatType> = {

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
                    let muy10: Complex<FloatType> = (&cis_fh_f)
                        .mul(Complex::<FloatType>::new(1.0, 0.0)
                            .add(h1
                                .iter()
                                .zip(h2)
                                .map(|(&h1_, &h2_)| Complex::new(h2_, 1.0-h1_))
                                .collect::<Array1<_>>()))
                        .mul(muy_prefactor)
                        .sum();

                    // muy11 is FT(H1 + iH2) at f = fdhat - fd
                    let muy11: Complex<FloatType> = (&cis_fh_f)
                        .mul(h1
                            .iter()
                            .zip(h2)
                            .map(|(&h1_, &h2_)| Complex::new(h1_, h2_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy12 is FT(-H5 + iH4) at f = fdhat-fd
                    let muy12: Complex<FloatType> = (&cis_fh_f)
                        .mul(h4
                            .iter()
                            .zip(h5)
                            .map(|(&h4_, &h5_)| Complex::new(-h5_, h4_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy13 is FT(H3-H4-iH5) at f = fdhat-fd
                    let muy13: Complex<FloatType> = (&cis_fh_f)
                        .mul(h3
                            .iter()
                            .zip(h4)
                            .zip(h5)
                            .map(|((&h3_, &h4_), &h5_)| Complex::new(h3_ - h4_, -h5_))
                            .collect::<Array1<_>>())
                        .mul(muy_prefactor)
                        .sum();

                    // muy14 is FT of (-H7 + iH6) at f = fdhat-fd
                    let muy14: Complex<FloatType> = (&cis_fh_f)
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
                    let muz_prefactor: FloatType = 2.0 * mux_prefactor;
                    
                    // muz 2, 3, 4 are all at f = -fdhat
                    let fdhat = (approximate_sidereal(frequency_bin).to_f64().expect("usize to double failed") * frequency_bin.lower) as FloatType;
                    let cis_mfh = (start..end)
                        .map(|x| Complex { re: x as FloatType, im: 0.0 })
                        .collect::<Array1<Complex<FloatType>>>()
                        .mul(Complex::new(0.0, 2.0 * DOUBLE_PI as FloatType * -fdhat))
                        .mapv(Complex::exp);
                    let cis_fh = (start..end)
                        .map(|x| Complex { re: x as FloatType, im: 0.0 })
                        .collect::<Array1<Complex<FloatType>>>()
                        .mul(Complex::new(0.0, 2.0 * DOUBLE_PI as FloatType * fdhat))
                        .mapv(Complex::exp);

                    // muz2 is FT(H6) at f = -fdhat
                    let muz2: Complex<FloatType> = (&cis_mfh).mul(&h6).mul(muz_prefactor).sum();
                    // muz3 is FT(-H7) at f = -fdhat
                    let muz3: Complex<FloatType> = -(&cis_mfh).mul(&h7).mul(muz_prefactor).sum();
                    // muz4 is -FT(H3-1) at f = -fdhat (notice negative)
                    let muz4: Complex<FloatType> = -(&cis_mfh).mul(&h3.sub(1.0)).mul(muz_prefactor).sum();

                    // Note: These terms need the - along with prefactor for the f32 -> Complex<f32> into() to work
                    // muz7 is FT(H6) at f = 0
                    let muz7: Complex<FloatType> = h6.mul(muz_prefactor).sum().into();
                    // muz8 is FT(-H7) at f = 0
                    let muz8: Complex<FloatType> = h7.mul(-muz_prefactor).sum().into();
                    // muz9 is -FT(H3-1) at f = 0 (notice negative in front of FT)
                    let muz9: Complex<FloatType> = h3.sub(1.0).mul(-muz_prefactor).sum().into();

                    // muz12 is FT(H6) at f = fdhat
                    let muz12: Complex<FloatType> = (&cis_fh).mul(&h6).mul(muz_prefactor).sum();
                    // muz13 is FT(-H7) at f = fdhat
                    let muz13: Complex<FloatType> = -(&cis_fh).mul(&h7).mul(muz_prefactor).sum();
                    // muz14 is -FT(H3-1) at f = fdhat (notice negative in front of FT)
                    let muz14: Complex<FloatType> = -(&cis_fh).mul(&h3.sub(1.0)).mul(muz_prefactor).sum();

                    let chunk_mu = DarkPhotonMu {
                        x: [mux0, mux1, mux2, mux3, mux4, mux5, mux6, mux7, mux8, mux9, mux10, mux11, mux12, mux13, mux14],
                        y: [muy0, muy1, muy2, muy3, muy4, muy5, muy6, muy7, muy8, muy9, muy10, muy11, muy12, muy13, muy14],
                        z: [muz0, muz1, muz2, muz3, muz4, muz5, muz6, muz7, muz8, muz9, muz10, muz11, muz12, muz13, muz14],
                    };

                    // Insert chunk mu into chunk map
                    inner_chunk_map.insert(
                        chunk,
                        chunk_mu,
                    );
                });
                
                // Insert chunk map into coherence time map
                self.theory_mean.put(coherence_time.to_le_bytes(),
                    bincode::serialize(&inner_chunk_map).unwrap(),
                ).expect("rocksdb failure");
            });
    }


    fn calculate_theory_var(
        &self,
        set: &Vec<(usize, FrequencyBin)>,
        projections_complete: &ProjectionsComplete,
        coherence_times: usize,
        stationarity: Stationarity,
        #[allow(unused)]
        auxiliary_values: Arc<Self::AuxiliaryValue>,
    ) {

        // Clear database
        const CLEAR_VAR_DB: bool = true;
        if CLEAR_VAR_DB {
            self.theory_var.clear();
            println!("cleared var db");
        }

        // Map of Map of Vars
        // key is stationary time chunk (e.g. year)
        let spectra = DashMap::with_capacity(coherence_times);

        // Initialize progress bar
        let initial_spectra_pb = ProgressBar::new(2021-2003);
        log::info!("Calculating power spectra for stationarity times");

        // NOTE: This is hardcoded for stationarity = 1 year
        // NOTE: These are the ones that 
        // TODO: revert. This was used for extracting only a single year/tau
        // let test_year = 2009;
        // std::iter::once(test_year).for_each(|year| {

        (2003..2021).into_iter().for_each(|year| {

            // Get stationarity period indices (place within entire SUPERMAG dataset)
            // MINOR NOTE: this definition varies from original implementation. The original
            // python implementation defines the `end` index to be the first index of the
            // next chunk, since start:end is not end inclusive. This means the size of 
            // the chunks are (end - start + 1)
            let (start_stationarity, end_stationarity) = stationarity.get_year_second_indices(year);
            
            // Now convert these indices to the indices within the subset used
            let secs: Range<usize> = projections_complete.secs();
            let (start_in_series, end_in_series) = (
                secs.clone().position(|i| i == start_stationarity).unwrap(),//.unwrap_or(secs.start),
                secs.clone().position(|i| i == end_stationarity).unwrap(),//.unwrap_or_else(|| { log::warn!("falling back to final element!"); secs.end-1}),
            );
            assert!(start_in_series < end_in_series, "invalid range");

            // Get the subseries for this year
            let projections_subset: DashMap<NonzeroElement, Vec<FloatType>> = projections_complete
                .iter()
                .map(|kv| {
                    // Get element and the complete longest contiguous series
                    let (element, complete_series) = kv.pair();
                    let pair = (element.clone(), complete_series.slice(s![start_in_series..=end_in_series]).to_vec());
                    pair
                }).collect();

            // Set up an fft planner to reuse for all ffts in this year
            let mut planner = FftPlanner::new();
            let fft_handler = planner.plan_fft_forward(2*TAU);

            // Get chunk indices
            let num_chunks: usize = ((end_in_series - start_in_series + 1) / TAU).max(1);
            let chunk_size: usize = (end_in_series - start_in_series + 1) / num_chunks;
            let chunk_mod: usize = (end_in_series - start_in_series + 1) % num_chunks;
            let stationarity_chunks: Vec<[usize; 2]> = (0..num_chunks)
                .map(|k| {
                    // lo = k * size + min(k, mod)
                    let lo = k * chunk_size + k.min(chunk_mod);
                    // hi = (k + 1) * size + min(k + 1, mod)   (exclusive end index)
                    let hi = (k + 1) * chunk_size + (k + 1).min(chunk_mod);
                    [lo, hi]
                }).collect_vec();
        
            let chunk_collection: Vec<_> = stationarity_chunks.into_par_iter().map(|stationarity_chunk| {
                // Get the data for this chunk, pad it to have length equal to a power of 2, and take its fft
                let chunk_ffts: Vec<(NonzeroElement, Power<FloatType>)> = projections_subset
                    .iter()
                    .map(|kv| {

                        // Get element and series
                        let (element, year_series) = kv.pair();

                        // Get chunk from year
                        let chunk_from_year = &year_series[stationarity_chunk[0]..stationarity_chunk[1]];

                        // Padded series
                        let chunk_size = stationarity_chunk[1] - stationarity_chunk[0];
                        let mut padded_series = Array1::<Complex<FloatType>>::zeros(2*TAU);
                        log::trace!(
                            "theory var: size of chunk is {}, being placed in a {} series of zeros",
                            chunk_size, padded_series.len(),
                        );

                        padded_series
                            .slice_mut(s![0..chunk_size])
                            .iter_mut()
                            .zip(chunk_from_year)
                            .for_each(|(ps, s)| { *ps = s.into(); });


                        // FFT of padded series
                        fft_handler
                            .process(padded_series.as_slice_mut().expect("should be in contiguous order"));

                        // Package power with metadata
                        let power = Power {
                            power: padded_series,
                            start_sec: start_in_series,
                            end_sec: end_in_series,
                        };

                        (element.clone(), power)
                    }).collect();

                // Initialize dashmap for correlation
                let chunk_ffts_squared: DashMap<(NonzeroElement, NonzeroElement), Power<FloatType>> = DashMap::new();
                for (e1, fft1) in chunk_ffts.iter() {
                    // Check ffts
                    assert!(fft1.power.iter().all(|x| !x.is_nan()), "fft has nan");
                    for (e2, fft2) in chunk_ffts.iter() {

                        chunk_ffts_squared.insert(
                            (e1.clone(), e2.clone()),
                             // NOTE: in this scope, fft1 and fft2 should have the same start/end
                             // so its okay to use fft2.power and discard start/end and inherit
                             // start/end from fft1
                             (fft1.mul(&fft2.power.map(Complex::conj)))
                                .mul_scalar(((stationarity_chunk[1] - stationarity_chunk[0]) as FloatType).recip())
                        );
                    }
                }

                chunk_ffts_squared
            }).collect();

            // Take the average. 
            // First, initialize zero arrays 
            let avg_power: DashMap<(NonzeroElement, NonzeroElement), Power<FloatType>> = DARK_PHOTON_NONZERO_ELEMENTS
                .iter()
                .cartesian_product(DARK_PHOTON_NONZERO_ELEMENTS.iter())
                .map(|(e1, e2)| ((*e1, *e2), Power { power: Array1::zeros(2*TAU), start_sec: start_in_series, end_sec: end_in_series }))
                .collect();
            // Then, sum (get denominator before consuming iterator)
            let denominator = chunk_collection.len() as FloatType;
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
            
            // TODO revert: this roughly checks scaled vs interp power
            if year == 2009 {            
                let test_coherence_time = 10_198_740;
                let mut freqs: Option<Vec<FloatType>> = None;
                let test_power: HashMap<_, _> = avg_power.iter().flat_map(|kv| {
                    let ((e1, e2), p) = kv.pair();
                    if e1 == e2 {
                        if freqs.is_none() {
                            freqs = Some(p.frequencies())
                        }
                        Some(((*e1, *e2), (&p.power) * test_coherence_time as f32))
                    } else {
                        None
                    }
                }).collect();
                for ((e,_), p) in &test_power {
                    let i = DARK_PHOTON_NONZERO_ELEMENTS.iter().position(|dpe| *e == *dpe).unwrap();
                    let mut file = std::fs::File::create(format!("test_power_{i}")).unwrap();
                    for p_ in p {
                        let p_ = p_.re;
                        file.write(format!("{p_}\n").as_ref()).unwrap();
                    }
                }

                let closest_freq_idx_to_0p1 = freqs.as_ref().unwrap().iter().position_min_by(|a, b| {
                    let adist = (*a-0.1).abs();
                    let bdist = (*b-0.1).abs();
                    adist.partial_cmp(&bdist).unwrap()
                }).unwrap();

                println!("scaled power closest to 0.1hz ({}) = {}", freqs.as_ref().unwrap()[closest_freq_idx_to_0p1], test_power[&(DARK_PHOTON_NONZERO_ELEMENTS[0], DARK_PHOTON_NONZERO_ELEMENTS[0])][closest_freq_idx_to_0p1])
            }
            // let chunk_window_map = self
            //         .theory_var
            //         .get_chunk_window_map(test_coherence_time)
            //         .expect("failed to get window map")
            //         .expect("window map not present in db");
            // let test_chunk = 20;
            // for i in 0..5 {
            //     let mut file = std::fs::File::create(format!("v{i}{i}")).unwrap();
            //     for window in 0..59126 {
            //         let p_ = chunk_window_map.get(&(test_chunk, window)).unwrap().mid.diag()[i].re;
            //         file.write(format!("{p_}\n").as_ref()).unwrap();
            //     }
            // }
            // assert!(false);
            
            // Add along with the rest of the stationarity times
            spectra.insert(year, avg_power);
            initial_spectra_pb.inc(1)
        });

        // Initialize progress bar
        let interpolators_pb = ProgressBar::new(spectra.len() as u64);
        log::info!("Generating interpolators");

        // TODO: debug, this was used to extract a single year/taus
        // let kv_test = spectra.get(&test_year).unwrap();
        // let power_for_test = kv_test.value();
        // let test_db = rocksdb::DB::open_default("test_power").unwrap();
        // // 2013 had no year key
        // // test_db.put(TAU.to_le_bytes(), bincode::serialize(&power_for_test).unwrap()).unwrap();
        // let [t0, t1, t2, t3, t4, t5, t6, t7] = TAU.to_le_bytes();
        // let key = [t0, t1, t2, t3, t4, t5, t6, t7, (test_year-2000) as u8];
        // test_db.put(key, bincode::serialize(&power_for_test).unwrap()).unwrap();
        // assert!(false);
        
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
                                let power_frequencies: Vec<FloatType> = power.frequencies();
                                let power_vec = power.power.to_vec();
                                (*element_pair, Interp1d::new_sorted(power_frequencies, power_vec).expect("failed to construct power interpolator"))
                            }).collect();
                        interpolators_pb.inc(1);
                        result
                    }
                )
            }).collect();
        
        // Initialize progress bar
        let stitch_pb = ProgressBar::new(set.len() as u64);
        log::debug!("Stitching spectra; there are {coherence_times} coherence times");
        let disk_db = &self.theory_var;

        // Stitch spectra together according to coherence times
        let triplet_counter = AtomicU32::new(0);
        set
            .into_iter()
            .for_each(|(coherence_time, frequency_bin)| {

                // First get the domain for the data used
                let secs = projections_complete.secs();
                let len_data = secs.len();

                // Then, get each of the coherence chunks
                let num_chunks = len_data / coherence_time;
                let last_chunk: bool = len_data % coherence_time > 0;
                let total_chunks = num_chunks + if last_chunk { 1 } else { 0 };
                log::info!("theory var: coherence_time {coherence_time} has {total_chunks} chunks");
                
                // This is the inner chunk map from chunk to coherence time
                // let inner_chunk_map: DashMap<(NonzeroElement, NonzeroElement), DashMap<usize, Triplet<f32>>> = DashMap::new();
                let inner_chunk_window_map: InnerVarChunkWindowMap = DashMap::new();

                (0..total_chunks)
                    .into_par_iter()
                    .for_each(|chunk| {

                    // Beginning and end index for this chunk in the total series
                    let chunk_start: usize  = chunk * coherence_time;
                    let chunk_end_exclusive: usize = (chunk + 1) * coherence_time;

                    let overlap_counter = AtomicUsize::new(0);

                    // Check all spectra (over all stationarity periods) for any overlap
                    spectra
                        .par_iter()
                        // .iter()
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

                                    // The element indices start at 1 so subtract 1
                                    // i.e. (X1, X2, X3, X4, X5) -> (0, 1, 2, 3, 4)
                                    let (ix, iy) = (e1.index-1, e2.index-1);

                                    // Get start and end seconds for this power
                                    let (power_start, power_end) = (power.start_sec, power.end_sec);

                                    // Calculate overlap (in number of seconds)
                                    // as that is what the function assumes.
                                    let chunk_end_inclusive = chunk_end_exclusive - 1;
                                    let overlap: usize = calculate_overlap(
                                        chunk_start,
                                        chunk_end_inclusive,
                                        power_start,
                                        power_end
                                    );

                                    // Add contribution to chunk if there is overlap
                                    if overlap > 0 {
                                        log::debug!("detected overlap in chunk {chunk} of magnitude {overlap} of {coherence_time}");

                                        overlap_counter.fetch_add(overlap, Ordering::Relaxed);

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

                                        // Interpolate power to appropriate frequencies
                                        let frequencies_to_interpolate_to: Vec<FloatType> = relevant_range
                                            .map(|i| i as FloatType * frequency_bin.lower as FloatType)
                                            .collect();
                                        // TODO: revert checking particular chunk
                                        if (*coherence_time == 10_198_740) & (chunk == 20) & (e1 == e2) & (*e1 == DARK_PHOTON_NONZERO_ELEMENTS[0]) {
                                            println!("interpolating frequencies: {}..{}", frequencies_to_interpolate_to[0], frequencies_to_interpolate_to.last().unwrap());
                                        }
                                        let interpolated_power: Array1<Complex<FloatType>> = frequencies_to_interpolate_to
                                            .iter()
                                            .flat_map(|f| {
                                                if *f <= MAX_FREQUENCY {
                                                    Some(power_interpolators
                                                        .get(&stationarity_time)
                                                        .expect("interpolator should exist for this stationarity time")
                                                        .get(element_pair)
                                                        .expect("interpolator should exist for this element pair")
                                                        .interpolate_checked(*f)
                                                        .unwrap(/* unwrapping for now if out of bounds */))
                                                } else {
                                                    None
                                                }
                                            }).collect();

                                        let num_windows = interpolated_power.len() - (2*approx_sidereal + 1) + 1;
                                        log::info!("theory var: coherence_time {coherence_time} chunk {chunk} has {num_windows} windows");

                                        // Get all relevant triplets and multiply them by their overlap weight
                                        interpolated_power
                                            .axis_windows(ndarray::Axis(0), 2*approx_sidereal + 1)
                                            .into_iter()
                                            .enumerate()
                                            .par_bridge()
                                            .for_each(|(window_index, window)| {

                                                // Get triplet
                                                let low: Complex<FloatType> = window[0_usize].mul(overlap as FloatType);
                                                let mid: Complex<FloatType> = window[approx_sidereal].mul(overlap as FloatType);
                                                let high: Complex<FloatType> =  window[2*approx_sidereal].mul(overlap as FloatType);

                                                // TODO: revert checking particular coh/chunk
                                                if (*coherence_time == 10_198_740) & (chunk == 20) & (window_index == 0) & (e1 == e2) & (*e1 == DARK_PHOTON_NONZERO_ELEMENTS[0]) {
                                                    println!("test overlap for 10_198_740, 20, 0 = {overlap} x {} = {}", window[approx_sidereal], mid);
                                                }

                                                // get frequencies
                                                // let lowf: f32 = frequencies_to_interpolate_to[window_index];
                                                let midf: FloatType = frequencies_to_interpolate_to[window_index + approx_sidereal];
                                                // let hif: f32 = frequencies_to_interpolate_to[window_index + 2*approx_sidereal];
                                                
                                                // Add/store triplet
                                                inner_chunk_window_map
                                                    .entry((chunk, window_index))
                                                    .and_modify(|triplet: &mut Triplet| {

                                                        // First, add to low triplet matrix (fa-fdhat)
                                                        triplet.low[[ix, iy]] += low;

                                                        // Then, add to mid triplet matrix (fa)
                                                        triplet.mid[[ix, iy]] += mid;

                                                        // Finally, add to hi triplet matrix (fa+fdhat)
                                                        triplet.high[[ix, iy]] += high;
                                                    }).or_insert_with(|| {

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
                                                            midf: midf,
                                                        }
                                                    });
                                            });
                                    }
                            });
                        });

                    log::debug!("expected total overlap {coherence_time}, got {}", overlap_counter.load(Ordering::Relaxed));
                });

                // now we are just inserting into db
                log::debug!("storing coh {coherence_time} map, with {} elements", inner_chunk_window_map.len());

                // TODO revert: checking power 
                if *coherence_time == 10_198_740 {
                    let test_chunk = 20;
                    for i in 0..5 {
                        let mut file = std::fs::File::create(format!("v{i}{i}")).unwrap();
                        for window in 0..59126 {
                            let p_ = inner_chunk_window_map.get(&(test_chunk, window)).unwrap().mid.diag()[i].re;
                            file.write(format!("{p_}\n").as_ref()).unwrap();
                        }
                    }
                }

                inner_chunk_window_map.into_iter().for_each(|((chunk, window), triplet)| {
                    disk_db.store_chunk_window_map(*coherence_time, chunk, window, &triplet).unwrap();
                });

                // log::info!("Finished calculating power for coherence_time {coherence_time}");
                stitch_pb.inc(1)
            });

        let triplet_count = triplet_counter.load(std::sync::atomic::Ordering::Acquire);
        log::debug!("initialized {} triplets -> {} total arrays", triplet_count, 3 * triplet_count);
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
        #[allow(unused)]
        coherence_times: usize,
        #[allow(unused)]
        stationarity: Stationarity,
    ) -> Vec<(FloatType, FloatType)> {

        // Get number of seconds in the continguous subset of dataset
        let num_secs = projections_complete.num_secs();
        let zdb = rocksdb::DBWithThreadMode::<MultiThreaded>::open_default("z").unwrap();
        let sdb = rocksdb::DBWithThreadMode::<MultiThreaded>::open_default("s").unwrap();
        
        let sz_coherence: DashMap<_, _> = set
            .into_iter()
            // TODO: revert
            // .filter(|(coh, _)| *coh == 10198740)
            .map(|(coherence_time, frequency_bin)| {

                // Get number of chunks for this coherence time
                let num_chunks = num_secs / coherence_time;
                let last_chunk: bool = num_secs % coherence_time > 0;
                let total_chunks = num_chunks + if last_chunk { 1 } else { 0 };
                log::info!("likelihood: coherence_time {coherence_time} has {total_chunks} chunks");

                // Get theory mean for this coherence time
                let theory_mean_ct: DashMap<usize, DarkPhotonMu> = bincode::deserialize(
                    &self.theory_mean
                        .get(&coherence_time.to_le_bytes())
                        .expect("rocks db failure")
                        .expect("theory mean should exist for every coherence time here")
                ).unwrap();

                // For this coherence time, get the map to every triplet window (for sigma)
                let chunk_window_map = self
                    .theory_var
                    .get_chunk_window_map(*coherence_time)
                    .expect("failed to get window map")
                    .expect("window map not present in db");

                // // TODO: revert
                if *coherence_time == 10198740 {
                    let test_chunk = 20;
                    let averaged_xii2: Array1<FloatType> = (0..59126)
                        .map(|window| {
                            let dv_key = [*coherence_time, test_chunk, window].map(usize::to_le_bytes).concat();
                            let data_vector_window: DarkPhotonVec<FloatType> = bincode::deserialize(
                                &self.data_vector
                                .get(&dv_key)
                                            .expect("rocks db failure")
                                            .expect("data vector should exist for every window")
                                    ).unwrap();
                            data_vector_window.mid.map(|xii| xii.norm_sqr())
                        }).fold(Array1::<FloatType>::zeros(5), |acc, x| acc + x) / 59126.0;
                    let averaged_vii: Array1<FloatType> = (0..59126)
                        .map(|window| {
                            chunk_window_map.get(&(test_chunk, window)).unwrap().mid.diag().map(|v| v.re)
                        }).fold(Array1::<FloatType>::zeros(5), |acc, x| acc + x) / 59126.0;
                    println!("xii2 vs vii is:");
                    println!("    {averaged_xii2}");
                    println!("    {averaged_vii}");
                }




                    // Step 1 and 2: calculate inverse of A_k for all chunks and windows
                    let ainv = chunk_window_map
                        .par_iter()
                        .map(|kv2| {
                            let ((chunk, window), triplet) = kv2.pair();
                            ((*chunk, *window), triplet.block_cholesky().block_inv())
                            }).collect::<DashMap<_,_>>();

                    // Step 3: Calculate Y_k = Ainv_k * X_k for all windows in this chunk
                    let y: DashMap<(Chunk, Window), DarkPhotonVec<FloatType>> = ainv
                        .par_iter()
                        .map(|kv2| {
                            // Get triplet window and corresponding map
                            let ((chunk, window), ainv_triplet) = kv2.pair();
                            
                            // Get data vector for this coh time, chunk, window
                            let dv_key = [*coherence_time, *chunk, *window].map(usize::to_le_bytes).concat();
                            let data_vector_window: DarkPhotonVec<FloatType> = bincode::deserialize(
                                &self.data_vector
                                    .get(&dv_key)
                                    .expect("rocks db failure")
                                    .expect("data vector should exist for every window")
                            ).expect("deser failure");

                            // (window, Ainv * Xk_)
                            ((*chunk, *window), ainv_triplet.dot_vec(&data_vector_window))
                            }).collect::<DashMap<(Chunk, Window), DarkPhotonVec<FloatType>>>();
                    // Step 4: Calculate nu_ik = Ainv_k * mu_ik
                    let nu = ainv
                        .par_iter()
                        .map(|kv2| {
                            // Get triplet window and corresponding map
                            let ((chunk, window), ainv_triplet) = kv2.pair();

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
                            let nu: Array2<Complex<FloatType>> = Array2::from_shape_fn((15, 3), |(i, j)| { 
                                match i {
                                    _i if _i < 5 => three_blocks.low[[i, j]],
                                    _i if _i < 10 => three_blocks.mid[[i-5, j]],
                                    _i if _i < 15 => three_blocks.high[[i-10, j]],
                                    _ => unreachable!("should not reach this"),
                                }
                            });
                            assert_eq!(nu.shape(), &[15, 3]);

                            // Return nu with window as the key
                            ((*chunk, *window), nu)
                        }).collect::<DashMap<_, Array2<Complex<FloatType>>>>();

                    // Step 5: Calculate svd of Nik
                    // Step 6: Calculate Zk = Udag_k * Y_k
                    // This gets us s and z
                    let sz_chunk_window_map = nu
                        .into_par_iter()
                        .map(|((chunk, window), nu_window)| {
                            // Step 5: Carry out svd
                            // nu is a 15x3 matrix
                            assert_eq!(nu_window.shape(), &[15, 3]);
                            // u should be 15x3, S should be 3x3, and v should be 3x3.
                            let (Some(u), s, None) = compact_svd(&nu_window, true, false).expect("svd failed") else {
                                panic!("u was requested but was not given, or v was given but was not requested")
                            };
                            assert_eq!(u.shape(), &[15, 3], "svd: u does not have correct shape");
                            assert_eq!(s.shape(), &[3], "svd: s does not have correct shape");
                            // assert_eq!(v.shape(), &[3, 3], "svd: v does not have correct shape");

                            // Conjugate and transpose v and u
                            // let vdag = v.t().map(|c| c.conj());
                            let udag = u.t().map(|c| c.conj());
                            // assert_eq!(vdag.shape(), &[3, 3], "congj t: vdag does not have correct shape");
                            assert_eq!(udag.shape(), &[3, 15], "congj t: udag does not have correct shape");
                            
                            // Step 6: Zk = Udag_k * Y_k
                            let y_k = y.get(&(chunk, window)).expect("yk should exist for this chunk, window").to_vec();
                            assert_eq!(y_k.shape(), &[15]);
                            let z = udag.dot(&y_k);
                            assert_eq!(z.shape(), &[3]);

                            // Save things to disk
                            let z_and_s_key: Vec<u8> = [*coherence_time, chunk, window].map(usize::to_le_bytes).concat();
                            zdb.put(&z_and_s_key, bincode::serialize(&z).unwrap()).unwrap();
                            sdb.put(&z_and_s_key, bincode::serialize(&s).unwrap()).unwrap();

                            ((chunk, window), (s, z))
                        }).collect::<DashMap<_, _>>();

                // Return sz map
                (coherence_time, (frequency_bin, sz_chunk_window_map.into_read_only()))
            }).collect();

        let sz_coherence = sz_coherence.into_read_only();

        let (tx, rx) = bounded::<(f64, Vec<f64>)>(10);
        
        let handle = std::thread::spawn(move || {
            let mut file = std::fs::File::create("pdf.txt").unwrap();
            while let Ok((freq, log10pdf)) = rx.recv() {
                file.write(format!("{freq} ").as_bytes()).unwrap();
                for value in log10pdf {
                    file.write(format!("{value} ").as_bytes()).unwrap();
                }
                file.write(b"\n").unwrap();
            }
        });
        
        let (tx2, rx2) = unbounded::<(f64, BoundError)>();
        let handle2 = std::thread::spawn(move || {
            let mut failed_bounds = std::fs::File::create("failed_freqs.txt").unwrap();
            while let Ok((failed_freq, err)) = rx2.recv() {
                failed_bounds.write(format!("{failed_freq} {err:?}\n").as_bytes()).unwrap();
            }
        });

        let (tx3, rx3) = unbounded::<(usize, usize)>();
        let handle3 = std::thread::spawn(move || {
            let mut missing_chunk_windows = std::fs::File::create("missing_chunk_windows.txt").unwrap();
            while let Ok((chunk, window)) = rx3.recv() {
                missing_chunk_windows.write(format!("{chunk} {window}\n").as_bytes()).unwrap();
            }
        });

        // Use SZ to calculate bounds
        let freqs_and_bounds: Vec<(FloatType, FloatType)> = sz_coherence
            .into_par_iter()
            .map(|(coherence_time, (frequency_bin, sz_chunk_window_map))| {

                // Total number of chunks for this coherence time
                let num_chunks = num_secs / coherence_time;
                let last_chunk: bool = num_secs % coherence_time > 0;
                let total_chunks = num_chunks + if last_chunk { 1 } else { 0 };

                let approx_sidereal: usize = approximate_sidereal(frequency_bin);
                let num_fft_elements = *coherence_time;
                if num_fft_elements < 2*approx_sidereal + 1 {
                    println!("no triplets exist");
                }
                let start_relevant: usize = frequency_bin.multiples.start().saturating_sub(approx_sidereal);
                let end_relevant: usize = (*frequency_bin.multiples.end()+approx_sidereal).min(num_fft_elements-1);
                let relevant_range = start_relevant..=end_relevant;
                log::debug!("coherence_time = {coherence_time}, {start_relevant}..={end_relevant}");

                // Gather windows from relevant range
                let num_windows = num_windows(start_relevant, end_relevant, approx_sidereal);
                assert_eq!(2 * approx_sidereal + num_windows, relevant_range.end()-relevant_range.start() + 1, "failed for coherence time {coherence_time}");
                let relevant_windows = relevant_range.skip(approx_sidereal).take(num_windows);

                // Now that we have all the window indices,
                // for each of them collect all chunks and calculate bound
                let bounds = Arc::new(Mutex::new(Vec::with_capacity(end_relevant-start_relevant+1)));
                relevant_windows
                    .into_iter()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(window_idx, multiple)| {
                        let _tx2 = tx2.clone();

                        // Initialize vec to hold references to sz pairs for this frequency/window
                        let mut sz_references = Vec::<&(Array1<FloatType>, Array1<Complex<FloatType>>)>::with_capacity(num_chunks);
                        for chunk in 0..total_chunks {
                            if let Some(reference) = sz_chunk_window_map
                            .get(&(chunk, window_idx)) {
                                sz_references.push(reference);
                            } else {
                                log::error!("chunk, window = {chunk}, {window_idx} should exist for coh {coherence_time} but doesnt");
                                tx3.send((chunk, window_idx)).unwrap();
                            }
                        }
                        let frequency = multiple as f64 * frequency_bin.lower;

                        // Step 7: calculate likelihood, 95% bound.
                        match bound(frequency, &sz_references, tx.clone()) {
                            Ok(eps_bound) => bounds.lock().unwrap().push((frequency as FloatType, eps_bound)),
                            Err(err) => {
                                _tx2.send((frequency, err)).unwrap();
                            }
                        }
                });

                Arc::try_unwrap(bounds).unwrap().into_inner().unwrap()
            }).flatten().collect();

        drop(tx);
        drop(tx2);
        drop(tx3);
        handle.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();

        println!("calculated {} bounds", freqs_and_bounds.len());

        freqs_and_bounds
    }
}


/// Calculates the bound for a particular (coherence_time, chunk, window)
fn bound(
    // only here to send via tx
    frequency: f64,
    // This collects all z and s that have the same frequency
    sz: &[&(Array1<FloatType>, Array1<Complex<FloatType>>)],
    tx: crossbeam_channel::Sender<(f64, Vec<f64>)>,
) -> std::result::Result<FloatType, BoundError> {

    // The pdf is of the form N * sqrt(sum(...)) * prod(a exp(b))
    // so we will break down logpdf into summands 
    // 1) log N
    // 2) log sqrt(sum(...))
    // 3) log sum(a)
    // 4) log sum(b)
    let logpdf = |norm: f64, eps: f64| -> f64 {
        // Term 1: logarithm of normalization factor
        let term_1: f64 = norm.ln();

        // Term 2: logarithm of square root of sum, from jeffery's prior
        let term_2: f64 = sz.iter().map(|(si, _zi)| {
            si.into_iter().map(|sik| {
                let sik = *sik as f64;
                assert!(sik.is_finite());
                // (4.0 * eps.powi(2) * sik.powi(4)) / (3.0 + eps.powi(2) * sik.powi(2)).powi(2)
                4.0  / (3.0 / (eps.powi(1) * sik.powi(2)) + eps.powi(1)).powi(2)
            }).sum::<f64>()
        }).sum::<f64>().sqrt().ln();


        // Term 3: log sum(a)
        let term_3: f64 = sz.iter().map(|(si, _zi)| {
            si.into_iter().map(|sik| {
                let sik = *sik as f64;
                // - ((3.0 + eps.powi(2) * sik.powi(2)).powi(2)).ln()
                - ((3.0 + eps.powi(2) * sik.powi(2))).ln() * 2.0
            }).sum::<f64>()
        }).sum::<f64>();


        // Term 4: log(a)
        let term_4: f64 = {
            let mut acc = 0.0_f64; 
            for j in 0..sz.len() {
                let (sj, zj) = sz[j];
                for i in 0..3 {
                    acc -= 3.0 * zj[i].norm_sqr() as f64 / (3.0 + (eps as f64).powi(2) * (sj[i] as f64).powi(2));
                }
            }
            acc
        };


        // Add all terms
        assert!(!term_1.is_nan(), "{term_1}");
        assert!(!term_2.is_nan(), "{term_2}");
        assert!(!term_3.is_nan(), "{term_3}");
        assert!(!term_4.is_nan(), "{term_4}");
        term_1 + term_2 + term_3 + term_4
    };

    // Find where logeps is maximum in -10, 1
    // TODO: refactor into DarkPhoton: Theory
    let hi_logeps = -10.0;
    let lo_logeps = 1.0;
    let num_eps = 1000;
    let log10eps_grid: Vec<f64> = (0..num_eps).map(|i| hi_logeps + i as f64 * (lo_logeps-hi_logeps) / num_eps.sub(1) as f64).collect();
    let logp_logeps_grid: Vec<f64> = log10eps_grid
        .iter()
        .map(|&log10eps| {
            let eps = 10_f64.powf(log10eps);
            // see below for more info regarding coordinate transformation
            // log(pdf(logeps)) 
            //  = log(pdf(eps) * eps) 
            //  = log(pdf(eps)) + log(eps)
            logpdf(1.0, eps) + eps.ln()
        }).collect();
    tx.send((frequency, logp_logeps_grid.clone())).unwrap();
    drop(tx);
    let (max_log10eps, max_logp) = log10eps_grid
        .into_iter()
        .zip(&logp_logeps_grid)
        .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap()) // find max logpdf
        .unwrap();
    let max_logeps = max_log10eps / f64::exp(1.0).log10();
    // log::debug!("found max_logp(logeps = {max_logeps:.3e}) = {max_logp:.2e}");
    if max_logp.is_finite() {
        if max_logeps > 0.0 {
            return Err(BoundError::HighMax { max_logeps });
        }
    } else {
        return Err(BoundError::InvalidMax);
    }

    // Transform pdf(x) to pdf(y)  
    // for 
    // x = eps
    // y = log(eps)
    // --> x(y) = exp(y)
    // --> dx(y)/dy = d/dy exp(y) = exp(y) ---> exp(logeps)
    // via
    // pdf(y) = pdf(x(y)) |dx(y)/dy| 
    //        = pdf(exp(logeps)) * exp(logeps) 
    //        = exp(logpdf(exp(logeps))) * exp(logeps) 
    //        = exp(logpdf(exp(logeps)) + logeps)
    //
    // To add a rough normalization, 
    //        = exp(logpdf(exp(logeps)) - lognorm + logeps)
    //
    // The integration library used below expects double precision
    let transformed_unnormalized_pdf = |logeps: f64| {
        let eps = logeps.exp();
        let logpdf_ = logpdf(1.0, eps);
        (logpdf_ - max_logp + logeps).exp()
    };

    let function = |upper: f64| {

        // Try clenshaw curtis
        let mut result = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, 1e-10.ln(), upper, 1e-4).integral;

        // Fallback if invalid
        if !result.is_normal() {
            let mut num_steps = 1000;
            while result.is_normal() {
                let hi_logeps = -10.0;
                let lo_logeps = upper;
                let dlogeps = (lo_logeps-hi_logeps) / num_steps.sub(1) as f64;
                let log10eps_grid = (0..num_steps).map(|i| hi_logeps + i as f64 * dlogeps);
                let p_grid = log10eps_grid
                    .map(|log10eps| {
                        let eps = 10_f64.powf(log10eps);
                        logpdf(1.0, eps)
                    });
                result = p_grid.tuple_windows().map(|(left_p, right_p)| {
                    (left_p + right_p) * dlogeps / 2.0
                }).sum();

                num_steps *= 10;
            }
            // One final, higher resolution integration
            log::info!("failed clenshaw-curtis. using fallback value with {num_steps} steps");
            let hi_logeps = -10.0;
            let lo_logeps = upper;
            let dlogeps = (lo_logeps-hi_logeps) / num_steps.sub(1) as f64;
            let log10eps_grid = (0..num_steps).map(|i| hi_logeps + i as f64 * dlogeps);
            let p_grid = log10eps_grid
                .map(|log10eps| {
                    let eps = 10_f64.powf(log10eps);
                    logpdf(1.0, eps)
                });
            result = p_grid.tuple_windows().map(|(left_p, right_p)| {
                (left_p + right_p) * dlogeps / 2.0
            }).sum();
        }

        if !result.is_sign_positive() || !result.is_normal() {
            return Err(BoundError::IntegralUnderflow)
        }

        Ok(result)
    };

    // Find 95% confidence interval
    let mut upper_bound_logeps = max_logeps;
    #[allow(unused_mut)]
    let mut norm: f64 = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, 1e-10.ln(), 1e1.ln(), 1e-4).integral;
    if !norm.is_normal() {
        return Err(BoundError::InvalidNorm { max_logeps })
    }
    
    let initial_delta = 0.1;
    let tol = 1e-4;
    let target = 0.95;
    let max = 1000;
    optimize(function, &mut upper_bound_logeps, initial_delta, target, norm, tol, max)?;

    // Upper bound is for logeps so exponentiate to get exp bound
    Ok(upper_bound_logeps.exp() as FloatType)
}

fn optimize<F: Fn(f64) -> std::result::Result<f64, BoundError>>(function: F, input: &mut f64, mut delta: f64, target: f64, norm: f64, tol: f64, max: u64) -> std::result::Result<u64, BoundError> {
    const ABOVE: u8 = 1;
    const BELOW: u8 = 2;
    let mut last_status = 0u8;
    let mut num_steps = 0;
    loop {
        let current: f64 = function(*input)?;
        assert!(current.is_finite(), "current is not finite f({input}) = {current}");
        if current == 0.0 {
            return Err(BoundError::IntegralUnderflow)
        }
        let i = current / norm;
        if (i-target).abs() < tol {
            // If within tolerance we are converged
            log::debug!("converged {i} vs {target} with input {input} (within tolerance {tol}) in {num_steps} steps (delta = {delta})");
            break;
        } 

        num_steps += 1;
        if i > target {
            // Else, if above target decrease upper_bound
            // First check if we crossed target
            if last_status == BELOW {
                // we have skipped over target, so reduce delta
                delta /= 10.0;

            }
            *input -= delta;
            last_status = ABOVE;
        } else if i < target {
            // Else, if below target increase upper_bound
            // First check if we crossed target
            if last_status == ABOVE {
                // we have skipped over target, so reduce delta
                delta /= 10.0;

            }
            *input += delta;
            last_status = BELOW;
        }

        log::trace!("step {num_steps}: input = {input}, {i} vs {target}, delta = {delta}");

        if num_steps > max {
            log::debug!("breaking with {} ({} < {}, delta = {})", input, (i-target).abs(), tol, delta);
            return Err(BoundError::DidNotConverge)
        }
    }
    Ok(num_steps)
}

pub fn read_dark_photon_projections_auxiliary() -> Result<
    (
        Arc<ProjectionsComplete>,
        Arc<<DarkPhoton as Theory>::AuxiliaryValue>,
    ),
    Box<dyn std::error::Error>,
> {
    // Open projections_complete and auxiliary_complete file
    let mut projections_file = std::fs::File::open("projections_complete").expect("failed to open file");
    let mut auxiliary_file = std::fs::File::open("auxiliary_complete").expect("failed to open file");

    // Initialize buffer for projections and auxiliary values
    let mut projection_buffer = Vec::new();
    let mut auxiliary_buffer = Vec::new();

    // Read bytes in files
    projections_file
        .read_to_end(&mut projection_buffer)
        .expect("failed to read projections");
    auxiliary_file
        .read_to_end(&mut auxiliary_buffer)
        .expect("failed to read auxiliary");

    // Deserialize bytes into respective types
    let projections_complete: Arc<ProjectionsComplete> = Arc::new(
        bincode::deserialize(&projection_buffer)
            .expect("failed to deserialize projections_complete"),
    );
    let auxiliary_complete: Arc<<DarkPhoton as Theory>::AuxiliaryValue> = Arc::new(
        bincode::deserialize(&auxiliary_buffer)
            .expect("failed to deserialize projections_complete"),
    );

    Ok((projections_complete, auxiliary_complete))
}


#[derive(Debug)]
pub enum BoundError {
    DidNotConverge,
    InvalidMax,
    IntegralUnderflow,
    InvalidNorm { max_logeps: f64 },
    HighMax { max_logeps: f64 },
}

#[test]
fn test_constant_on_01() {
    use approx_eq::assert_approx_eq;

    // integral of 1 from 0 to u = u
    let function = |u: f64| Ok(u);
    let target = 0.2;
    let tol = 1e-6;
    let norm = function(1.0).unwrap();
    let initial_delta = 1.0;
    let mut upper_bound = 0.1;
    let max = 100;
    optimize(function, &mut upper_bound, initial_delta, target, norm, tol, max).unwrap();

    assert_approx_eq!(upper_bound, target, tol);
}


#[derive(Default, Serialize, Deserialize)]
pub struct DarkPhotonMu {
    // #[serde(with = "ComplexDef")]
    pub x: [Complex<FloatType>; 15],
    // #[serde(with = "ComplexDef")]
    pub y: [Complex<FloatType>; 15],
    // #[serde(with = "ComplexDef")]
    pub z: [Complex<FloatType>; 15],
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
    block1: Array2<Complex<FloatType>>,
    block2: Array2<Complex<FloatType>>,
    block3: Array2<Complex<FloatType>>,
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

    fn scale(&self, factor: FloatType) -> Self {
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
        x: [1.0 as FloatType, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| x.into()),
        y: [1.0 as FloatType, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x| (-x).into()),
        z: [1.0 as FloatType, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0].map(|x|  I * x),
    };
    let block1: Array2<Complex<FloatType>> = array![
        [1.0 as FloatType * ONE, 2.0 * ONE, 3.0 * ONE, 4.0 * ONE, 5.0 * ONE],
        [-1.0 as FloatType * ONE, -2.0 * ONE, -3.0 * ONE, -4.0 * ONE, -5.0 * ONE],
        [1.0 as FloatType * I, 2.0 * I, 3.0 * I, 4.0 * I, 5.0 * I]
    ].t().to_owned();
    let block2: Array2<Complex<FloatType>> = array![
        [6.0 * ONE, 7.0 * ONE, 8.0 * ONE, 9.0 * ONE, 10.0 * ONE],
        [-6.0 * ONE, -7.0 * ONE, -8.0 * ONE, -9.0 * ONE, -10.0 * ONE],
        [6.0 * I, 7.0 * I, 8.0 * I, 9.0 * I, 10.0 * I]
    ].t().to_owned();
    let block3: Array2<Complex<FloatType>> = array![
        [11.0 * ONE, 12.0 * ONE, 13.0 * ONE, 14.0 * ONE, 15.0 * ONE],
        [-11.0 * ONE, -12.0 * ONE, -13.0 * ONE, -14.0 * ONE, -15.0 * ONE],
        [11.0 * I, 12.0 * I, 13.0 * I, 14.0 * I, 15.0 * I]
    ].t().to_owned();
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
    (end1.min(end2)+1)
        .checked_sub(start1.max(start2))
        .unwrap_or(0)
}

#[test]
fn test_overlap() {
    assert_eq!(calculate_overlap(0, 4, 0, 4), 5);
    assert_eq!(calculate_overlap(0, 4, 3, 4), 2);
    assert_eq!(calculate_overlap(3, 4, 0, 4), 2);
    assert_eq!(calculate_overlap(3, 4, 4, 10), 1);
    assert_eq!(calculate_overlap(3, 4, 5, 10), 0);
}

/// This function takes in the weights w_i along with the station coordinates and calculates H_i(t)
/// This doesn't necessarily need to be parallelized because this is done per coherence chunk, which is parallelized.
/// Thus, no further delegation is necessary (likely).
fn dark_photon_auxiliary_values(
    weights_n: &DashMap<StationName, FloatType>,
    weights_e: &DashMap<StationName, FloatType>,
    weights_wn: &TimeSeries,
    weights_we: &TimeSeries,
    valid_entry_map: &DashMap<StationName, Array1<bool>>,
    // chunk_dataset: &DashMap<StationName, Dataset>,
) -> [TimeSeries; 7] {

    // Gather coordinate table
    let coordinates = construct_coordinate_map();

    let auxiliary_values = [1, 2, 3, 4, 5, 6, 7].into_par_iter().map(|i| {

        log::trace!("calculating auxiliary value H{i}");

        // Here we iterate thrhough weights_n and not chunk_dataset because
        // stations in weight_n are a subset (filtered) of those in chunk_dataset.
        // Could perhaps save memory by dropping coressponding invalid datasets in chunk_dataset.
        let mut auxiliary_value_series_unnormalized: TimeSeries = Array1::zeros(weights_wn.len());

        for key_value in weights_n.iter() {
            // .map(|key_value| -> Array1<f32> { {

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
                1 => (sc.longitude.cos().powi(2) as FloatType).mul(*weights_n.get(station_name).unwrap()),

                // H2 summand = wn * sin(phi) * cos(phi)
                2 => ((sc.longitude.sin() * sc.longitude.cos()) as FloatType).mul(*weights_n.get(station_name).unwrap()),

                // H3 summand = we * cos(polar)^2
                3 => (sc.polar.cos().powi(2) as FloatType).mul(*weights_e.get(station_name).unwrap()),

                // H4 summand = we * cos(phi)^2 * cos(polar)^2
                4 => ((sc.longitude.cos().powi(2) * sc.polar.cos().powi(2)) as FloatType).mul(*weights_n.get(station_name).unwrap()),

                // H5 summand = we * sin(phi) * cos(phi) * cos(polar)^2
                5 => ((sc.longitude.sin() * sc.longitude.cos() * sc.polar.cos().powi(2)) as FloatType)
                    .mul(*weights_n.get(station_name).unwrap()),

                // H6 summand = we * cos(phi) * sin(polar) * cos(polar)
                6 => ((sc.longitude.cos() * sc.polar.sin() * sc.polar.cos()) as FloatType).mul(*weights_n.get(station_name).unwrap()),

                // H7 summand = we * sin(phi) * sin(polar) * cos(polar)
                7 => ((sc.longitude.sin() * sc.polar.sin() * sc.polar.cos()) as FloatType).mul(*weights_n.get(station_name).unwrap()),
                
                _ => unreachable!("hardcoded to iterate from 1 to 7"),
            };

            let station_series: Array1<FloatType> = (*valid_entry_map.get(station_name).expect("station should exist")).map(|flag| if *flag { 1.0 } else { 0.0 }) * auxiliary_value;
            auxiliary_value_series_unnormalized.add_assign(&station_series);
        }

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
    }).collect::<Vec<_>>().try_into().unwrap();

    auxiliary_values
}

fn num_windows(
    start_padded: usize,
    // end_padded is inclusive
    end_padded: usize,
    approximate_sidereal: usize
) -> usize {
    // end_padded is inclusive so we must add 1
    let size = (end_padded - start_padded) + 1;
    let window_size = (2*approximate_sidereal) + 1;
    size - window_size + 1
}

#[test]
fn test_num_windows() {
    assert_eq!(num_windows(0, 4, 1), 3);
    assert_eq!(num_windows(0, 9, 1), 8);
    assert_eq!(num_windows(0, 9, 2), 6);
}

#[derive(Serialize, Deserialize)]
pub struct Power<T: Num + Clone> {
    pub power: Array1<Complex<T>>,
    start_sec: usize,
    end_sec: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
/// Note: these should all be 5x5 arrays
pub struct Triplet {
    pub low: Array2<Complex<FloatType>>,
    pub mid: Array2<Complex<FloatType>>,
    pub high: Array2<Complex<FloatType>>,
    pub midf: FloatType,
}

impl Triplet {
    fn mat_mul(&self, arrays: [Array2<Complex<FloatType>>; 3]) -> Triplet {
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

    /// This takes the [Triplet] self and multipled a [DarkPhotonVec] to produce another [DarkPhotonVec]
    /// via matrix multiplication, i.e. A.x = y
    fn dot_vec(&self, vecs: &DarkPhotonVec<FloatType>) -> DarkPhotonVec<FloatType> {
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
        let Ok(low) = self.low.inv() else {
            panic!("failed matrix inversion {:?}", self.low)
        };
        let Ok(mid) = self.mid.inv() else {
            panic!("failed matrix inversion {:?}", self.mid)
        };
        let Ok(high) = self.high.inv() else {
            panic!("failed matrix inversion {:?}", self.high)
        };
        Triplet {
            low,
            mid,
            high,
            midf: self.midf,
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
            midf: self.midf,
        
        }
    }
}

impl<T: Float> Power<T> 
{
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
    fn frequencies(&self) ->  Vec<FloatType> {
        get_frequency_range_1s(2*TAU)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DarkPhotonAuxiliary {
    pub h: [TimeSeries; 7]
}

impl FromChunkMap for DarkPhotonAuxiliary {
    fn from_chunk_map(
        auxiliary_values_chunked: &DashMap<usize, Self>,
        stationarity: Stationarity,
        starting_value: usize,
        size: usize,
    ) -> Self {

        // Initialize auxiliary array to zeros
        let (start_first, _) = stationarity.get_year_second_indices(starting_value);
        let (_, end_last) = stationarity.get_year_second_indices(starting_value + size - 1);
        let auxiliary_values = [(); 7]
            .map(|_| Mutex::new(TimeSeries::zeros(end_last - start_first + 1)));

        let mut values_assigned: usize = 0;
        for chunk in starting_value..(starting_value+size) {
            let (_, auxiliary) = auxiliary_values_chunked.remove(&chunk).unwrap();

            let start_index = values_assigned;
            let end_index = values_assigned + auxiliary.h[0].len();
            values_assigned += auxiliary.h[0].len();

            (0..7).into_par_iter().for_each(|i| {
                let mut series = auxiliary_values
                    .get(i)
                    .unwrap()
                    .lock()
                    .unwrap();

                println!("H{} start {start_index}, end {end_index}, last {}", i+1, series.len()-1);

                series
                    .slice_mut(s![start_index..end_index])
                    .assign(&auxiliary.h[i]);
            });
        }

        DarkPhotonAuxiliary { h: auxiliary_values.map(Mutex::into_inner).map(|h| h.unwrap()) }
    }
}