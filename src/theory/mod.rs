/// This module contains both the trait `Theory` required for a theory,
/// e.g. a dark photon, but also the functions that produce expected signals.

pub mod dark_photon;

use sphrs::{ComplexSHType, Coordinates as SphrsCoordinates, SHEval};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::fmt::Debug;
use num_complex::Complex;
use ndarray::Array1;
use special::Gamma;
use crate::{
    utils::{
        coordinates::construct_coordinate_map,
        loader::Dataset,
    },
};

const NULL: f32 = 0.0;
const I: Complex<f32> = Complex::new(0.0, 1.0);

pub type Degree = i64;
pub type Order = i64;
pub type Angle = f32;
pub type Theta = Angle;
pub type Phi = Angle;
pub type PhiLM = [Complex<f32>; 2];
pub type StationName = String;
pub type VecSphFn = Arc<dyn Fn(Theta, Phi) -> VecSph + 'static + Send + Sync>;
pub type TimeSeries = Array1<f32>;
pub type Series = Array1<f32>;
pub type ComplexSeries = Array1<ndrustfft::Complex<f32>>;
pub type DataVector = DashMap<NonzeroElement, TimeSeries>;
pub type Modes = Vec<Mode>;
pub type NonzeroElements = Vec<NonzeroElement>;
pub type Real = bool;
pub type Frequency = f32;
pub type FrequencyIndex = usize;
pub type DFTValue = Complex<f32>;


pub trait Theory: Clone + Send {

    // const MODES: Modes;
    const NONZERO_ELEMENTS: usize;
    const MIN_STATIONS: usize;

    /// Gets nonzero elements for the theory.
    fn get_nonzero_elements() -> HashSet<NonzeroElement>;

    /// This calculates the pre-FFT data vector X^n_i's for a given theory. It combines data from many
    /// stations into a smaller subset of time series, weighted by their noise.
    fn calculate_projections(
        &self,
        weights_n: &DashMap<StationName, f32>,
        weights_e: &DashMap<StationName, f32>,
        weights_wn: &TimeSeries,
        weights_we: &TimeSeries,
        chunk_dataset: &DashMap<StationName, Dataset>,
    ) -> DashMap<NonzeroElement, TimeSeries>;

    // /// This calculates the theoretical signal expected for the Earth's magnetic field
    // /// for a given set of coordinates.
    fn calculate_data_vector(
        &self,
        projections: DashMap<NonzeroElement, TimeSeries>,
        // frequencies: &[Frequency],
        // total_time: f32,
    ) -> DashMap<NonzeroElement, ComplexSeries>;
}


#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Mode(pub Degree, pub Order);

#[derive(Debug)]
pub struct VecSphs<T: IntoIterator<Item=Complex<f32>>> {
    pub vec_sphs: HashMap<StationName, T>,
    modes: Modes,
}

/// These are the nonzero elements for a given theory
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct NonzeroElement {
    /// This is the index assigned to the element. Not to be confused with the
    /// chunk index
    pub index: usize,
    /// Optional name for the field
    pub name: Option<String>,
    /// Associated mode, component
    pub assc_mode: (Mode, Component),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub enum Component {
    RadialReal,
    RadialImag,
    PolarReal,
    PolarImag,
    AzimuthReal,
    AzimuthImag,
}



impl VecSphs<PhiLM> {
    pub fn new(modes: Modes) -> Self {

        // Construct Coordinate Map
        let coordinate_map = construct_coordinate_map();

        // // Initialize HashMap that stores signal
        // let mut signal = HashMap::new();

        // Construct spherical harmonic functions for each mode
        let mode_fns: Vec<(Mode, VecSphFn)> = modes
            .iter()
            .map(|&mode| (mode, vector_spherical_harmonic(mode)))
            .collect();

            
        let vec_sphs: HashMap<StationName, PhiLM> = coordinate_map.iter().map(|(station, coordinates)| {

            // Unpack coord
            // let coordinates: Coordinates = *coordinate_map.get(&station).unwrap();
            let [theta, phi] = [coordinates.polar as f32, coordinates.longitude as f32];

            // Initialize value
            let mut phi_lm: [Complex<f32>; 2] = [Complex::default(); 2];

            // Add modes
            for (mode, vec_sph_fn) in mode_fns.iter() {
                
                let _phi = vec_sph_fn(theta, phi).phi;

                phi_lm[0] += _phi[0];
                phi_lm[1] += _phi[1];
            }

            // assert!(signal.insert(station, phi_lm).is_none());
            (station.clone(), phi_lm)
        }).collect();


        Self {
            vec_sphs,
            modes
        }
    }
}

#[derive(Debug)]
pub struct VecSph {

    // This is Y_lm * r_hat. Only the non-zero component is returned (radial).
    pub y: Complex<f32>,
    
    // This is |vec(r)| * grad(Y_lm). Only the nonzero components are returned (angular components)
    pub psi: [Complex<f32>; 2],

    // This is vec(r) x grad(Y_lm). Only the nonzero components are returned (angular components)
    pub phi: [Complex<f32>; 2],
}


/// This returns a lookup table of vector spherical harmonic functions for a given set of modes.
pub fn vector_spherical_harmonics(modes: Box<[Mode]>) -> DashMap<Mode, VecSphFn> {

    // Initialize return value
    let mut hashmap = DashMap::new();

    // Get vector spherical harmonic functions for every mode
    for &mode in modes.iter() {
        hashmap.insert(mode, vector_spherical_harmonic(mode));
    }

    hashmap
}

/// Given a mode, this function returns a VecSphFn which is a function f(theta, phi) -> Complex<f32>
pub fn vector_spherical_harmonic(mode: Mode) -> VecSphFn {

    // Unpack mode
    let Mode(l, m) = mode;

    // Construct and Box function
    Arc::new(
        move |theta: Theta, phi: Phi| -> VecSph {

            // Set up spherical harmonic calculation
            let sh = ComplexSHType::Spherical;
            let p: SphrsCoordinates<f32> = SphrsCoordinates::spherical(NULL, theta, phi);

            // Calculate Y_lm
            let y = sh.eval(l, m, &p);
        
            // Calculate components of psi_lm and phi_lm
            let a = (m as f32) * theta.tan().recip() * y 
                + (sqrt(gamma((1 + l - m) as f32)) * sqrt(gamma((2 + l + m) as f32)) * sh.eval(l, m + 1, &p))
                    / ((I*phi).exp()*sqrt(gamma((l - m) as f32))*sqrt(gamma((1 + l + m) as f32)));
            let b = I * m as f32 * theta.sin().recip() * y;
            
            // Consruct psi_lm and phi_lm
            let psi = [ a, b];
            let phi = [-b, a];

            VecSph {
                y,
                psi,
                phi,
            }
        }
    )
}


fn gamma(x: f32) -> f32 {
    x.gamma()
}
fn sqrt(x: f32) -> f32 {
    x.sqrt()
}