use sphrs::{ComplexSHType, Coordinates as SphrsCoordinates, SHEval};
use dashmap::DashMap;
use num_complex::Complex;
use special::Gamma;
use num_traits::{Float, FromPrimitive, FloatConst};
use std::fmt::Debug;
use std::ops::Mul;

/// A helpful type alias to distingush the two i64 arguments of the function in `VecSphFn`.
pub type Degree = i64;
/// A helpful type alias to distingush the two i64 arguments of the function in `VecSphFn`.
pub type Order = i64;


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
/// A struct with two fields `polar` and `azim` describing a position on the sphere.
pub struct Coordinates<T: Float> {
    /// The polar coordinate ranging from [0, pi] (periodic)
    pub polar: T,
    /// The azimuthal coordinate ranging from [0, 2pi] (periodic)
    pub azim: T,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Mode(pub Degree, pub Order);

pub type VecSphFn<U> = Box<dyn Fn(U, U) -> VecSph<U> + 'static + Send + Sync>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VecSph<T: Float> {

    /// This is Y_lm * r_hat. Only the non-zero component is returned (radial).
    pub y: Complex<T>,
    
    /// This is |vec(r)| * grad(Y_lm). Only the nonzero components are returned (angular components)
    pub psi: [Complex<T>; 2],

    /// This is vec(r) x grad(Y_lm). Only the nonzero components are returned (angular components)
    pub phi: [Complex<T>; 2],
}


/// This returns a lookup table of vector spherical harmonic functions for a given set of modes.
pub fn vector_spherical_harmonics<T>(modes: &[Mode]) -> DashMap<Mode, VecSphFn<T>> 
where
    T: Float + FromPrimitive + FloatConst + FromPrimitive + Debug + Gamma + Send + 'static + Sync,
{

    // Initialize return value
    let mut hashmap = DashMap::new();

    // Get vector spherical harmonic functions for every mode
    for &mode in modes.iter() {
        hashmap.insert(mode, vector_spherical_harmonic(mode));
    }

    hashmap
}

/// Given a mode, this function returns a VecSphFn which is a function f(theta, phi) -> Complex<f32>
pub fn vector_spherical_harmonic<T>(mode: Mode) -> VecSphFn<T>
where
    T: Float + FloatConst + FromPrimitive + Debug + Gamma + Send + 'static + Sync,
    Complex<T>: Mul<T, Output=Complex<T>> + Mul<Complex<T>, Output=Complex<T>>,
{

    // Unpack mode
    let Mode(l, m) = mode;

    // Define i and null
    let i: Complex<T> = Complex::<T>::new(T::zero(), T::one());

    // Construct and Box function
    Box::new(
        move |theta: T, phi: T| -> VecSph<T> {

            // Set up spherical harmonic calculation
            let sh = ComplexSHType::Spherical;
            let p: SphrsCoordinates<T> = SphrsCoordinates::spherical(T::zero(), theta, phi);

            // Calculate Y_lm
            let y = sh.eval(l, m, &p);
        
            // Calculate components of psi_lm and phi_lm
            let a: Complex<T> = y * T::from_i64(m).unwrap() * theta.tan().recip() 
                + ( sh.eval(l, m + 1, &p) * ((T::from_i64(1 + l - m).unwrap()).gamma()).sqrt() * ((T::from_i64(2 + l + m).unwrap()).gamma()).sqrt())
                    / ((i*phi).exp()*((T::from_i64(l - m).unwrap()).gamma()).sqrt()*((T::from_i64(1 + l + m).unwrap()).gamma()).sqrt());
            let b: Complex<T> = i * y * T::from_i64(m).unwrap() * theta.sin().recip();
            
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

/// This struct stores the map between Mode <--> VecSphFn<T>
pub struct VectorSphericalHarmonics<T: Float> {
    /// Store the map between Mode <--> VecSphFn<T>
    pub vec_sph_fns: DashMap<Mode, VecSphFn<T>>,
}

impl<T: Float> VectorSphericalHarmonics<T> {

    /// Given any collection of modes, generate a map between Mode <-> VecSphFn<T> and store it in the struct (`Self`).
    /// The modes in the iterator must be unique. If they are not, the function will simply ignore repeated modes and
    /// return a struct containing a `DashMap` whose number of elements is less than that of the provided collection of
    /// modes.
    pub fn new(modes: impl ExactSizeIterator<Item=Mode>) -> Self
    where
        T: Float + FloatConst + FromPrimitive + Debug + Gamma + Send + 'static + Sync,
        Complex<T>: Mul<T, Output=Complex<T>> + Mul<Complex<T>, Output=Complex<T>>,
    {
        
        // Initialize DashMap that stores the functions
        let vec_sph_fns = DashMap::with_capacity(modes.len());

        // Iterate through the modes, inserting the functions
        for mode in modes {
            vec_sph_fns.insert(mode, vector_spherical_harmonic(mode));
        }

        // Return struct
        VectorSphericalHarmonics{
            vec_sph_fns
        }
    }

    /// Given some set of coordinates on the sphere, calculate all of the vector spherical harmonics `VecSph` for the modes provided
    /// into the `VectorSphericalHarmonics` struct. Since `Coordinates` is not hashable, the key used is the index/position of the
    /// `Coordinates<T>` in the collection.
    pub fn calculate_harmonics(&self, coordinates: impl ExactSizeIterator<Item=Coordinates<T>>) -> DashMap<usize, DashMap<Mode, VecSph<T>>>
    where
        T: Float + FloatConst + FromPrimitive + Debug + Gamma + Send + 'static + Sync,
        Complex<T>: Mul<T, Output=Complex<T>> + Mul<Complex<T>, Output=Complex<T>>,
    {

        // Initialize DashMap that will store the vector spherical harmonics for every mode for every set of coodinates
        let harmonics = DashMap::new();

        // Iterate through coordinates to calculate the vector spherical harmonics at each position
        for (index, position) in coordinates.enumerate() {

            // At this position, calculate all vector spherical harmonics
            let vec_sphs: DashMap<Mode, VecSph<T>> = self
                .vec_sph_fns
                .iter()
                .map(|key_value| {
                    let (&mode, vec_sph_fn) = key_value.pair();
                    let value = vec_sph_fn(position.polar, position.azim);
                    (mode, value)
                })
                .collect();

            // Insert all (`Mode`, `VecSph`) pairs for this position into the `harmonics` map
            harmonics.insert(index, vec_sphs);
        }

        // Return map to the map of values
        harmonics
    }

    /// Given some set of coordinates on the sphere, calculate all of the vector spherical harmonics `VecSph` for the modes provided
    /// into the `VectorSphericalHarmonics` struct. Since `Coordinates` is not hashable, the key used is the index/position of the
    /// `Coordinates<T>` in the collection.
    /// 
    /// Equivalent to `calculate_harmonics` but is done using a `rayon` parallel iterator. Requires `rayon` feature.
    #[cfg(feature = "rayon")]
    pub fn calculate_harmonics_parallel(&self, coordinates: impl ExactSizeIterator<Item=Coordinates<T>>) -> DashMap<usize, DashMap<Mode, VecSph<T>>>
    where
        T: Float + FloatConst + FromPrimitive + Debug + Gamma + Send + 'static + Sync,
        Complex<T>: Mul<T, Output=Complex<T>> + Mul<Complex<T>, Output=Complex<T>>,
    {

        // Initialize DashMap that will store the vector spherical harmonics for every mode for every set of coodinates
        let harmonics = DashMap::new();

        // Iterate through coordinates to calculate the vector spherical harmonics at each position
        for (index, position) in coordinates.enumerate() {

            // At this position, calculate all vector spherical harmonics
            let vec_sphs: DashMap<Mode, VecSph<T>> = self
                .vec_sph_fns
                .par_iter_mut()
                .map(|key_value| {
                    let (&mode, vec_sph_fn) = key_value.pair();
                    let value = vec_sph_fn(position.polar, position.azim);
                    (mode, value)
                })
                .collect();

            // Insert all (`Mode`, `VecSph`) pairs for this position into the `harmonics` map
            harmonics.insert(index, vec_sphs);
        }

        // Return map to the map of values
        harmonics
    }
}

