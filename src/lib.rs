#[macro_use]
extern crate lazy_static;

pub mod constants;
pub mod theory;
pub mod utils;
pub mod weights;

// Note that this has to stay constant throughout the entire pipeline
// e.g. for both projections_auxiliary, analysis, and any check scripts
pub type FloatType = f32;

pub type Index = usize;
pub type Weight = FloatType;
pub type StationName = String;
pub type TimeSeries = ndarray::Array1<FloatType>;
