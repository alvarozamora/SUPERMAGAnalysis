use std::ops::Range;

/// Number of leap years in the 1s SUPERMAG dataset
pub const NUM_LEAP_YEARS: usize = 6;

/// Days for which there is data
pub const DATA_DAYS: Range<usize> = 0..365 * (2020 - 1998 + 1) + NUM_LEAP_YEARS;

/// This is the value that appears in the SUPERMAG dataset to represent a null entry.
pub const SUPERMAG_NAN: f32 = 999999.0;

/// Radius of earth in meters
pub const EARTH_RADIUS: f32 = 6_371_009.0;

/// Seconds per day
pub const SECONDS_PER_DAY: usize = 24 * 60 * 60;

/// Magnetic field components
pub const NUM_FIELDS: usize = 3;

/// Number of bytes per float
pub const BYTES_PER_FLOAT: usize = 4;

/// Number of seconds in a sidereal day
// pub const SIDEREAL_DAY_SECONDS: usize = 86164/*.0905*/;
pub const SIDEREAL_DAY_SECONDS: f64 = 86164.0905;

/// Canonical minimum multiple for frequency bin
pub const I_MIN: usize = (INV_VEL_SQ / (1.0 + THRESHOLD)) as usize;

/// Inverse velocity squared of dm
pub const INV_VEL_SQ: f64 = 1e6;

// percent level accuracy of all of the frequencies in a frequency bins
pub const THRESHOLD: f64 = 0.03;
