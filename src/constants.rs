use std::ops::Range;

///  Number of years with data
pub const DATA_YEARS: usize = 50;

/// Number of leap years in the 1s SUPERMAG dataset
pub const NUM_LEAP_YEARS: usize = 6;

/// Days for which there is data
pub const DATA_DAYS: Range<usize> = 0..365*23 + NUM_LEAP_YEARS;

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



