use dashmap::{DashMap, ReadOnlyView};
use interp1d::Interp1d;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::constants::SECONDS_PER_DAY;

use super::loader::{day_since_first, StationName};

type SecFloat = f64;
type Declination = f64;

const IGRF_DATA_FILE: &str = "./src/IGRF_declinations_for_1sec.txt";

// #[derive(Debug, Deserialize)]
// /// This struct is just used for deserialization of the text file
// struct IgrfDataLine {
//     year: usize,
//     station: String,
//     geolat: f64,
//     geolon: f64,
//     declination: f64,
// }

lazy_static! {
    pub static ref IGRF_DATA_INTERPOLATOR: Declinations = Declinations::load();
}

pub struct Declinations {
    pub inner: ReadOnlyView<StationName, Interp1d<f64, f64>>,
}
impl Declinations {
    pub fn load() -> Declinations {
        // Initialize return value
        let inner = DashMap::new();

        // Iterate through lines, skipping header
        let file = File::open(IGRF_DATA_FILE).expect("IGRF_DATA_FILE should exist");
        let buf_reader = BufReader::new(file);
        let entries: Vec<(usize, String, f64, f64, f64)> = buf_reader
            .lines()
            .skip(1)
            .map(|line| {
                // Iterator over entries in the line
                let line = line.unwrap();
                let mut entries = line.split_whitespace();

                // Construct tuple
                (
                    entries.next().unwrap().parse().unwrap(),
                    entries.next().unwrap().to_owned(),
                    entries.next().unwrap().parse().unwrap(),
                    entries.next().unwrap().parse().unwrap(),
                    entries.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
        //     // .multiunzip();

        // let mut rdr = ReaderBuilder::new()
        //     .has_headers(true)
        //     // .trim(Trim::All)
        //     .flexible(true)
        //     .delimiter(b' ')
        //     .from_path(IGRF_DATA_FILE)
        //     .expect("IGRF_DATA_FILE should exist");

        // let other_entries = rdr
        //     .deserialize::<IgrfDataLine>()
        //     .collect::<Result<Vec<IgrfDataLine>, csv::Error>>().unwrap();
        // println!("{other_entries:?}");

        for (year, station, _geolat, _geolon, declination) in entries {
            // Convert declinations from degrees to radians
            let declination = declination * PI / 180.0;

            // Add declination (radians) to map
            inner
                .entry(station)
                .and_modify(|map: &mut (Vec<SecFloat>, Vec<Declination>)| {
                    map.0.push(convert_entry_year_to_sec_float(year));
                    map.1.push(declination);
                })
                .or_insert((
                    vec![convert_entry_year_to_sec_float(year)],
                    vec![declination],
                ));
        }

        let inner = inner
            .into_par_iter()
            .map_with((), |_, (station_name, (secs, declinations))| {
                // Sort declinations for this station
                let (secs, declinations) = secs
                    .into_iter()
                    .zip(declinations.into_iter())
                    .sorted_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unzip();
                // Construct key-value pair with interp1d
                (
                    station_name,
                    Interp1d::new_sorted(secs, declinations)
                        .expect("failed to create interpolator"),
                )
            })
            .collect::<DashMap<_, _>>()
            .into_read_only();

        Declinations { inner }
    }

    // Return a reference to the station interpolator
    pub fn interpolator(&self, station: &str) -> &Interp1d<f64, f64> {
        self.inner
            .get(station)
            .expect(&format!("station {station} should exist"))
    }

    pub fn interpolate(&self, station: String, sec: usize) -> Declination {
        // Get x and y vecs
        let interpolator = self.interpolator(&station);

        // Intepolate
        interpolator
            .interpolate_checked(sec as f64)
            .expect("out of bounds")
    }
}

/// The values given are for the start of the 180th day of each year.
fn convert_entry_year_to_sec_float(year: usize) -> SecFloat {
    // Index day by day since first day of data
    let day: usize = day_since_first(179 /* days start at 0 */, year);

    // Here we assume we refer to the first second of the 180th day
    let sec: usize = day * SECONDS_PER_DAY + 1;

    sec as SecFloat
}

// #[inline(always)]
// fn apply_rotation(f1: f32, f2: f32, theta: f32) -> (f32, f32) {
//     (
//         f1 * (theta).cos() - f2 * (theta).sin(),
//         f1 * (theta).sin() + f2 * (theta).cos(),
//     )
// }
#[inline(always)]
pub(crate) fn apply_rotation_f1(f1: f32, f2: f32, theta: f32) -> f32 {
    f1 * (theta).cos() - f2 * (theta).sin()
}

#[inline(always)]
pub(crate) fn apply_rotation_f2(f1: f32, f2: f32, theta: f32) -> f32 {
    f1 * (theta).sin() + f2 * (theta).cos()
}

#[test]
fn test_convert_entry_year_to_sec_float() {
    assert_eq!(
        convert_entry_year_to_sec_float(1998),
        // On the beginning of the 180th day, 179 full days and 1 second have passed
        (179 * SECONDS_PER_DAY + 1) as SecFloat
    );

    assert_eq!(
        convert_entry_year_to_sec_float(1999),
        // On the beginning of the 180th day, 179 full days and 1 second have passed
        ((179 + 365) * SECONDS_PER_DAY + 1) as SecFloat
    );

    assert_eq!(
        convert_entry_year_to_sec_float(2000),
        // On the beginning of the 180th day, 179 full days and 1 second have passed
        ((179 + 365 * 2) * SECONDS_PER_DAY + 1) as SecFloat
    );

    // 2000 was a leap year
    assert_eq!(
        convert_entry_year_to_sec_float(2001),
        // On the beginning of the 180th day, 179 full days and 1 second have passed
        ((179 + 365 * 2 + 366) * SECONDS_PER_DAY + 1) as SecFloat
    );
}

#[test]
fn test_declination_interpolation() {
    use approx_eq::assert_approx_eq;
    /*
     * IGRF_declinations_for_1sec.txt
     *
     * year   IAGA     geolat      geolon    declination
     * 1998   SON       25.12       66.44        0.34
     * ..
     * 1999   SON       25.12       66.44        0.36
     * ..
     *
     */

    let sec_1998: usize = 179 * SECONDS_PER_DAY + 1;
    let sec_1999: usize = (179 + 365) * SECONDS_PER_DAY + 1;
    let sec_mid = (sec_1998 + sec_1999) / 2;

    // Check left
    assert_approx_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_1998),
        0.34 * PI / 180.0,
        1e-6
    );

    // Check right
    assert_approx_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_1999),
        0.36 * PI / 180.0,
        1e-6
    );

    // Check mid (actually interpolating now)
    assert_approx_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_mid),
        0.35 * PI / 180.0, // (0.34 + 0.36) / 2.0
        1e-6
    );
}
