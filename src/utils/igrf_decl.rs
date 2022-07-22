use dashmap::DashMap;
use interp::interp;
use csv::{ReaderBuilder, StringRecord, Trim};
use std::io::{self, BufReader, BufRead};
use std::fs::File;
use serde_derive::Deserialize;

use crate::constants::SECONDS_PER_DAY;

use super::loader::{StationName, day_since_first};

type SecFloat = f64;
type Declination = f64;


const IGRF_DATA_FILE: &str = "./src/IGRF_declinations_for_1sec.txt";

#[derive(Debug, Deserialize)]
/// This struct is just used for deserialization of the text file
pub struct IgrfDataLine {
    year: usize,
    station: String,
    geolat: f64,
    geolon: f64,
    declination: f64,
}


lazy_static! {
    pub static ref IGRF_DATA_INTERPOLATOR: Declinations = load_igrf_data();
}

pub struct Declinations {
    pub inner: DashMap<StationName, (Vec<SecFloat>, Vec<Declination>)>,
}


fn load_igrf_data() -> Declinations {

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
            (entries.next().unwrap().parse().unwrap(),
            entries.next().unwrap().to_owned(),
            entries.next().unwrap().parse().unwrap(),
            entries.next().unwrap().parse().unwrap(),
            entries.next().unwrap().parse().unwrap())
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

        inner
            .entry(station)
            .and_modify(|map: &mut (Vec<SecFloat>, Vec<Declination>)| {
                map.0.push(convert_entry_year_to_sec_float(year));
                map.1.push(declination);
            })
            .or_insert((vec![convert_entry_year_to_sec_float(year)], vec![declination]));
    }


    Declinations { inner }
}


impl Declinations {

    pub fn interpolate(&self, station: String, sec: usize) -> Declination {

        // Get x and y vecs
        let station_data = self.inner.get(&station).unwrap();

        // Intepolate
        interp(&station_data.0, &station_data.1, sec as f64)
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
        ((179 + 365*2) * SECONDS_PER_DAY + 1) as SecFloat
    );

    // 2000 was a leap year
    assert_eq!(
        convert_entry_year_to_sec_float(2001),
        // On the beginning of the 180th day, 179 full days and 1 second have passed
        ((179 + 365*2 + 366) * SECONDS_PER_DAY + 1) as SecFloat
    );
}


#[test]
fn test_declination_interpolation() {

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

    let sec_1998: usize  = 179 * SECONDS_PER_DAY + 1;
    let sec_1999: usize  = (179 + 365) * SECONDS_PER_DAY + 1;
    let sec_mid = (sec_1998 + sec_1999) / 2;

    // Check left
    assert_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_1998),
        0.34
    );

    // Check right
    assert_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_1999),
        0.36
    );

    // Check mid (actually interpolating now)
    assert_eq!(
        IGRF_DATA_INTERPOLATOR.interpolate(String::from("SON"), sec_mid),
        0.35 // (0.34 + 0.36) / 2.0
    );
}