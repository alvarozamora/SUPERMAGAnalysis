use std::fs::File;
use std::collections::HashMap;
use std::io::prelude::*;
use std::sync::Arc;
use anyhow::Result;
use std::convert::TryInto;
use ndarray::{arr1, Array1};
use glob::glob;
use crate::utils::coordinates::*;


/// Size of first header in bytes
const HEADER_SIZE: usize = 315 * 4;

/// Size of intra-field buffer
const BUFFER_SIZE: usize = 25*4;

/// Size of final buffer
const END_SIZE: usize = 48*4;

/// Size of fiinal buffer (2018-2019)
const END_SIZE_2018_2019: usize = 29*4;

pub struct DatasetLoader {
    pub coordinate_map: Arc<HashMap<String, Coordinates>>,
    pub station_files: Vec<String>,
    pub year: usize,
}

impl DatasetLoader {

    /// Construct a new `DatasetLoader` given a `year`.
    pub fn new_from_year(year: usize) -> Self {

        // Construct coordinate map 
        let coordinate_map = Arc::new(construct_coordinate_map());

        // Find all stations for which there is data for this year
        let station_files = retrieve_stations(year);

        DatasetLoader{
            coordinate_map,
            station_files,
            year
        }

    }
    
    /// Changes the loader to a different year. (This skips construction of coordinate map, 
    /// which is rather cheap but I hate unnecessarily repeating calculations).
    pub fn change_to_year(self, year: usize) -> Result<Self> {

        // Find all stations for which there is data for this year
        let station_files = retrieve_stations(year);

        if station_files.len() > 0 {
            Ok(DatasetLoader {
                coordinate_map: self.coordinate_map,
                station_files,
                year
            })
        } else {
            Err(anyhow::Error::msg("No stations found for this year"))
        }
        
    }

    /// Loads the `Dataset` specified by the next file in the `station_files` vector.
    pub fn load(year: usize, station_file: String, coordinate_map: Arc<HashMap<String, Coordinates>>) -> Dataset {

        // Calculate expected size of f32 fields, in bytes
        let expected_size = if year % 4 == 0 {
            // Leap year
            60 * 60 * 24 * 366 * 4
        } else {
            // Non-leap year
            60 * 60 * 24 * 365 * 4
        };

        // // Grab file from end
        // let station_file = self.station_files.pop().unwrap();

        // Open station data for this year
        let mut file = File::open(&station_file).expect("couldn't open station file");

        // Load file contents into buffer
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).expect("failed to load file contents into buffer");

        // Verification, 3e-45 seems to be "end of data"
        for i in 0..3 {
            let start = HEADER_SIZE + (i+1)*expected_size + i*BUFFER_SIZE-16;
            let end = HEADER_SIZE + (i+1)*expected_size + (i+1)*BUFFER_SIZE + 16;
            // println!("{:?}", String::from_utf8_lossy(&buffer[start..end]));
            // if i != 2 { println!("Header {i}:\n{:?}", buffer[start..end].to_f32()) } else { println!("Header {i}:\n{:?}", buffer[start..].to_f32())};
            assert_eq!(buffer[start..end].to_f32()[4], 3e-45);
        }

        // Ensure buffer size matches expectations
        validate_buffer_size(&buffer, expected_size, year);

        // Break up buffer into expected_size chunks
        //let _header: Vec<u8> = buffer[0..HEADER_SIZE].to_vec();
        let field_1: Vec<f32> = buffer[HEADER_SIZE..HEADER_SIZE+expected_size].to_f32();
        let field_2: Vec<f32> = buffer[HEADER_SIZE+BUFFER_SIZE+expected_size..HEADER_SIZE+BUFFER_SIZE+2*expected_size].to_f32();
        let field_3: Vec<f32> = buffer[HEADER_SIZE+2*BUFFER_SIZE+2*expected_size..HEADER_SIZE+BUFFER_SIZE+3*expected_size].to_f32();
        
        // Convert to ndarrays
        let field_1: Array1<f32> = arr1(&field_1);
        let field_2: Array1<f32> = arr1(&field_2);
        let field_3: Array1<f32> = arr1(&field_3);

        // Find station name, which is HashMap key for Coordinates
        let station_name: String = station_file
            .split("_")
            .collect::<Vec<&str>>()
            .get(1)
            .expect("Non-standard station filename format")
            .to_string();
        let coordinates: Coordinates = coordinate_map
            .get(&station_name)
            .map(|&x| x)
            .unwrap();

        // Return Dataset
        Dataset {
            field_1,
            field_2,
            field_3,
            coordinates,
            station_name,
            year,
        }
    }

    /// Computes remaining number of files
    pub fn len(&self) -> usize {
        self.station_files.len()
    }
}

pub struct Dataset {
    pub field_1: Array1<f32>,
    pub field_2: Array1<f32>,
    pub field_3: Array1<f32>,
    pub coordinates: Coordinates,
    pub station_name: String,
    pub year: usize,
}




fn retrieve_stations(year: usize) -> Vec<String> {
    let paths: Vec<String> = glob(format!("../{}/*.xdr", year).as_str())
        .expect("couldn't find datasets")
        .map(|x| x.unwrap().display().to_string())
        .collect();
    println!("Found {} paths for {year}", paths.len());
    paths
}

fn validate_buffer_size(buffer: &[u8], expected_size: usize, year: usize) -> Result<(), &str> {
    if year != 2018 && year != 2019 {
        assert_eq!(
            buffer.len(),
            HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE,
            "invalid buffer size: size diff is {}",
            buffer.len() as i64 - (HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE) as i64
        );
    } else {
        assert_eq!(
            buffer.len(),
            HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE_2018_2019,
            "invalid buffer size: size diff is {}",
            buffer.len() as i64 - (HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE_2018_2019) as i64
        );
    }
    Ok(())
}


/// Simple trait which converts all values in a collection to `f32`.
pub trait Tof32{
    fn to_f32(&mut self) -> Vec<f32>;
}

/// For a given collection of bytes (`u8`), convert to a collection of floats (`f32`).
impl Tof32 for [u8] {
    fn to_f32(&mut self) -> Vec<f32> {
        assert!(self.len() % 4 == 0, "Invalid buffer size for bytes -> f32");
        self
            .chunks(4)
            .map(|x| f32::from_be_bytes(x.try_into().unwrap()))
            .collect()
    }
}