use std::fs::File;
use std::collections::HashMap;
use std::io::prelude::*;
use std::sync::Arc;
use std::path::PathBuf;
use anyhow::Result;
use std::convert::TryInto;
use ndarray::{arr1, Array1};
use glob::glob;
use crate::utils::coordinates::*;
use tokio::io::AsyncReadExt;
use dashmap::DashMap;
use crate::constants::*;
use anyhow::Error;
use futures::future::join_all;
use ndarray::Axis;
use rayon::prelude::*;
use rayon::iter::*;

pub type TimeSeries = Array1<f32>;
pub type Index = usize;
pub type StationName = String;


/// Size of first header in bytes
const HEADER_SIZE: usize = 315 * 4;

/// Size of intra-field buffer
const BUFFER_SIZE: usize = 25*4;

/// Size of final buffer
const END_SIZE: usize = 48*4;

/// Size of final buffer (2018-2019)
const END_SIZE_2018_2019: usize = 29*4;

/// Size of daily file
const DAILY_FILE_SIZE: usize = SECONDS_PER_DAY * NUM_FIELDS * BYTES_PER_FLOAT;


/// This struct contains the three fields for a station, along with the station's name and coordinates.
/// It also contains some `index` used to represent the year, day, or chunk that these time series represent.
pub struct Dataset {
    pub field_1: TimeSeries,
    pub field_2: TimeSeries,
    pub field_3: TimeSeries,
    pub coordinates: Coordinates,
    pub station_name: StationName,
    /// This is e.g the year, day, or chunk
    pub index: Index,
}

pub struct YearlyDatasetLoader {
    pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
    pub station_files: Vec<String>,
    pub year: Index,
}

impl YearlyDatasetLoader {

    /// Construct a new `DatasetLoader` given a `year`.
    pub fn new_from_year(year: usize) -> Self {

        // Construct coordinate map 
        let coordinate_map = Arc::new(construct_coordinate_map());

        // Find all stations for which there is data for this year
        let station_files = retrieve_stations(year);

        YearlyDatasetLoader{
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
            Ok(YearlyDatasetLoader {
                coordinate_map: self.coordinate_map,
                station_files,
                year
            })
        } else {
            Err(anyhow::Error::msg("No stations found for this year"))
        }
        
    }

    /// Loads the `Dataset` specified by the next file in the `station_files` vector.
    pub fn load(year: usize, station_file: String, coordinate_map: Arc<HashMap<StationName, Coordinates>>) -> Dataset {

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
            index: year,
        }
    }

    /// Computes remaining number of files
    pub fn len(&self) -> usize {
        self.station_files.len()
    }
}

pub struct DailyDatasetLoader {
    pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
    pub station_files: Vec<String>,
    pub day: Index,
}

impl DailyDatasetLoader {

    /// Construct a new `DatasetLoader` given a `year`.
    pub fn new_from_day(day: usize) -> Self {

        // Construct coordinate map 
        let coordinate_map = Arc::new(construct_coordinate_map());

        // Find all stations for which there is data for this day
        let station_files = retrieve_stations_daily(day);

        DailyDatasetLoader {
            coordinate_map,
            station_files,
            day
        }

    }
    
    /// Changes the loader to a different day. (This skips construction of coordinate map, 
    /// which is rather cheap but I hate unnecessarily repeating calculations).
    pub fn change_to_day(self, day: usize) -> Result<Self> {

        // Find all stations for which there is data for this year
        let station_files = retrieve_stations_daily(day);

        if station_files.len() > 0 {
            Ok(DailyDatasetLoader {
                coordinate_map: self.coordinate_map,
                station_files,
                day
            })
        } else {
            Err(anyhow::Error::msg("No stations found for this year"))
        }
        
    }

    /// Loads the `Dataset` specified by the next file in the `station_files` vector.
    pub async fn load_next(&mut self) -> Option<Dataset> {

        // Return None if empty. Get next file if not.
        if self.len() == 0 { return None };
        let station_file = self.station_files.pop().unwrap();

        // Calculate expected size of each f32 fields, in bytes
        let expected_size = 24 * 60 * 60 * 4;

        // Open station data for this year
        let mut file = tokio::fs::File::open(&station_file).await.expect("couldn't open station file");

        // Load file contents into buffer
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await.expect("failed to load file contents into buffer");

        // Ensure buffer size matches expectations
        assert_eq!(buffer.len(), 3*expected_size, "buffer size is not correct");

        // Break up buffer into expected_size chunks
        let [field_1, field_2, field_3]: [Array1<f32>; 3] = {

            buffer
                .chunks_exact_mut(expected_size)
                .map(|x| arr1(&x.to_f32()))
                .collect::<Vec<TimeSeries>>()
                .try_into()
                .unwrap()
        };

        // Find station name, which is HashMap key for Coordinates
        let station_name: String = station_file
            .split("/")
            .collect::<Vec<&str>>()
            .get(2)
            .expect("Non-standard station filename format")
            .to_string();
        println!("station_name = {station_name}");
        let coordinates: Coordinates = self.coordinate_map
            .get(&station_name)
            .map(|&x| x)
            .unwrap();
        println!("{:?}", field_1);

        // Return Dataset
        Some(Dataset {
            field_1,
            field_2,
            field_3,
            coordinates,
            station_name,
            index: self.day,
        })
    }

    /// Computes remaining number of files
    pub fn len(&self) -> usize {
        self.station_files.len()
    }
}

#[derive(Clone)]
pub struct Chunk {
    station: StationName,
    files: Vec<PathBuf>,
}

/// This `DatasetLoader` is intended to be used to gather 1s data in chunks (integer number of days).
/// The struct contains all the metadata required to load chunks of data, and comes equipped with methods
/// to load those chunks.
pub struct DatasetLoader {
    pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
    pub semivalid_chunks: Arc<DashMap<Index, Vec<Chunk>>>,
    days: usize,
    chunk: usize,
}

impl DatasetLoader {

    // This function gathers all metadata required for loading a chunk of data
    pub fn new(days: usize) -> Self {

        // Construct coordinate map 
        let coordinate_map = Arc::new(construct_coordinate_map());

        // Find all stations for which there is data for this year
        let semivalid_chunks = retrieve_chunks(days);

        // Chunk initialized at zero
        let chunk = 0;

        Self {
            coordinate_map,
            semivalid_chunks,
            days,
            chunk,
        }

    }

    /// This function loads and returns a chunk of data. Internally, it loads and combines the files
    /// that comprise a chunk to return a single `TimeSeries` for each field. Here, `index` is the index
    /// of the chunk, i.e. 0 is the first chunk.
    pub async fn load_chunk(&self, index: Index) -> Result<DashMap<StationName, Dataset>> {

        // Check for valid chunk index
        match self.semivalid_chunks.get(&index) {

            // Load data if valid
            Some(chunks) => {

                // Initialize DashMap that contains the combined `Dataset` for every station for this chunk
                let result = DashMap::with_capacity(chunks.len());
                
                // Construct a collection of futures for every station/chunk
                let futs = chunks
                    .iter()
                    .map(|chunk| async { 
                        let station_name: String = chunk.station.split("/").collect::<Vec<&str>>().get(2).unwrap().to_string();
                        tokio::spawn(
                            _load_chunk(index, chunk.clone(), *self.coordinate_map.get(&station_name).unwrap())
                        ).await.unwrap() 
                    })
                    .collect::<Vec<_>>();

                // Get all ordered, combined datasets
                let data: Vec<Dataset> = join_all(futs).await;
                
                for (chunk, data) in chunks.iter().zip(data) {
                    let cleaned_station_name: String = chunk.station.split("/").collect::<Vec<_>>().get(2).unwrap().to_string();
                    assert!(result.insert(cleaned_station_name, data).is_none());
                }

                Ok(result)
            },

            // Return Err otherwise
            None => Err(Error::msg("Invalid Index")),
        }
    }
}


async fn _load_chunk(index: Index, chunk: Chunk, coordinates: Coordinates) -> Dataset {

    // Construct a collection of futures for the daily datasets in this chunk for this station
    let dataset_futures = chunk
        .files
        .iter()
        .map(|filepath| load_daily(filepath.clone()))
        .collect::<Vec<_>>();

    // Execute futures concurrently on one thread with join_all
    let mut datasets: Vec<[TimeSeries; 3]> = join_all(dataset_futures).await;

    // Concatenate datasets
    let [field_1, field_2, field_3]: [TimeSeries; 3] = {
        let combined_dataset = datasets
        .iter_mut()
        .fold(Dataset::default(), |mut acc, dataset| {
            acc.field_1.append(Axis(0), dataset[0].view()).unwrap();
            acc.field_2.append(Axis(0), dataset[1].view()).unwrap();
            acc.field_3.append(Axis(0), dataset[2].view()).unwrap();
            acc
        });
        [combined_dataset.field_1, combined_dataset.field_2, combined_dataset.field_3]
    };

    Dataset {
        field_1,
        field_2,
        field_3,
        station_name: chunk.station,
        coordinates,
        index,
    }
}

async fn load_daily(filepath: PathBuf) -> [TimeSeries; 3] {

    // Open file
    let mut file = tokio::fs::File::open(filepath).await.expect("failed to open daily file");

    // Load data
    let mut buf = Vec::with_capacity(DAILY_FILE_SIZE);
    file.read_to_end(&mut buf).await.expect("failed to load file into buffer");

    // Convert file to fields
    let [field_1, field_2, field_3]: [TimeSeries; 3] = buf
        .to_f32()
        .chunks_exact_mut(SECONDS_PER_DAY)
        .map(|field| arr1(field))
        .collect::<Vec<TimeSeries>>()
        .try_into()
        .unwrap();

    [field_1, field_2, field_3]
}


fn retrieve_stations(year: usize) -> Vec<String> {
    let paths: Vec<String> = glob(format!("../{}/*.xdr", year).as_str())
        .expect("couldn't find datasets")
        .map(|x| x.unwrap().display().to_string())
        .collect();
    println!("Found {} files for {year}", paths.len());
    paths
}

fn retrieve_chunks(days: usize) -> Arc<DashMap<Index, Vec<Chunk>>> {

    // Gather paths to all stations
    let stations: Vec<_> = glob("../stations/*").unwrap().map(|x| x.unwrap()).collect::<Vec<_>>();
    let station_days: std::collections::HashSet<PathBuf> = glob("../stations/*/*").unwrap().map(|x| x.unwrap()).collect();

    // Initialize DashMap containing semivalid chunks
    let semivalid_chunks = Arc::new(DashMap::new());


    // Iterate through every chunk of days
    DATA_DAYS
        .step_by(days)
        .map(|x| (x..).take(days))
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .for_each(|(index, chunk)| {

        // Initialize vector which holds which stations contain semi-valid data for this chunk
        let chunks: Vec<Chunk> = stations
            .iter()
            .filter_map(|station| {

                // Check if files exist for that (station, chunk)
                // First, construct file paths.
                let files: Vec<PathBuf> = chunk
                    .clone()
                    .map(|day|  PathBuf::from(format!("{}/{day}", station.display())))
                    .collect();

                // Then check if they all exist
                if files.iter().all(|file| station_days.contains(file)) {
                    
                    // If so, return `Chunk`
                    Some(Chunk {
                        station: station.display().to_string(),
                        files,
                    })
                } else {
                    // Otherwise, don't include this station for this chunk
                    None
                }
            })
            .collect();

            // Add entries to semivalid_chunks
            semivalid_chunks.insert(index, chunks);
    });

  semivalid_chunks
}


fn retrieve_stations_daily(day: usize) -> Vec<String> {
    let paths: Vec<String> = glob(format!("../stations/*/{}", day).as_str())
        .expect("couldn't find datasets")
        .map(|x| x.unwrap().display().to_string())
        .collect();
    println!("Found {} files for {day}", paths.len());
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
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    }
}

impl Default for Dataset {
    fn default() -> Dataset {
        Dataset {
            field_1: arr1(&[]),
            field_2: arr1(&[]),
            field_3: arr1(&[]),
            station_name: String::new(),
            coordinates: Coordinates { latitude: 0.0, longitude: 0.0, polar: std::f64::consts::FRAC_PI_2 },
            index: 0,
        }
    }
}