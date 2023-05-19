use super::igrf_decl::{apply_rotation, shift_point, IGRF_DATA_INTERPOLATOR};
use crate::utils::coordinates::*;
use crate::{constants::*, FloatType, TimeSeries};
use anyhow::Result;
use dashmap::DashMap;
use glob::glob;
use indicatif::ProgressBar;
use ndarray::{arr1, Array1};
use rayon::iter::*;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::prelude::*;
use std::sync::Arc;

pub type Index = usize;
pub type Year = usize;
pub type StationName = String;

/// Size of first header in bytes
const HEADER_SIZE: usize = 315 * 4;

/// Size of intra-field buffer
const BUFFER_SIZE: usize = 25 * 4;

/// Size of final buffer
const END_SIZE: usize = 48 * 4;

/// Size of final buffer (2018-2019)
const END_SIZE_2018_2019: usize = 29 * 4;

// /// Size of daily file
// const DAILY_FILE_SIZE: usize = SECONDS_PER_DAY * NUM_FIELDS * BYTES_PER_FLOAT;

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

// static STATIONS: Lazy<Vec<PathBuf>> =
//     Lazy::new(|| glob("../stations/*").unwrap().map(|x| x.unwrap()).collect());
// static STATION_DAYS: Lazy<Vec<PathBuf>> = Lazy::new(|| {
//     glob("../stations/*/*")
//         .unwrap()
//         .map(|x| x.unwrap())
//         .collect()
// });

pub struct YearlyDatasetLoader {
    pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
    pub station_files: Arc<HashMap<Year, Vec<String>>>,
}

impl YearlyDatasetLoader {
    /// Construct a new `DatasetLoader` given a `year`.
    pub fn new() -> Self {
        // Construct coordinate map
        let coordinate_map = Arc::new(construct_coordinate_map());

        // Find all stations for which there is data for this year
        let station_files = retrieve_stations();

        YearlyDatasetLoader {
            coordinate_map,
            station_files,
        }
    }

    /// Loads the `Dataset` specified by the next file in the `station_files` vector.
    pub async fn load(&self, year: usize) -> DashMap<String, Dataset> {
        println!("Loading stations for year {year}");
        let pb = ProgressBar::new(self.station_files[&year].len() as u64);
        pb.inc(0);
        let map = self.station_files[&year]
            .par_iter()
            .flat_map(|station_file| {
                // Open station data for this year
                let mut file = File::open(&station_file).expect("couldn't open station file");

                // Get station name to retrieve interpolator
                let station_name: String = station_file
                    .split("_")
                    .collect::<Vec<&str>>()
                    .get(1)
                    .expect("Non-standard station filename format")
                    .to_string();

                // Calculate expected size of f32 fields, in bytes
                let mut expected_size = if year % 4 == 0 {
                    // Leap year
                    60 * 60 * 24 * 366 * 4
                } else {
                    // Non-leap year
                    60 * 60 * 24 * 365 * 4
                };

                // Load file contents into buffer
                let mut buffer = vec![];
                file.read_to_end(&mut buffer)
                    .expect("failed to load file contents into buffer");

                // Update expected buffer size if double precision is detected
                let single: SinglePrecision =
                    validate_buffer_size(&buffer, &mut expected_size, year).unwrap();

                // Verification, 3e-45 seems to be "end of data"
                for i in 0..3 {
                    let start = HEADER_SIZE + (i + 1) * expected_size + i * BUFFER_SIZE - 16;
                    let end = HEADER_SIZE + (i + 1) * expected_size + (i + 1) * BUFFER_SIZE + 16;

                    if single {
                        assert_eq!(
                            buffer[start..end].to_f32()[4],
                            3e-45,
                            "buffer is not what we expected {:?}",
                            buffer[start..end].to_f32()
                        );
                    } else {
                        if i == 0 {
                            println!("Double Precision Detected!")
                        };
                        assert_eq!(
                            buffer[start..end].to_f32()[4],
                            3e-45,
                            "buffer is not what we expected {:?}",
                            buffer[start..end].to_f32()
                        );
                    }
                }

                // Break up buffer into expected_size chunks
                let mut field_1: TimeSeries = Array1::from_vec(if single {
                    buffer[HEADER_SIZE..HEADER_SIZE + expected_size]
                        .to_f32()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                } else {
                    buffer[HEADER_SIZE..HEADER_SIZE + expected_size]
                        .to_f64()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                });
                let mut field_2: TimeSeries = Array1::from_vec(if single {
                    buffer[HEADER_SIZE + BUFFER_SIZE + expected_size
                        ..HEADER_SIZE + BUFFER_SIZE + 2 * expected_size]
                        .to_f32()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                } else {
                    buffer[HEADER_SIZE + BUFFER_SIZE + expected_size
                        ..HEADER_SIZE + BUFFER_SIZE + 2 * expected_size]
                        .to_f64()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                });
                let field_3: TimeSeries = Array1::from_vec(if single {
                    buffer[HEADER_SIZE + 2 * BUFFER_SIZE + 2 * expected_size
                        ..HEADER_SIZE + 2 * BUFFER_SIZE + 3 * expected_size]
                        .to_f32()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                } else {
                    buffer[HEADER_SIZE + 2 * BUFFER_SIZE + 2 * expected_size
                        ..HEADER_SIZE + 2 * BUFFER_SIZE + 3 * expected_size]
                        .to_f64()
                        .into_iter()
                        .map(|x| x as FloatType)
                        .collect()
                });

                if field_1.iter().any(|f1| f1.is_nan())
                    || field_2.iter().any(|f2| f2.is_nan())
                    || field_3.iter().any(|f3| f3.is_nan())
                {
                    log::info!("{station_name} has a nan, ignoring");
                    return None;
                }

                // Apply rotations to f1, f2
                // First, retrieve interpolator for this station
                let interpolator = IGRF_DATA_INTERPOLATOR.interpolator(&station_name);

                // Get offset
                let first_sec_of_year = day_since_first(0, year) * SECONDS_PER_DAY;
                // a few inline checks
                if year == 1998 {
                    assert_eq!(first_sec_of_year, 0)
                } else if year == 1999 {
                    assert_eq!(first_sec_of_year, 365 * SECONDS_PER_DAY)
                }

                // Apply rotations
                (field_1, field_2) = match field_1
                    .as_slice()
                    .unwrap()
                    .into_par_iter()
                    .zip(field_2.as_slice().unwrap())
                    .enumerate()
                    .map(|(sec, (&f1, &f2))| {
                        // Interpolate declination to this second
                        let sec_declination = interpolator
                            .interpolate_checked(
                                shift_point(first_sec_of_year + sec)
                                    .expect("something went wrong with chunk seconds"),
                            )
                            .expect(&format!(
                                "failed on station {station_name} on year {}",
                                year
                            ));
                        // Apply rotations
                        assert!(!f1.is_nan(), "f1 is nan");
                        assert!(!f2.is_nan(), "f2 is nan");
                        // these should be f32
                        let (f1_, f2_) = if f1 as f32 == SUPERMAG_NAN || f2 as f32 == SUPERMAG_NAN {
                            (SUPERMAG_NAN as FloatType, SUPERMAG_NAN as FloatType)
                        } else {
                            apply_rotation(f1, f2, sec_declination as FloatType)
                        };
                        assert!(!f1_.is_nan(), "rotated f1 is nan");
                        assert!(!f2_.is_nan(), "rotated f2 is nan");
                        (f1_ as FloatType, f2_ as FloatType)
                    })
                    .unzip()
                {
                    // Convert vecs to arrays
                    (vec1, vec2) => (Array1::from_vec(vec1), Array1::from_vec(vec2)),
                };

                // Find station name, which is HashMap key for Coordinates
                let station_name: String = station_file
                    .split("_")
                    .collect::<Vec<&str>>()
                    .get(1)
                    .expect("Non-standard station filename format")
                    .to_string();
                let coordinates: Coordinates =
                    self.coordinate_map.get(&station_name).map(|&x| x).unwrap();

                // Construct Dataset
                let dataset = Dataset {
                    field_1,
                    field_2,
                    field_3,
                    coordinates,
                    station_name,
                    index: year,
                };

                // format is raw/YEAR/YEAR_station_*
                // need third / element and second _ element
                let station_name = station_file
                    .split("/")
                    .skip(2)
                    .next()
                    .unwrap()
                    .split("_")
                    .skip(1)
                    .next()
                    .unwrap()
                    .to_string();

                // Increment counter and return
                pb.inc(1);
                Some((station_name, dataset))
            })
            .collect();
        pb.finish();
        map
    }

    /// Computes remaining number of files
    pub fn len(&self) -> usize {
        self.station_files.len()
    }
}

pub trait Tof64 {
    fn to_f64(&mut self) -> Vec<f64>;
}

impl Tof64 for [u8] {
    fn to_f64(&mut self) -> Vec<f64> {
        assert!(self.len() % 8 == 0, "Invalid buffer size for bytes -> f64");
        self.chunks(8)
            .map(|x| f64::from_be_bytes(x.try_into().unwrap()))
            .collect()
    }
}

// struct DailyDatasetLoader {
//     pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
//     pub station_files: Vec<String>,
//     pub day: Index,
// }

// impl DailyDatasetLoader {
//     /// Construct a new `DatasetLoader` given a `year`.
//     pub fn new_from_day(day: usize) -> Self {
//         // Construct coordinate map
//         let coordinate_map = Arc::new(construct_coordinate_map());

//         // Find all stations for which there is data for this day
//         let station_files = retrieve_stations_daily(day);

//         DailyDatasetLoader {
//             coordinate_map,
//             station_files,
//             day,
//         }
//     }

//     /// Changes the loader to a different day. (This skips construction of coordinate map,
//     /// which is rather cheap but I hate unnecessarily repeating calculations).
//     pub fn change_to_day(self, day: usize) -> Result<Self> {
//         // Find all stations for which there is data for this year
//         let station_files = retrieve_stations_daily(day);

//         if station_files.len() > 0 {
//             Ok(DailyDatasetLoader {
//                 coordinate_map: self.coordinate_map,
//                 station_files,
//                 day,
//             })
//         } else {
//             Err(anyhow::Error::msg("No stations found for this year"))
//         }
//     }

//     /// Loads the `Dataset` specified by the next file in the `station_files` vector.
//     pub async fn load_next(&mut self) -> Option<Dataset> {
//         // Return None if empty. Get next file if not.
//         if self.len() == 0 {
//             return None;
//         };
//         let station_file = self.station_files.pop().unwrap();

//         // Calculate expected size of each f32 fields, in bytes
//         let expected_size = 24 * 60 * 60 * 4;

//         // Open station data for this year
//         let mut file = tokio::fs::File::open(&station_file)
//             .await
//             .expect("couldn't open station file");

//         // Load file contents into buffer
//         let mut buffer = Vec::new();
//         file.read_to_end(&mut buffer)
//             .await
//             .expect("failed to load file contents into buffer");

//         // Ensure buffer size matches expectations
//         assert_eq!(
//             buffer.len(),
//             3 * expected_size,
//             "buffer size is not correct"
//         );

//         // Break up buffer into expected_size chunks
//         let [field_1, field_2, field_3]: [Array1<f32>; 3] = {
//             buffer
//                 .chunks_exact_mut(expected_size)
//                 .map(|x| arr1(&x.to_f32()))
//                 .collect::<Vec<TimeSeries>>()
//                 .try_into()
//                 .unwrap()
//         };

//         // Find station name, which is HashMap key for Coordinates
//         let station_name: String = station_file
//             .split("/")
//             .collect::<Vec<&str>>()
//             .get(2)
//             .expect("Non-standard station filename format")
//             .to_string();
//         println!("station_name = {station_name}");
//         let coordinates: Coordinates = self.coordinate_map.get(&station_name).map(|&x| x).unwrap();
//         println!("{:?}", field_1);

//         // Return Dataset
//         Some(Dataset {
//             field_1,
//             field_2,
//             field_3,
//             coordinates,
//             station_name,
//             index: self.day,
//         })
//     }

//     /// Computes remaining number of files
//     pub fn len(&self) -> usize {
//         self.station_files.len()
//     }
// }

// #[derive(Clone)]
// pub struct Chunk {
//     station: StationName,
//     files: Vec<PathBuf>,
//     start_sec: usize,
// }

// /// This `DatasetLoader` is intended to be used to gather 1s data in chunks (integer number of days).
// /// The struct contains all the metadata required to load chunks of data, and comes equipped with methods
// /// to load those chunks.
// pub struct DatasetLoader {
//     pub coordinate_map: Arc<HashMap<StationName, Coordinates>>,
//     /// A semivalid chunk is a set of days for which there exists a data file. It is semi-valid because
//     /// (at the very least) the data file exists, but the data file may still contain NaNs and thus be invalid.
//     pub semivalid_chunks: Arc<DashMap<Index, Vec<Chunk>>>,
//     stationarity: Stationarity,
//     // chunk: usize,
// }

// impl DatasetLoader {
//     // This function gathers all metadata required for loading a chunk of data
//     pub fn new(stationarity: Stationarity) -> Self {
//         // Construct coordinate map
//         let coordinate_map = Arc::new(construct_coordinate_map());
//         log::trace!("constructed coordinate map");

//         // Find all stations for which there is data for this year
//         let semivalid_chunks = retrieve_chunks(stationarity);
//         log::trace!("retrieved chunks");

//         Self {
//             coordinate_map,
//             semivalid_chunks,
//             stationarity,
//         }
//     }

//     /// This function loads and returns a chunk of data. Internally, it loads and combines the files
//     /// that comprise a chunk to return a single `TimeSeries` for each field. Here, `index` is the index
//     /// of the chunk, i.e. 0 is the first chunk.
//     pub async fn load_chunk(&self, index: Index) -> Result<DashMap<StationName, Dataset>> {
//         // Check for valid chunk index
//         match self.semivalid_chunks.get(&index) {
//             // Load data if valid
//             Some(chunks) => {
//                 // Initialize DashMap that contains the combined `Dataset` for every station for this chunk
//                 let result = DashMap::with_capacity(chunks.len());

//                 // Construct a collection of futures for every station/chunk
//                 let futs = chunks
//                     .iter()
//                     .map(|chunk| async {
//                         let station_name: String = chunk
//                             .station
//                             .split("/")
//                             .collect::<Vec<&str>>()
//                             .get(2)
//                             .unwrap()
//                             .to_string();
//                         // log::trace!("loading {station_name}/{index}");
//                         let chunk = _load_chunk(
//                             index,
//                             chunk.clone(),
//                             *self.coordinate_map.get(&station_name).unwrap(),
//                             &station_name,
//                         )
//                         .await;
//                         log::trace!("loaded {station_name}/{index}");
//                         chunk
//                     })
//                     .collect::<Vec<_>>();

//                 // Get all ordered, combined datasets
//                 let data: Vec<Dataset> = futures::stream::iter(futs).buffered(5).collect().await;
//                 println!("loaded all chunk datasets");

//                 for (chunk, data) in chunks.iter().zip(data) {
//                     let cleaned_station_name: String = chunk
//                         .station
//                         .split("/")
//                         .collect::<Vec<_>>()
//                         .get(2)
//                         .unwrap()
//                         .to_string();
//                     log::trace!(
//                         "inserting {cleaned_station_name} at sec {}",
//                         chunk.start_sec
//                     );
//                     assert!(
//                         result.insert(cleaned_station_name, data).is_none(),
//                         "duplicate entry"
//                     );
//                 }
//                 log::trace!("inserted all chunks into map");

//                 Ok(result)
//             }

//             // Return Err otherwise
//             None => Err(Error::msg("Invalid Index")),
//         }
//     }
// }

// async fn _load_chunk(
//     index: Index,
//     chunk: Chunk,
//     coordinates: Coordinates,
//     station: &str,
// ) -> Dataset {
//     // Construct a collection of futures for the daily datasets in this chunk for this station
//     let dataset_futures = chunk
//         .files
//         .iter()
//         .map(|filepath| load_daily(filepath.clone()))
//         .collect::<Vec<_>>();

//     // Execute futures concurrently on one thread with join_all
//     let mut datasets: Vec<[TimeSeries; 3]> = futures::stream::iter(dataset_futures)
//         .buffered(5)
//         .collect()
//         .await;

//     // Concatenate datasets
//     let [mut field_1, mut field_2, field_3]: [TimeSeries; 3] = {
//         let combined_dataset = datasets
//             .iter_mut()
//             .fold(Dataset::default(), |mut acc, dataset| {
//                 acc.field_1.append(Axis(0), dataset[0].view()).unwrap();
//                 acc.field_2.append(Axis(0), dataset[1].view()).unwrap();
//                 acc.field_3.append(Axis(0), dataset[2].view()).unwrap();
//                 acc
//             });
//         [
//             combined_dataset.field_1,
//             combined_dataset.field_2,
//             combined_dataset.field_3,
//         ]
//     };

//     // Rotate fields: this has to be done simultaneously otherwise
//     // new values will be used to rotate the second field.
//     //
//     // First get interpolator
//     let interpolator = IGRF_DATA_INTERPOLATOR.interpolator(station);

//     (field_1, field_2) = match field_1
//         .as_slice()
//         .unwrap()
//         .into_par_iter()
//         .zip(field_2.as_slice().unwrap())
//         .enumerate()
//         .map(|(sec, (&f1, &f2))| {
//             // Interpolate declination to this second
//             let sec_declination = interpolator
//                 .interpolate_checked(
//                     shift_point(chunk.start_sec + sec)
//                         .expect("something went wrong with chunk seconds"),
//                 )
//                 .expect(&format!(
//                     "failed on station {station} on chunk index {}",
//                     index
//                 ));
//             // Apply rotations
//             apply_rotation(f1, f2, sec_declination as f32)
//         })
//         .unzip()
//     {
//         // Convert vecs to arrays
//         (vec1, vec2) => (Array1::from_vec(vec1), Array1::from_vec(vec2)),
//     };

//     Dataset {
//         field_1,
//         field_2,
//         field_3,
//         station_name: chunk.station,
//         coordinates,
//         index,
//     }
// }

// async fn load_daily(filepath: PathBuf) -> [TimeSeries; 3] {
//     // Open file
//     // log::trace!("loading {}", filepath.display());
//     let mut file = tokio::fs::File::open(filepath)
//         .await
//         .expect("failed to open daily file");

//     // Load data
//     let mut buf = Vec::with_capacity(DAILY_FILE_SIZE);
//     file.read_to_end(&mut buf)
//         .await
//         .expect("failed to load file into buffer");

//     // Convert file to fields
//     let [field_1, field_2, field_3]: [TimeSeries; 3] = buf
//         .to_f32()
//         .chunks_exact_mut(SECONDS_PER_DAY)
//         .map(|field| arr1(field))
//         .collect::<Vec<TimeSeries>>()
//         .try_into()
//         .unwrap();

//     [field_1, field_2, field_3]
// }

fn retrieve_stations() -> Arc<HashMap<usize, Vec<String>>> {
    Arc::new(
        (1998..=2020)
            .map(|year| {
                let paths: Vec<String> = glob(format!("raw/{}/*.xdr", year).as_str())
                    .expect("couldn't find datasets")
                    .map(|x| x.unwrap().display().to_string())
                    .collect();
                println!("Found {} files for {year}", paths.len());
                (year, paths)
            })
            .collect(),
    )
}

// fn retrieve_chunks(stationarity: Stationarity) -> Arc<DashMap<Index, Vec<Chunk>>> {
//     // Iterate through every chunk of days
//     Arc::new(
//         stationarity
//             .get_chunks()
//             .into_par_iter()
//             .inspect(|chunk| println!("{chunk:?}"))
//             .enumerate()
//             .map(|(index, chunk)| {
//                 // Initialize vector which holds which stations contain semi-valid data for this chunk
//                 let chunks: Vec<Chunk> = STATIONS
//                     .par_iter()
//                     .filter_map(move |station| {
//                         // Check if files exist for that (station, chunk)
//                         // First, construct file paths.
//                         let files: Vec<PathBuf> = chunk
//                             .clone()
//                             .map(|day| PathBuf::from(format!("{}/{day}", station.display())))
//                             .collect();

//                         // Get the second at which this chunk begins
//                         let (start_sec, _) = stationarity.get_year_second_indices(index + 1998);

//                         // Then check if they all exist
//                         if files.par_iter().all(|file| STATION_DAYS.contains(file)) {
//                             // If so, return `Chunk`
//                             Some(Chunk {
//                                 station: station.display().to_string(),
//                                 files,
//                                 start_sec,
//                             })
//                         } else {
//                             // Otherwise, don't include this station for this chunk
//                             None
//                         }
//                     })
//                     .collect();
//                 log::trace!("index {index} has {} semivalid elements", chunks.len());
//                 (index, chunks)
//             })
//             .collect(),
//     )
// }

// fn retrieve_stations_daily(day: usize) -> Vec<String> {
//     let paths: Vec<String> = glob(format!("../stations/*/{}", day).as_str())
//         .expect("couldn't find datasets")
//         .map(|x| x.unwrap().display().to_string())
//         .collect();
//     println!("Found {} files for {day}", paths.len());
//     paths
// }

type SinglePrecision = bool;
fn validate_buffer_size(
    buffer: &[u8],
    expected_size: &mut usize,
    year: usize,
) -> Result<SinglePrecision, String> {
    // Different years have different header/buffer sizes
    if year != 2018 && year != 2019 {
        // Check for single precision
        if buffer.len() == HEADER_SIZE + 3 * *expected_size + 2 * BUFFER_SIZE + END_SIZE {
            return Ok(true);

        // Check for double precision
        } else if buffer.len() == HEADER_SIZE + 3 * 2 * *expected_size + 2 * BUFFER_SIZE + END_SIZE
        {
            *expected_size *= 2;
            return Ok(false);

        // Otherwise something is wrong
        } else {
            return Err(format!(
                "invalid buffer size: size diff is {}",
                buffer.len() as i64
                    - (HEADER_SIZE + 3 * *expected_size + 2 * BUFFER_SIZE + END_SIZE) as i64
            ));
        }
    } else {
        // Check for single precision
        if buffer.len() == HEADER_SIZE + 3 * *expected_size + 2 * BUFFER_SIZE + END_SIZE_2018_2019 {
            return Ok(true);

        // Check for double precision
        } else if buffer.len()
            == HEADER_SIZE + 3 * 2 * *expected_size + 2 * BUFFER_SIZE + END_SIZE_2018_2019
        {
            *expected_size *= 2;
            return Ok(false);

        // Otherwies something is wrong
        } else {
            return Err(format!(
                "invalid buffer size: size diff is {}",
                buffer.len() as i64
                    - (HEADER_SIZE + 3 * *expected_size + 2 * BUFFER_SIZE + END_SIZE_2018_2019)
                        as i64
            ));
        }
    }
}

/// This is a recursive function that returns the number of days since the first day there was data.
pub const fn day_since_first(day: usize, year: usize) -> usize {
    match year {
        1998 => day,
        1999.. => {
            // Calculate how many days there were in the last year
            let days_in_last_year = if (year - 1) % 4 == 0 { 366 } else { 365 };

            return day + day_since_first(days_in_last_year, year - 1);
        }
        _ => panic!("the year provided is before there was any data"),
    }
}

#[test]
fn test_day_since_first() {
    type Day = usize;
    type Year = usize;

    // 1st day ever (0th day)
    assert_eq!(0, day_since_first(0, 1998));

    // 101st day of the first year (100th day, zero-indexed)
    assert_eq!(100, day_since_first(100, 1998));

    // First day of the second year
    assert_eq!(365, day_since_first(0, 1999));

    // Testing C08_4029 (which was apparently wrongly indexed)
    assert_eq!(4018, day_since_first(0, 2009));

    // The day this unit test was written
    // (182nd day of 2022, zero-indexed)
    const TODAY: (Day, Year) = (181, 2022);
    assert_eq!(8947, day_since_first(TODAY.0, TODAY.1));
}

/// Simple trait which converts all values in a collection to `f32`.
pub trait Tof32 {
    fn to_f32(&mut self) -> Vec<f32>;
}

/// For a given collection of bytes (`u8`), convert to a collection of floats (`f32`).
impl Tof32 for [u8] {
    fn to_f32(&mut self) -> Vec<f32> {
        assert!(self.len() % 4 == 0, "Invalid buffer size for bytes -> f32");
        self.chunks(4)
            .map(|x| f32::from_be_bytes(x.try_into().unwrap()))
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
            coordinates: Coordinates {
                latitude: 0.0,
                longitude: 0.0,
                polar: std::f64::consts::FRAC_PI_2,
            },
            index: 0,
        }
    }
}
