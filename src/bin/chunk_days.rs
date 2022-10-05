//! A self-contained script that reads in the yearly datasets and breaks them up
//! into more manageable daily chunks.
use tokio::io::AsyncReadExt;
use tokio::fs::File;
use tokio::task::spawn;
use std::convert::TryInto;
use glob::glob;

use futures::prelude::*;



/// Size of first header in bytes
const HEADER_SIZE: usize = 315 * 4;

/// Size of intra-field buffer
const BUFFER_SIZE: usize = 25*4;

/// Size of final buffer
const END_SIZE: usize = 48*4;

/// Size of fiinal buffer (2018-2019)
const END_SIZE_2018_2019: usize = 29*4;

/// Seconds in a day
const DAY_IN_SECONDS: usize = 24 * 60 * 60;

const BYTES_PER_FLOAT: usize = 4;
const NUM_FIELDS: usize = 3;

fn main() {

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(32)
        .enable_all()
        .build()
        .unwrap();

    let now = std::time::Instant::now();

    rt.block_on(async {
        chunk_into_days().await
    });

    println!("Chunking took {} seconds", now.elapsed().as_secs());
}

async fn chunk_into_days() {

    // Years for which there is data
    let years = 1998..=2020;

    // Futures collector
    let mut futures = vec![];

    // Iterate through years
    for year in years {
        
        // Find all stations for which there is data for this year
        let stations = retrieve_stations(year);
    
        // For every year, iterate through stations
        for station_file in stations.into_iter() {
            // futures.push(process_station_year(station_file, year));
            futures.push(async move { spawn(process_station_year(station_file, year)).await });
        }

    }

    // Turn into futures stream
    let stream = futures::stream::iter(futures).buffered(32*5);
    
    // Wait for all futures to complete
    assert!(
        stream.collect::<Vec<_>>()
            .await
            .iter()
            .all(|x| x.as_ref().unwrap() == &())
    );
}

async fn process_station_year(station_file: String, year: usize) {

    println!("{:?} is starting {station_file}", std::thread::current().id());

    // Calculate expected size of f32 fields, in bytes
    let expected_size = if year % 4 == 0 {
        // Leap year
        60 * 60 * 24 * 366 * 4
    } else {
        // Non-leap year
        60 * 60 * 24 * 365 * 4
    };

    // Open station data for this year
    let mut file = File::open(&station_file).await.expect(format!("couldn't open station_file {station_file}").as_str());
    println!("{:?} opened {station_file}", std::thread::current().id());


    // Load file contents into buffer
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await.expect("failed to load file contents into buffer");
    println!("{:?} loaded {station_file}", std::thread::current().id());


    // Ensure buffer size matches expectations
    let single: SinglePrecision = validate_buffer_size(&buffer, expected_size, year).unwrap();
    let factor = if single { 1 } else { 2 };

    // Verification, 3e-45 f32 seems to be "end of data"; 
    for i in 0..3 {
        let start = HEADER_SIZE + (i+1)*factor*expected_size + i*BUFFER_SIZE-16;
        let end = HEADER_SIZE + (i+1)*factor*expected_size + (i+1)*BUFFER_SIZE + 16;
        // println!("{:?}", String::from_utf8_lossy(&buffer[start..end]));
        // if i != 2 { println!("Header {i}:\n{:?}", buffer[start..end].to_f32()) } else { println!("Header {i}:\n{:?}", buffer[start..].to_f32())};
        if single {
            assert_eq!(buffer[start..end].to_f32()[4], 3e-45, "buffer is not what we expected {:?}", buffer[start..end].to_f32());
        } else {
            println!("Double Precision Detected!");
            assert_eq!(buffer[start..end].to_f32()[4], 3e-45, "buffer is not what we expected {:?}", buffer[start..end].to_f32());
        }
    }

    // Break up buffer into day_long chunks
    let field_1: Vec<f32> = if single {
        buffer[HEADER_SIZE..HEADER_SIZE+expected_size].to_f32()
    } else {
        buffer[HEADER_SIZE..HEADER_SIZE+expected_size].to_f64()
        .iter()
        .map(|&x| x as f32)
        .collect()
    };
    let field_2: Vec<f32> = if single {
        buffer[HEADER_SIZE+BUFFER_SIZE+expected_size..HEADER_SIZE+BUFFER_SIZE+2*expected_size].to_f32()
    } else {
        buffer[HEADER_SIZE+BUFFER_SIZE+expected_size..HEADER_SIZE+BUFFER_SIZE+2*expected_size].to_f64()
        .iter()
        .map(|&x| x as f32)
        .collect()
    };
    let field_3: Vec<f32> = if single {
        buffer[HEADER_SIZE+2*BUFFER_SIZE+2*expected_size..HEADER_SIZE+2*BUFFER_SIZE+3*expected_size].to_f32()
    } else {
        buffer[HEADER_SIZE+2*BUFFER_SIZE+2*expected_size..HEADER_SIZE+2*BUFFER_SIZE+3*expected_size].to_f64()
        .iter()
        .map(|&x| x as f32)
        .collect()
    };

    // Chunk the year-long chunks into day chunks and append
    let mut field_1 = field_1.chunks_exact(DAY_IN_SECONDS);
    let mut field_2 = field_2.chunks_exact(DAY_IN_SECONDS);
    let mut field_3 = field_3.chunks_exact(DAY_IN_SECONDS);

     // Check that chunks have no remainder
     assert_eq!(field_1.remainder().len(), 0);
     assert_eq!(field_2.remainder().len(), 0);
     assert_eq!(field_3.remainder().len(), 0);

     // Check that chunks are of the same len
     assert_eq!(field_1.len(), field_2.len());
     assert_eq!(field_1.len(), field_3.len());

     // Iterate through chunks, append to temporary return variable
     for day in 0..field_1.len() {

        // Initialize temporary vector for this day
        let mut day_vec: Vec<u8> = vec![];
       
            field_1
                .next()
                .unwrap()
                .iter()
                .for_each(|x| {
                    day_vec.append(&mut x.to_le_bytes().to_vec());
            });
            field_2
                .next()
                .unwrap()
                .iter()
                .for_each(|x| {
                    day_vec.append(&mut x.to_le_bytes().to_vec());
            });
            field_3
                .next()
                .unwrap()
                .iter()
                .for_each(|x| {
                    day_vec.append(&mut x.to_le_bytes().to_vec());
            });
    
        // Check len of temp is correct
        assert_eq!(day_vec.len(), DAY_IN_SECONDS * BYTES_PER_FLOAT * NUM_FIELDS);

        // Construct path/filename
        let station_name: String = station_file
            .split("_")
            .collect::<Vec<&str>>()
            .get(1)
            .expect("Non-standard station filename format")
            .to_string();
        let station_dir = format!("../stations/{station_name}");
        let day: usize = day_since_first(day, year);
        let day_file: String = format!("{station_dir}/{day}");
        
        println!("{year}: {:?} saving {day_file}", std::thread::current().id());
        tokio::fs::create_dir_all(station_dir).await.expect("failed to create directory for station");
        tokio::fs::write(day_file, day_vec).await.expect("failed to write to disk");
    };

    // Check chunks are empty
    assert!(field_1.next().is_none(), "Something went wrong, as there are still days of data to iterate through");
    assert!(field_2.next().is_none(), "Something went wrong, as there are still days of data to iterate through");
    assert!(field_3.next().is_none(), "Something went wrong, as there are still days of data to iterate through");
}


// fn convert_to_npy() -> Result<()> {

//     // Years for which there is data
//     let years = 1998..=2020;

//     // Iterate through years
//     for year in years {
        
//         // Find all stations for which there is data for this year
//         let stations = retrieve_stations(year);

//         // Calculate expected size of f32 fields, in bytes
//         let expected_size = if year % 4 == 0 {
//             // Leap year
//             60 * 60 * 24 * 366 * 4
//         } else {
//             // Non-leap year
//             60 * 60 * 24 * 365 * 4
//         };

//         // For every year, iterate through stations
//         for station_file in stations.into_iter() {

//             // Open station data for this year
//             let mut file = File::open(&station_file).expect("couldn't open station file");

//             // Load file contents into buffer
//             let mut buffer = Vec::new();
//             file.read_to_end(&mut buffer).expect("failed to load file contents into buffer");

//             // Verification, 3e-45 seems to be "end of data"
//             for i in 0..3 {
//                 let start = HEADER_SIZE + (i+1)*expected_size + i*BUFFER_SIZE-16;
//                 let end = HEADER_SIZE + (i+1)*expected_size + (i+1)*BUFFER_SIZE + 16;
//                 // println!("{:?}", String::from_utf8_lossy(&buffer[start..end]));
//                 // if i != 2 { println!("Header {i}:\n{:?}", buffer[start..end].to_f32()) } else { println!("Header {i}:\n{:?}", buffer[start..].to_f32())};
//                 assert_eq!(buffer[start..end].to_f32()[4], 3e-45);
//             }

//             // Ensure buffer size matches expectations
//             validate_buffer_size(&buffer, expected_size, year);

//             // Break up buffer into expected_size chunks
//             //let _header: Vec<u8> = buffer[0..HEADER_SIZE].to_vec();
//             let field_1: Vec<f32> = buffer[HEADER_SIZE..HEADER_SIZE+expected_size].to_f32();
//             let field_2: Vec<f32> = buffer[HEADER_SIZE+BUFFER_SIZE+expected_size..HEADER_SIZE+BUFFER_SIZE+2*expected_size].to_f32();
//             let field_3: Vec<f32> = buffer[HEADER_SIZE+2*BUFFER_SIZE+2*expected_size..HEADER_SIZE+BUFFER_SIZE+3*expected_size].to_f32();
            
//             // Convert to ndarrays
//             let field_1: Array1<f32> = arr1(&field_1);
//             let field_2: Array1<f32> = arr1(&field_2);
//             let field_3: Array1<f32> = arr1(&field_3);

//             // Save to disk
//             let mut npz = NpzWriter::new(File::create(station_file.replace(".xdr",".npz"))?);
//             npz.add_array("field_1", &field_1)?;
//             npz.add_array("field_2", &field_2)?;
//             npz.add_array("field_3", &field_3)?;
//             npz.finish()?;

//         }
//     }
//     Ok(())
// }





fn retrieve_stations(year: usize) -> Vec<String> {
    let paths: Vec<String> = glob(format!("../{}/*.xdr", year).as_str())
        .expect("couldn't find datasets")
        .map(|x| x.unwrap().display().to_string())
        .collect();
    println!("Found {} paths for {year}", paths.len());
    paths
}

type SinglePrecision = bool;
fn validate_buffer_size(buffer: &[u8], expected_size: usize, year: usize) -> Result<SinglePrecision, String> {

    // Different years have different header/buffer sizes
    if year != 2018 && year != 2019 {

        // Check for single precision
        if buffer.len() == HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE {
            return Ok(true)

        // Check for double precision
        } else if buffer.len() == HEADER_SIZE + 3*2*expected_size + 2*BUFFER_SIZE + END_SIZE {
            return Ok(false)

        // Otherwise something is wrong
        } else {
            return Err(format!(
                "invalid buffer size: size diff is {}",
                buffer.len() as i64 - (HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE) as i64
            ))
        }

    } else {

        // Check for single precision
        if buffer.len() == HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE_2018_2019 {
            return Ok(true)

        // Check for double precision
        } else if buffer.len() == HEADER_SIZE + 3*2*expected_size + 2*BUFFER_SIZE + END_SIZE_2018_2019 {
            return Ok(false)

        // Otherwies something is wrong
        } else {
            return Err(format!(
                "invalid buffer size: size diff is {}",
                buffer.len() as i64 - (HEADER_SIZE + 3*expected_size + 2*BUFFER_SIZE + END_SIZE_2018_2019) as i64
            ))
        }
    }
}


pub trait Tof32{
    fn to_f32(&mut self) -> Vec<f32>;
}

impl Tof32 for [u8] {
    fn to_f32(&mut self) -> Vec<f32> {
        assert!(self.len() % 4 == 0, "Invalid buffer size for bytes -> f32");
        self
            .chunks(4)
            .map(|x| f32::from_be_bytes(x.try_into().unwrap()))
            .collect()
    }
}

pub trait Tof64{
    fn to_f64(&mut self) -> Vec<f64>;
}

impl Tof64 for [u8] {
    fn to_f64(&mut self) -> Vec<f64> {
        assert!(self.len() % 8 == 0, "Invalid buffer size for bytes -> f64");
        self
            .chunks(8)
            .map(|x| f64::from_be_bytes(x.try_into().unwrap()))
            .collect()
    }
}


/// This is a recursive function that returns the number of days since the first day there was data.
fn day_since_first(day: usize, year: usize) -> usize {

    match year {
        1998 => day,
        1999.. => {

            // Calculate how many days there were in the last year
            let days_in_last_year = if ( year - 1 ) % 4 == 0 { 366 } else { 365 };

            return day + day_since_first(days_in_last_year, year - 1)
        },
        _ => panic!("the year provided is before there was any data")
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