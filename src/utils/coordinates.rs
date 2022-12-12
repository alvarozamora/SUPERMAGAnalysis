use csv;
use serde_derive::Deserialize;
use std::collections::HashMap;

type StationName = String;

#[derive(Debug, Deserialize)]
pub struct StationCsvEntry {
    name: StationName,
    longitude: f64,
    latitude: f64,
}

#[derive(Debug, Deserialize, PartialEq, Copy, Clone)]
pub struct Coordinates {
    pub longitude: f64,
    pub latitude: f64,
    pub polar: f64,
}

/// Location of csv file containing station coordinates
pub const COORD_CSV_FILEPATH: &str = "./src/utils/coordinates.csv";

/// This function reads in the .csv file containing the station coordinates.
/// This constructor function is preferred over hardcoding station coordinates
/// in case an updated file is used in the future.
pub fn construct_coordinate_map() -> HashMap<StationName, Coordinates> {
    // Load coordinates.csv file
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(COORD_CSV_FILEPATH)
        .expect(format!("Failed to open file at {}", COORD_CSV_FILEPATH).as_str());

    // Initialize Hashmap
    let mut hash_map = HashMap::new();

    // Add station coordinates to hashmap
    for row in reader.deserialize::<StationCsvEntry>() {
        // Convert the (name, f64, f64) to (name, coordinate)
        let (name, coordinate) = row.expect("CSV file row format is invalid").to_map_entry();

        // .insert() returns None if the key was not already in the hash map
        assert!(
            hash_map.insert(name, coordinate).is_none(),
            "Attempted to insert duplicate station"
        );
    }

    hash_map
}

#[test]
fn construct_coord_map() {
    // Construct the hash map
    let hash_map = construct_coordinate_map();

    // Check if the first entry matches. Will obviously fail if the first station in the file changes in the future
    assert_eq!(
        hash_map.get("A01").unwrap(),
        &Coordinates {
            longitude: 7.39,
            latitude: 8.99,
            polar: std::f64::consts::FRAC_PI_2 - 8.99
        },
        "Failed to construct A01 properly based on CSV as of May 23, 2022"
    );
}

trait ToMapEntry {
    fn to_map_entry(self) -> (StationName, Coordinates);
}

impl ToMapEntry for StationCsvEntry {
    fn to_map_entry(self) -> (StationName, Coordinates) {
        (
            self.name,
            Coordinates {
                longitude: self.longitude,
                latitude: self.latitude,
                polar: std::f64::consts::FRAC_PI_2 - self.latitude,
            },
        )
    }
}
