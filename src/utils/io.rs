use std::{error::Error, path::Path};

use dashmap::DashMap;
use rocksdb::{DBWithThreadMode, MultiThreaded, Options};

use crate::theory::dark_photon::{InnerVarChunkWindowMap, Triplet};

pub struct DiskDB {
    db: DBWithThreadMode<MultiThreaded>,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

impl DiskDB {
    pub(crate) fn connect(path: impl AsRef<Path>) -> Result<DiskDB> {
        let mut opts = Options::default();
        opts.set_target_file_size_base(128 * 1024 * 1024);
        opts.set_target_file_size_multiplier(2);
        opts.set_max_bytes_for_level_base(32 * 1024 * 1024 * 1024);
        opts.set_max_bytes_for_level_multiplier(1.1);
        opts.create_if_missing(true);

        let db = DBWithThreadMode::<MultiThreaded>::open(&opts, path.as_ref())?;

        Ok(DiskDB { db })
    }

    pub(crate) fn get_chunk_window_map(
        &self,
        coherence_time: usize,
    ) -> Result<Option<InnerVarChunkWindowMap>> {
        let map = self
            .db
            .iterator(rocksdb::IteratorMode::Start)
            .map(|value| value.expect("rocksdb read error"))
            .flat_map(|(key, value)| {
                let read_coherence_time = usize::from_le_bytes(key[..8].try_into().unwrap());
                if coherence_time == read_coherence_time {
                    let read_chunk = usize::from_le_bytes(key[8..16].try_into().unwrap());
                    let read_window = usize::from_le_bytes(key[16..].try_into().unwrap());
                    let read_triplet: Triplet = bincode::deserialize(&value).unwrap();
                    Some(((read_chunk, read_window), read_triplet))
                } else {
                    None
                }
            })
            .collect::<DashMap<_, _>>();
        Ok(if map.len() == 0 { None } else { Some(map) })
    }

    pub(crate) fn get_chunk_window(
        &self,
        coherence_time: usize,
        chunk: usize,
        window: usize,
    ) -> Result<Option<Triplet>> {
        let key = Self::get_key(coherence_time, chunk, window);
        Ok(self
            .db
            .get(key)?
            .map(|bytes| bincode::deserialize(&bytes).unwrap()))
    }

    pub(crate) fn store_chunk_window_map(
        &self,
        coherence_time: usize,
        chunk: usize,
        window: usize,
        map: &Triplet,
    ) -> Result<()> {
        let key = Self::get_key(coherence_time, chunk, window);
        Ok(self.db.put(key, bincode::serialize(map)?)?)
    }

    fn get_key(coherence_time: usize, chunk: usize, window: usize) -> [u8; 24] {
        let mut key = [0; 24];
        for i in 0..24 {
            if i < 8 {
                key[i] = coherence_time.to_le_bytes()[i];
            } else if i < 16 {
                key[i] = chunk.to_le_bytes()[i % 8];
            } else {
                key[i] = window.to_le_bytes()[i % 8];
            }
        }
        key
    }
}

#[tokio::test]
async fn test_connect_add_change() {
    use ndarray::Array2;

    use crate::theory::dark_photon::Triplet;

    const TEST_DB: &'static str = "__test__diskdb";

    // Connect
    let disk_db: DiskDB = DiskDB::connect(TEST_DB).expect("connection to db failed");

    // Add
    disk_db
        .store_chunk_window_map(
            0,
            0,
            0,
            &Triplet {
                low: Array2::zeros((0, 0)),
                mid: Array2::zeros((0, 0)),
                high: Array2::zeros((0, 0)),
                midf: 1.1,
            },
        )
        .unwrap();
    assert_eq!(
        disk_db
            .get_chunk_window_map(0)
            .expect("db retrieval error")
            .expect("entry not found")
            .len(),
        1,
    );

    // Change
    disk_db
        .store_chunk_window_map(
            0,
            0,
            0,
            &Triplet {
                low: Array2::zeros((0, 0)),
                mid: Array2::zeros((0, 0)),
                high: Array2::zeros((0, 0)),
                midf: 1.2,
            },
        )
        .unwrap();
    assert_eq!(
        disk_db
            .get_chunk_window_map(0)
            .expect("db retrieval error")
            .expect("entry not found")
            .len(),
        1,
    );

    std::fs::remove_dir_all(TEST_DB).unwrap();
}
