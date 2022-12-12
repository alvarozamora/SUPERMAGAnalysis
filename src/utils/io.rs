use std::{error::Error, path::Path};

use rocksdb::{DBWithThreadMode, MultiThreaded as Parallel};

use crate::theory::dark_photon::InnerVarChunkWindowMap;

pub struct DiskDB {
    db: DBWithThreadMode<Parallel>,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

impl DiskDB {
    pub(crate) fn connect(path: impl AsRef<Path>) -> Result<DiskDB> {
        // Attempt to read existing databse
        let db = DBWithThreadMode::<Parallel>::open_default(path.as_ref())?;

        Ok(DiskDB { db })
    }

    pub(crate) fn get_windows(
        &self,
        coherence_time: usize,
    ) -> Result<Option<InnerVarChunkWindowMap>> {
        match self.db.get(coherence_time.to_le_bytes()) {
            // Map Present
            Ok(Some(map_bytes)) => Ok(Some(serde_cbor::from_slice(&map_bytes)?)),

            // Map Not Present
            Ok(None) => Ok(None),

            // RocksDBError
            Err(err) => Err(err.into()),
        }
    }

    pub(crate) fn insert_windows(
        &self,
        coherence_time: usize,
        map: &InnerVarChunkWindowMap,
    ) -> Result<()> {
        Ok(self
            .db
            .put(coherence_time.to_le_bytes(), serde_cbor::to_vec(map)?)?)
    }
}

#[tokio::test]
async fn test_connect_add_change() {
    use ndarray::Array2;

    use crate::theory::dark_photon::Triplet;

    // Connect
    let disk_db: DiskDB = DiskDB::connect("./diskdb/").expect("connection to db failed");

    // Add
    disk_db.insert_windows(0, &dashmap::DashMap::new()).unwrap();
    assert_eq!(
        disk_db
            .get_windows(0)
            .expect("db retrieval error")
            .expect("entry not found")
            .len(),
        0,
    );

    // Change
    disk_db
        .insert_windows(
            0,
            &[(
                0,
                Triplet {
                    low: Array2::zeros((0, 0)),
                    mid: Array2::zeros((0, 0)),
                    high: Array2::zeros((0, 0)),
                    // lowf: 1.0,
                    midf: 1.1,
                    // hif: 1.2,
                    // coh_time: 0,
                    // window: Some(0),
                    // chunk: 0,
                },
            )]
            .into_iter()
            .collect(),
        )
        .unwrap();
    assert_eq!(
        disk_db
            .get_windows(0)
            .expect("db retrieval error")
            .expect("entry not found")
            .len(),
        1,
    );
}
