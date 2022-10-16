use std::{error::Error, path::{PathBuf, Path}};

use rocksdb::{MultiThreaded as Parallel, DBWithThreadMode, IteratorMode};

use crate::theory::dark_photon::{InnerVarChunkWindowMap, Triplet};

pub struct DiskDB {
    db: DBWithThreadMode<Parallel>,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

impl DiskDB {

    pub(crate) fn connect(
        path: impl AsRef<Path>
    ) -> Result<DiskDB> {

        // Attempt to read existing databse
        let db = DBWithThreadMode::<Parallel>::open_default(path.as_ref())?;

        Ok(DiskDB {
            db,
        })
    }


    pub(crate) fn get_windows(
        &self,
        coherence_time: usize
    ) -> Result<Option<InnerVarChunkWindowMap>> {

        match self.db.get(coherence_time.to_le_bytes()) {

            // Map Present
            Ok(Some(map_bytes)) => Ok(Some(serde_cbor::from_slice(&map_bytes)?)),

            // Map Not Present
            Ok(None) => Ok(None),

            // RocksDBError
            Err(err) => Err(err.into())
        }
    }

    pub(crate) fn insert_windows(
        &self,
        coherence_time: usize,
        map: &InnerVarChunkWindowMap
    ) -> Result<()> {
        Ok(self.db.put(
            coherence_time.to_le_bytes(),
            serde_cbor::to_vec(map)?
        )?)
    }

}

#[tokio::test]
async fn test_connect_add_change() {

    // Connect
    let disk_db: DiskDB = DiskDB::connect("./diskdb/")
        .expect("connection to db failed");

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
    disk_db.insert_windows(
        0, 
        &[
            (
                0,
                [(Triplet::Low, ndarray::Array2::zeros((1,1)))].into_iter().collect()
            )
        ].into_iter().collect()
    ).unwrap();
    assert_eq!(
        disk_db
            .get_windows(0)
            .expect("db retrieval error")
            .expect("entry not found")
            .len(),
        1,
    );
    
}