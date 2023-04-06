use std::{collections::HashSet, error::Error};

use rocksdb::{IteratorMode, Options, DB};

fn main() -> Result<(), Box<dyn Error>> {
    let var = DB::open(&Options::default(), "dark_photon_theory_var")?;

    let mut set = HashSet::new();
    for (k, _) in var.iterator(IteratorMode::Start).map(Result::unwrap) {
        let time = usize::from_le_bytes(k[..8].try_into().unwrap());
        if set.insert(time) {
            println!("found {time}");
        }
    }

    println!("found {} times: {:?}", set.len(), set);

    Ok(())
}
