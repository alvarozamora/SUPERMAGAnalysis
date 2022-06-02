

pub trait Theory<const T: usize, const S: usize> {

    // This calculates the X^n_i's for a given theory
    fn calculate_data_vec(&self) -> [[f64; T]; S];

}


pub struct DarkPhoton {
    kinetic_mixing: f64,
}