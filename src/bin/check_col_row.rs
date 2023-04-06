use ndarray::array;

fn main() {
    let test_array = array![1., 2., 3., 4., 5., 6., 7., 8.];

    let coherence_time = 4;
    let chunks = 2;

    let reshaped_array = test_array.into_shape((coherence_time, chunks)).unwrap();
    println!("reshaped into {reshaped_array:?}");
    println!("first row: {:?}", reshaped_array.column(0));
}
