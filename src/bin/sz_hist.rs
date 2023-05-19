use nalgebra::Complex;
use ndarray::{Array1, ArrayBase};
use plotters::prelude::*;
use rocksdb::DB;
use statrs::distribution::ContinuousCDF;

const OUT_FILE_NAME: &str = "szhist_2.png";
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(OUT_FILE_NAME, (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut z_chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("CDF of z", ("sans-serif", 50.0))
        .build_cartesian_2d(-2.5f32..2.5f32, 0f32..1f32)?;

    z_chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("CDF")
        .x_desc("z")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let zdb = DB::open_default("z")?;
    // let mut z: Vec<f32> = zdb
    //     .iterator(rocksdb::IteratorMode::Start)
    //     .map(Result::unwrap)
    //     .map(|(_key, value)| bincode::deserialize::<Array1<Complex<f32>>>(&value).unwrap()[0])
    //     // .flatten()
    //     .map(|x: Complex<f32>| x.re)
    //     // .inspect(|z| println!("{z}"))
    //     .collect();
    // let mut z: Vec<f32> = (0..59126)
    //     .map(|window_index| {
    //         let key = [10198740_usize, 20_usize, window_index]
    //             // let key = [8541288_usize, 10_usize, window_index]
    //             .map(usize::to_le_bytes)
    //             .concat();
    //         bincode::deserialize::<Array1<Complex<f32>>>(&zdb.get(key).unwrap().unwrap())
    //             .unwrap()
    //             .map(|z| z.re)
    //     })
    //     .flatten()
    //     .collect();
    let mut z: Vec<f32> = (0..59126)
        .map(|window_index| {
            let key = [10_198_740_usize, 20_usize, window_index]
                // let key = [8541288_usize, 10_usize, window_index]
                .map(usize::to_le_bytes)
                .concat();
            bincode::deserialize::<Array1<Complex<f32>>>(&zdb.get(key).unwrap().unwrap()).unwrap()
                [2]
            .re
        })
        .collect();
    z.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
    let cdfz: Vec<f32> = (1..=z.len()).map(|i| i as f32 / z.len() as f32).collect();

    // let sdb = DB::open_default("s")?;
    // let s: Vec<f32> = sdb
    //     .iterator(rocksdb::IteratorMode::Start)
    //     .map(|bytes| bincode::deserialize(&bytes.unwrap().1).unwrap())
    //     .flatten()
    //     .collect();
    // TODO: min, max, median, mean

    z_chart.draw_series(LineSeries::new(
        z.clone().into_iter().zip(cdfz.clone()),
        RED.mix(0.8).filled(),
    ))?;

    let normal = statrs::distribution::Normal::new(0.0, 0.33 / 2.0_f64.sqrt()).unwrap();
    z_chart.draw_series(LineSeries::new(
        z.into_iter().map(|z| (z, normal.cdf(z as f64) as f32)),
        GREEN.mix(0.8).filled(),
    ))?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}
