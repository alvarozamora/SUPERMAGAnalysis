use std::{error::Error, io::Write};

use dashmap::DashMap;
use itertools::Itertools;
use nalgebra::Complex;
use ndarray::Array1;
use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, IntoDrawingArea, IntoLinspace, IntoLogRange,
        IntoSegmentedCoord,
    },
    series::{Histogram, LineSeries},
    style::{full_palette::PURPLE, Color, BLACK, BLUE, GREEN, RED, WHITE},
};
use statrs::distribution::ContinuousCDF;
use supermag_analysis::theory::{
    dark_photon::{Power, DARK_PHOTON_NONZERO_ELEMENTS},
    NonzeroElement,
};

const OUT_FILE_NAME: &str = "test_power_plot.png";
// const OUT_FILE_NAME: &str = "test_power_cdf.png";
fn main() -> Result<(), Box<dyn Error>> {
    let test_power_db = rocksdb::DB::open_default("test_power")?;

    let root = BitMapBackend::new(OUT_FILE_NAME, (1080, 720)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut z_chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .margin(10)
        .caption("Power with varying tau", ("sans-serif", 50.0))
        .build_cartesian_2d((0.01..1.0_f32).log_scale(), (1e3f32..1e12f32).log_scale())?;
    // .build_cartesian_2d((1e3f32..1e12f32).log_scale(), (0.01..5.0_f32).log_scale())?;
    // .set_secondary_coord(
    //     (0.0..1e13f32).step(5e11).use_round().into_segmented(),
    //     0u32..500u32,
    // );

    z_chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Power")
        .x_desc("z")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let half_window_size = 512 * 32;
    for ((tau, color), hws) in [
        // 16384_usize * 64,
        // 16384 * 64 / 4,
        16384_usize * 64 / 8,
        16384 * 64 / 16,
        16384 * 64 / 32,
        16384 * 64 / 64,
        // 16384 * 64 / 128,
        // 16384 * 64 / 256,
    ]
    .into_iter()
    .zip([RED, GREEN, BLUE, PURPLE, BLACK])
    .zip([
        // half_window_size,
        // half_window_size / 4,
        half_window_size / 8,
        half_window_size / 16,
        half_window_size / 32,
        half_window_size / 64,
        // half_window_size / 128,
        // half_window_size / 256,
    ]) {
        let year = 18;
        let [t1, t2, t3, t4, t5, t6, t7, t8] = tau.to_le_bytes();
        let key = [t1, t2, t3, t4, t5, t6, t7, t8, year];
        let power_2013: DashMap<(NonzeroElement, NonzeroElement), Power<f32>> =
            bincode::deserialize(&test_power_db.get(key)?.unwrap())?;

        let df = ((2 * tau) as f32).recip();
        let freq: Vec<f32> = (0..2 * tau).map(|mul| mul as f32 * df).collect();
        let window_freqs = freq.windows(2 * hws + 1).map(|fwindow| fwindow[hws]);

        let element_00 = power_2013
            .get(&(
                DARK_PHOTON_NONZERO_ELEMENTS[2],
                DARK_PHOTON_NONZERO_ELEMENTS[2],
            ))
            .unwrap();
        let window_element_00 = element_00
            .value()
            .power
            .windows(2 * hws + 1)
            .into_iter()
            .map(|w| w.mean().unwrap().re);
        let window_element_stds = element_00
            .value()
            .power
            .windows(2 * hws + 1)
            .into_iter()
            .map(|w| w.map(|x| x.re).std(1.0));

        // for (f, (window, (mean, std))) in freq
        //     .windows(2 * hws + 1)
        //     .zip(
        //         element_00
        //             .value()
        //             .power
        //             .windows(2 * hws + 1)
        //             .into_iter()
        //             .zip(window_element_00.zip(window_element_stds)),
        //     )
        //     .skip(100_000)
        //     // .step_by(1)
        //     .take(1)
        // {
        //     // if f > 0.099 && f < 0.101 {
        //     let mean_man = window.sum().re / window.len() as f32;
        //     let min = window
        //         .iter()
        //         .map(|x| x.re)
        //         .min_by(|a, b| a.partial_cmp(&b).unwrap())
        //         .unwrap();
        //     let std_man = (window
        //         .iter()
        //         .map(|x| (x.re - mean_man).powi(2))
        //         .sum::<f32>()
        //         / (window.len() - 1) as f32)
        //         .sqrt();
        //     for w in window {
        //         println!("{w}");
        //     }
        //     println!("mean = {mean}, std = {std}");
        //     println!("mean_man = {mean_man}, std_man = {std_man}; min = {min}");

        //     // let normal =
        //     //     // statrs::distribution::Normal::new(mean_man as f64, std_man as f64).unwrap();
        //     //     statrs::distribution::ChiSquared::new(1.0).unwrap();
        //     // let cdf_window: Vec<f32> = (1..=window.len())
        //     //     .map(|i| i as f32 / window.len() as f32)
        //     //     .collect();
        //     // let sorted_window = window
        //     //     .into_iter()
        //     //     .map(|z| z.re)
        //     //     .sorted_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
        //     // z_chart.draw_series(LineSeries::new(
        //     //     sorted_window
        //     //         .clone()
        //     //         .map(|w| (w, normal.cdf((w as f64 / std_man as f64)) as f32)),
        //     //     GREEN.mix(0.8).filled(),
        //     // ))?;
        //     // z_chart.draw_series(LineSeries::new(
        //     //     sorted_window.zip(cdf_window.clone()),
        //     //     RED.mix(0.8).filled(),
        //     // ))?;
        //     // z_chart
        //     //     .draw_secondary_series(
        //     //         Histogram::vertical(&z_chart.borrow_secondary())
        //     //             .style(BLUE.filled())
        //     //             .margin(5)
        //     //             .data(window.iter().map(|x| (x.re, 1))),
        //     //     )
        //     //     .unwrap();

        //     // let mut save_file = std::fs::File::create("window_hist_data").unwrap();
        //     // for (ff, w) in f.into_iter().zip(window) {
        //     //     save_file.write(format!("{ff} ").as_ref()).unwrap();
        //     //     save_file.write(format!("{}\n", w.re).as_ref()).unwrap();
        //     // }

        //     // } else {
        //     // println!("{f}");
        //     // }
        // }

        let window_element_00: Array1<f32> = element_00
            .value()
            .power
            .windows(2 * hws + 1)
            .into_iter()
            .map(|w| w.mean().unwrap().re)
            .collect();
        let window_element_stds: Array1<f32> = element_00
            .value()
            .power
            .windows(2 * hws + 1)
            .into_iter()
            .map(|w| w.map(|x| x.re).std(1.0))
            .collect();
        let upper = &window_element_00 + &window_element_stds;
        let lower = &window_element_00 - &window_element_stds;

        z_chart.draw_series(LineSeries::new(
            // freq.into_iter()
            //     .zip(element_00.value().power.iter().map(|p| p.re)),
            window_freqs.clone().zip(window_element_00),
            color.filled(),
        ))?;

        z_chart.draw_series(LineSeries::new(
            // freq.into_iter()
            //     .zip(element_00.value().power.iter().map(|p| p.re)),
            window_freqs.clone().zip(upper),
            color.mix(1.0),
        ))?;
        z_chart.draw_series(LineSeries::new(
            // freq.into_iter()
            //     .zip(element_00.value().power.iter().map(|p| p.re)),
            window_freqs.zip(lower),
            color.mix(1.0),
        ))?;
        println!("finished {tau}");
    }

    // root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}
