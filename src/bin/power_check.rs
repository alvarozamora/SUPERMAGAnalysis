use std::ops::Range;

use dashmap::DashMap;
use ndarray::s;
use supermag_analysis::{
    theory::{
        dark_photon::{read_dark_photon_projections_auxiliary, DARK_PHOTON_NONZERO_ELEMENTS},
        NonzeroElement,
    },
    weights::Stationarity,
};

use nalgebra::Complex;
use ndarray::{Array1, ArrayBase};
use plotters::{prelude::*, style::full_palette::PURPLE};
use rocksdb::DB;
use statrs::distribution::ContinuousCDF;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("quarterly_power.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let stationarity = Stationarity::Yearly;

    // Load in projections and auxiliary values
    let (projections_complete, auxiliary_complete) =
        read_dark_photon_projections_auxiliary().unwrap();

    // Get stationarity period indices (place within entire SUPERMAG dataset)
    // MINOR NOTE: this definition varies from original implementation. The original
    // python implementation defines the `end` index to be the first index of the
    // next chunk, since start:end is not end inclusive. This means the size of
    // the chunks are (end - start + 1)
    let (start_stationarity, end_stationarity) = stationarity.get_year_second_indices(2013);

    // Now convert these indices to the indices within the subset used
    let secs: Range<usize> = projections_complete.secs();
    let (start_in_series, end_in_series) = (
        secs.clone().position(|i| i == start_stationarity).unwrap(), //.unwrap_or(secs.start),
        secs.clone().position(|i| i == end_stationarity).unwrap(), //.unwrap_or_else(|| { log::warn!("falling back to final element!"); secs.end-1}),
    );
    assert!(start_in_series < end_in_series, "invalid range");

    // Get the subseries for this year
    let projections_subset: DashMap<NonzeroElement, Vec<f32>> = projections_complete
        .iter()
        .map(|kv| {
            // Get element and the complete longest contiguous series
            let (element, complete_series) = kv.pair();
            let pair = (
                element.clone(),
                complete_series
                    .slice(s![start_in_series..=end_in_series])
                    .to_vec(),
            );
            pair
        })
        .collect();

    let first_element = &DARK_PHOTON_NONZERO_ELEMENTS[0];
    let first_series = projections_subset.get(first_element).unwrap();

    let series_len = first_series.len();
    let quarter_indices = [
        [0, series_len / 4],
        [series_len / 4, 2 * series_len / 4],
        [2 * series_len / 4, 3 * series_len / 4],
        [3 * series_len / 4, series_len],
    ];

    let mut quarter_powers: Vec<Vec<(f32, f32)>> = vec![];
    for [start, end] in quarter_indices {
        let mut quarter: Vec<nalgebra::Complex<f32>> =
            first_series[start..end].iter().map(|x| x.into()).collect();
        let mut fft_planner = rustfft::FftPlanner::new();
        let handler = fft_planner.plan_fft(quarter.len(), rustfft::FftDirection::Forward);
        handler.process(&mut quarter);

        let freqs: Vec<f32> = (0..quarter.len())
            .map(|i| i as f32 * 1.0 / quarter.len() as f32)
            .collect();

        quarter_powers.push(
            freqs
                .into_iter()
                .zip(quarter)
                .map(|(f, p)| (f, p.norm_sqr()))
                .collect(),
        )
    }

    let lof = quarter_powers
        .iter()
        .map(|x| x.iter())
        .flatten()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;
    let hif = quarter_powers
        .iter()
        .map(|x| x.iter())
        .flatten()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;
    let lop = quarter_powers
        .iter()
        .map(|x| x.iter())
        .flatten()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .1;
    let hip = quarter_powers
        .iter()
        .map(|x| x.iter())
        .flatten()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .1;

    let mut z_chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("Histogram Test", ("sans-serif", 50.0))
        .build_cartesian_2d(lof..hif, lop..hip)?;

    z_chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Quarterly Power")
        .x_desc("Frequency")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    for (qfp, color) in quarter_powers.into_iter().zip([RED, BLUE, PURPLE, BLACK]) {
        z_chart.draw_series(LineSeries::new(qfp, color))?;
    }

    root.present().expect("Unable to write result to file");

    Ok(())
}
