use std::ops::Range;

use dashmap::DashMap;
use itertools::Itertools;
use ndarray::s;
use supermag_analysis::{
    theory::{
        dark_photon::{read_dark_photon_projections_auxiliary, DARK_PHOTON_NONZERO_ELEMENTS},
        NonzeroElement,
    },
    weights::Stationarity,
    FloatType,
};

use plotters::{prelude::*, style::full_palette::PURPLE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("quarterly_power.png", (1080, 720)).into_drawing_area();
    let root2 = BitMapBackend::new("quarterly_signal.png", (1080, 720)).into_drawing_area();

    root.fill(&WHITE)?;
    root2.fill(&WHITE)?;

    let stationarity = Stationarity::Yearly;

    // Load in projections and auxiliary values
    let (projections_complete, _auxiliary_complete) =
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
    let projections_subset: DashMap<NonzeroElement, Vec<FloatType>> = projections_complete
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

    let mut quarters: Vec<Vec<(FloatType, FloatType)>> = vec![];
    let mut quarter_powers: Vec<Vec<(FloatType, FloatType)>> = vec![];
    for [start, end] in quarter_indices {
        let mut quarter: Vec<nalgebra::Complex<FloatType>> =
            first_series[start..end].iter().map(|x| x.into()).collect();

        // // TODO: REMOVE
        // if start == 0 {
        //     let check = quarter[2_000_000..2_001_000]
        //         .iter()
        //         .map(|z| z.norm_sqr())
        //         .collect_vec();
        //     println!("{check:?}");
        // }
        const WINDOW_SIZE: usize = 1024;

        quarters.push(
            first_series[start..end]
                .iter()
                .enumerate()
                .map(|(i, e)| (i as FloatType, *e))
                .collect(),
        );

        let mut fft_planner = rustfft::FftPlanner::new();
        let handler = fft_planner.plan_fft(quarter.len(), rustfft::FftDirection::Forward);
        handler.process(&mut quarter);

        // Used for average of windows
        let avg_quarter_power: Vec<f32> = quarter
            .windows(WINDOW_SIZE)
            .map(|w| w.iter().map(|z| z.norm_sqr()).sum::<f32>() / WINDOW_SIZE as f32)
            .collect();

        // Used for no windows
        let freqs: Vec<FloatType> = (0..quarter.len())
            .map(|i| i as FloatType * 1.0 / quarter.len() as FloatType)
            .collect();
        // Used for average of windows
        let window_freqs: Vec<FloatType> = (WINDOW_SIZE / 2..quarter.len() - WINDOW_SIZE / 2)
            .map(|i| i as FloatType * 1.0 / quarter.len() as FloatType)
            .collect();

        // No windows
        // quarter_powers.push(
        //     freqs
        //         .into_iter()
        //         .zip(&quarter)
        //         .map(|(f, p)| (f, p.norm_sqr() / quarter.len() as FloatType))
        //         .collect(),
        // )

        // avg of windows
        quarter_powers.push(
            window_freqs
                .into_iter()
                .zip(avg_quarter_power)
                .map(|(f, p)| (f, p))
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

    let lot = quarters
        .iter()
        .map(|x| x.iter())
        .flatten()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;
    let hit = quarters
        .iter()
        .map(|x| x.iter())
        .flatten()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;
    let los = quarters
        .iter()
        .map(|x| x.iter())
        .flatten()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .1;
    let his = quarters
        .iter()
        .map(|x| x.iter())
        .flatten()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .1;

    let mut power_chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .margin(5)
        .caption("Quarterly Power in 2013", ("sans-serif", 50.0))
        .build_cartesian_2d((lof..hif).log_scale(), (lop..hip).log_scale())?;

    let mut signal_chart = ChartBuilder::on(&root2)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .margin(5)
        .caption("Quarterly Signal in 2013", ("sans-serif", 50.0))
        .build_cartesian_2d(lot..hit, los..his)?;

    for (q, (qfp, color)) in quarter_powers
        .into_iter()
        .zip([RED, BLUE, PURPLE, GREEN])
        .enumerate()
    {
        // if q == 1 {
        power_chart
            .draw_series(LineSeries::new(qfp, color.mix(0.1)))?
            .label(&format!("Quarter {q}"))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
        // }
    }

    for (q, (qsignal, color)) in quarters
        .into_iter()
        .zip([RED, BLUE, PURPLE, GREEN])
        .enumerate()
    {
        signal_chart
            .draw_series(LineSeries::new(qsignal, color.mix(0.1)))?
            .label(&format!("Quarter {q}"))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    power_chart
        .configure_mesh()
        .disable_x_mesh()
        .y_label_formatter(&|x| format!("{x:1e}"))
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Quarterly Power")
        .x_desc("Frequency")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    power_chart
        .configure_series_labels()
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperMiddle)
        .draw()?;

    signal_chart
        .configure_mesh()
        .disable_x_mesh()
        .y_label_formatter(&|x| format!("{x:1e}"))
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Quarterly Signal")
        .x_desc("Time")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    signal_chart
        .configure_series_labels()
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperMiddle)
        .draw()?;

    root.present().expect("Unable to write result to file");
    root2.present().expect("Unable to write result to file");

    Ok(())
}
