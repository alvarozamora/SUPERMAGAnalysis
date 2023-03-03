use itertools::{Itertools, MinMaxResult};
use plotters::prelude::*;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use supermag_analysis::weights::ProjectionsComplete;
use sysinfo::{System, SystemExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sys = System::new_all();

    // Load in projections and auxiliary values
    let projections_complete = read_dark_photon_projections().unwrap();
    println!(
        "proj has {} elements, the first has length {}",
        projections_complete.len(),
        projections_complete.iter().next().unwrap().value().len()
    );
    // println!("auxiliary has {} elements", auxiliary_complete.h[0].len());
    sys.refresh_all();
    println!("total memory: {} bytes", sys.total_memory());
    println!("used memory : {} bytes", sys.used_memory());

    let root_drawing_area = BitMapBackend::new("images/proj.png", (1024, 768)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut len = 0;
    let (min, max) = projections_complete.projections_complete.iter().fold(
        (0.0_f32, 0.0_f32),
        |(mut min, mut max), element| {
            let MinMaxResult::MinMax(&element_min, &element_max) = element.iter().minmax() else {
            unreachable!()
            };
            len = element.len();
            min = min.min(element_min);
            max = max.max(element_max);
            (min, max)
        },
    );

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .margin(40)
        .build_cartesian_2d(0..len, min..max)
        .unwrap();

    for (e, series) in projections_complete.projections_complete.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                (0..len).zip(series.iter().cloned()),
                &Palette99::pick(e),
            ))
            .unwrap()
            .label(format!("X{}", e + 1))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &Palette99::pick(e)));
    }

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    Ok(())
}

fn read_dark_photon_projections() -> Result<Arc<ProjectionsComplete>, Box<dyn std::error::Error>> {
    // Open projections_complete and auxiliary_complete file
    let mut projections_file = File::open("projections_complete").expect("failed to open file");

    // Initialize buffer for projections and auxiliary values
    let mut projection_buffer = Vec::new();

    // Read bytes in files
    projections_file
        .read_to_end(&mut projection_buffer)
        .expect("failed to read projections");

    // Deserialize bytes into respective types
    let projections_complete: Arc<ProjectionsComplete> = Arc::new(
        bincode::deserialize(&projection_buffer)
            .expect("failed to deserialize projections_complete"),
    );

    Ok(projections_complete)
}
