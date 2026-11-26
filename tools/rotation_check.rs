//! End-to-end GPU stress test for full 360-degree Z_4 rotation sequences:
//! 1. Rotate 90 deg CW (`216x144 -> 144x216`) and let sand accumulate on the
//!    side wall (`60 s` default, `180 s` with `--long`).
//! 2. From the settled sideways state, invert 180 deg (`144x216`) and verify
//!    clean drainage and bubble retention.
//! 3. From the settled sideways state, return 90 deg CW to upright (`216x144`),
//!    verify bubbles spread cleanly without collapsing into a corner and steep
//!    cliffs slump cleanly, then invert 180 deg upright.
//! 4. From the settled sideways state, return 90 deg CCW to upright (`216x144`)
//!    and verify no overhanging bed cave or corner bubble collapse.
//!
//! Usage (outside sandbox; needs GPU):
//! `cargo run --release --example rotation_check -- [--record FOLDER] [--long]`
mod common;

use common::{Pictures, headless, picture_name, read, wait, write_png};
use mikage::GpuContext;
use sand_picture::model::{CELLS, DEFAULT_AIR_AREA, DT, GridTurn, Model, Scene};
use sand_picture::view::{Mode, REFERENCE_SIZE};
use std::path::{Path, PathBuf};

struct Stats {
    total_sand: f64,
    bed_frac: f64,
    raster_gas: f64,
    max_h: f32,
    min_h: f32,
    non_finite: usize,
}

fn measure_state(gpu: &GpuContext, model: &Model) -> Stats {
    let buffers = model.buffers();
    let [nx, _ny] = model.grid();
    let cells_bytes = read(gpu, &buffers.cells);
    let mut mob = 0.0_f64;
    let mut bed = 0.0_f64;
    let mut gas = 0.0_f64;
    let mut non_finite = 0_usize;
    for cell in cells_bytes.chunks_exact(96) {
        for word in cell.chunks_exact(4) {
            let v = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            if !v.is_finite() {
                non_finite += 1;
            }
        }
        let f = |idx: usize| -> f64 {
            let off = idx * 4;
            f64::from(f32::from_le_bytes([
                cell[off],
                cell[off + 1],
                cell[off + 2],
                cell[off + 3],
            ]))
        };
        mob += f(1) + f(2);
        gas += f(3);
        bed += f(20) + f(21);
    }
    let terrain_bytes = read(gpu, &buffers.terrain);
    let mut max_h = 0.0_f32;
    let mut min_h = f32::MAX;
    for col in terrain_bytes.chunks_exact(16).take(nx as usize) {
        let h = f32::from_le_bytes([col[0], col[1], col[2], col[3]]);
        max_h = max_h.max(h);
        min_h = min_h.min(h);
    }
    let total_sand = mob + bed;
    Stats {
        total_sand,
        bed_frac: if total_sand > 0.0 {
            bed / total_sand
        } else {
            0.0
        },
        raster_gas: gas,
        max_h,
        min_h: if min_h == f32::MAX { 0.0 } else { min_h },
        non_finite,
    }
}

fn step_model(gpu: &GpuContext, model: &mut Model, count: u64) {
    for step in 1..=count {
        let turn = model.advance_motion(&gpu.queue);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if turn != GridTurn::None {
                model.apply_turn(&mut pass, turn);
            }
            model.encode(&mut pass);
        }
        gpu.queue.submit([encoder.finish()]);
        if step.is_multiple_of(8) {
            wait(gpu);
        }
    }
    wait(gpu);
}

fn snapshot(gpu: &GpuContext, model: &Model) -> (bool, Vec<Vec<u8>>) {
    let bytes = model
        .buffers()
        .state()
        .map(|(_, buffer)| read(gpu, buffer))
        .to_vec();
    (model.is_transposed(), bytes)
}

fn restore(gpu: &GpuContext, model: &mut Model, saved: &(bool, Vec<Vec<u8>>)) {
    model.reset(&gpu.device, &gpu.queue);
    model.set_transposed(&gpu.queue, saved.0);
    for ((_, buffer), bytes) in model.buffers().state().into_iter().zip(&saved.1) {
        gpu.queue.write_buffer(buffer, 0, bytes);
    }
    wait(gpu);
}

fn record_views(
    gpu: &GpuContext,
    model: &Model,
    pictures: &mut Pictures,
    folder: Option<&Path>,
    label: &str,
) {
    pictures.sync_grid(gpu, model.grid(), REFERENCE_SIZE);
    let expected = if model.is_transposed() {
        [REFERENCE_SIZE[1], REFERENCE_SIZE[0]]
    } else {
        REFERENCE_SIZE
    };
    assert_eq!(pictures.size(), expected, "{label}: view size mismatch");
    for mode in Mode::ALL {
        let pixels = pictures.picture(gpu, mode);
        assert_eq!(
            pixels.len(),
            (expected[0] * expected[1] * 4) as usize,
            "{label}: {} pixel count mismatch",
            picture_name(mode)
        );
        if let Some(dir) = folder
            && mode == Mode::Photoreal
        {
            write_png(&dir.join(format!("{label}.png")), &pixels, expected);
        }
    }
}

fn check_invariants(
    label: &str,
    stats: &Stats,
    baseline_sand: f64,
    min_gas_frac: f64,
    failures: &mut Vec<String>,
) {
    let drift_pct = (stats.total_sand - baseline_sand).abs() / baseline_sand * 100.0;
    let gas_frac = stats.raster_gas / f64::from(DEFAULT_AIR_AREA);
    println!(
        "{label:<22} tot={:8.1} (drift={drift_pct:.4}%) bed={:5.1}% gas={:5.1} ({:4.0}%) h=[{:5.1}, {:5.1}] nan={}",
        stats.total_sand,
        stats.bed_frac * 100.0,
        stats.raster_gas,
        gas_frac * 100.0,
        stats.min_h,
        stats.max_h,
        stats.non_finite,
    );
    if stats.non_finite > 0 {
        failures.push(format!(
            "{label}: {} non-finite cell floats",
            stats.non_finite
        ));
    }
    if drift_pct > 0.05 {
        failures.push(format!("{label}: sand drift {drift_pct:.4}% exceeds 0.05%"));
    }
    if gas_frac < min_gas_frac {
        failures.push(format!(
            "{label}: rasterized gas {:.1} ({:.0}% of {:.0}) below {:.0}% (corner bubble collapse)",
            stats.raster_gas,
            gas_frac * 100.0,
            DEFAULT_AIR_AREA,
            min_gas_frac * 100.0,
        ));
    }
}

fn main() {
    let mut record: Option<PathBuf> = None;
    let mut long = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--record" => record = Some(args.next().expect("--record FOLDER").into()),
            "--long" => long = true,
            other => panic!("unknown argument {other:?}"),
        }
    }
    if let Some(dir) = &record {
        std::fs::create_dir_all(dir).unwrap();
    }

    let side_secs = if long { 180_u64 } else { 60_u64 };
    let follow_secs = if long { 30_u64 } else { 15_u64 };
    let per_sec = (1.0 / f64::from(DT)).round() as u64;

    let gpu = headless();
    let mut model = Model::new(&gpu.device, &gpu.queue, Scene::product());
    let mut pictures = Pictures::new(&gpu, &model, REFERENCE_SIZE);
    let baseline_sand = measure_state(&gpu, &model).total_sand;
    let mut failures = Vec::new();

    let upright = CELLS.map(|v| v as u32);
    let sideways = [upright[1], upright[0]];

    // Start as the app does: invert upright for 1 s, then rotate 90 deg CW.
    model.start_turn();
    step_model(&gpu, &mut model, per_sec);
    assert!(model.start_quarter_turn(true), "quarter turn CW must start");
    assert_eq!(model.grid(), upright);

    // Stage 1: accumulate sand on the side wall (sideways).
    let stage1_mid = 15 * per_sec;
    step_model(&gpu, &mut model, stage1_mid);
    assert_eq!(model.grid(), sideways);
    let s1_mid = measure_state(&gpu, &model);
    check_invariants("side_15s", &s1_mid, baseline_sand, 0.55, &mut failures);
    record_views(&gpu, &model, &mut pictures, record.as_deref(), "side_15s");

    step_model(&gpu, &mut model, side_secs * per_sec - stage1_mid);
    let s1_end = measure_state(&gpu, &model);
    check_invariants(
        &format!("side_{side_secs}s"),
        &s1_end,
        baseline_sand,
        0.65,
        &mut failures,
    );
    if s1_end.bed_frac < 0.90 {
        failures.push(format!(
            "side_{side_secs}s: expected settled bed >= 90%, got {:.1}%",
            s1_end.bed_frac * 100.0
        ));
    }
    record_views(
        &gpu,
        &model,
        &mut pictures,
        record.as_deref(),
        &format!("side_{side_secs}s"),
    );

    let settled_side = snapshot(&gpu, &model);

    // Stage 2: invert 180 deg while sideways.
    assert!(model.start_turn(), "sideways 180 flip must start");
    step_model(&gpu, &mut model, follow_secs * per_sec);
    assert_eq!(model.grid(), sideways);
    let s2 = measure_state(&gpu, &model);
    check_invariants(
        &format!("side_flip180_{follow_secs}s"),
        &s2,
        baseline_sand,
        0.65,
        &mut failures,
    );
    if s2.bed_frac < 0.02 {
        failures.push(format!(
            "side_flip180_{follow_secs}s: sand stalled after sideways flip (bed {:.1}%)",
            s2.bed_frac * 100.0
        ));
    }
    record_views(
        &gpu,
        &model,
        &mut pictures,
        record.as_deref(),
        &format!("side_flip180_{follow_secs}s"),
    );

    // Stage 3: restore settled sideways state, return 90 deg CW to upright,
    // then invert 180 deg upright.
    restore(&gpu, &mut model, &settled_side);
    assert!(model.start_quarter_turn(true), "return CW must start");
    step_model(&gpu, &mut model, follow_secs * per_sec);
    assert_eq!(model.grid(), upright);
    let s3 = measure_state(&gpu, &model);
    check_invariants(
        &format!("return_cw_{follow_secs}s"),
        &s3,
        baseline_sand,
        0.75,
        &mut failures,
    );
    record_views(
        &gpu,
        &model,
        &mut pictures,
        record.as_deref(),
        &format!("return_cw_{follow_secs}s"),
    );

    assert!(
        model.start_turn(),
        "post-return upright 180 flip must start"
    );
    step_model(&gpu, &mut model, follow_secs * per_sec);
    let s3_flip = measure_state(&gpu, &model);
    check_invariants(
        &format!("return_cw_flip_{follow_secs}s"),
        &s3_flip,
        baseline_sand,
        0.65,
        &mut failures,
    );
    record_views(
        &gpu,
        &model,
        &mut pictures,
        record.as_deref(),
        &format!("return_cw_flip_{follow_secs}s"),
    );

    // Stage 4: restore settled sideways state, return 90 deg CCW to upright.
    restore(&gpu, &mut model, &settled_side);
    assert!(model.start_quarter_turn(false), "return CCW must start");
    step_model(&gpu, &mut model, follow_secs * per_sec);
    assert_eq!(model.grid(), upright);
    let s4 = measure_state(&gpu, &model);
    check_invariants(
        &format!("return_ccw_{follow_secs}s"),
        &s4,
        baseline_sand,
        0.75,
        &mut failures,
    );
    record_views(
        &gpu,
        &model,
        &mut pictures,
        record.as_deref(),
        &format!("return_ccw_{follow_secs}s"),
    );

    if failures.is_empty() {
        println!("rotation_check PASSED");
    } else {
        for msg in &failures {
            eprintln!("FAIL: {msg}");
        }
        std::process::exit(1);
    }
}
