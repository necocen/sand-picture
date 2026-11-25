//! Runs the model without a window and writes its buffers after chosen
//! updates:
//! `cargo run --release --example run -- OUTPUT --steps N [--dump A,B,..] [--light EVERY] [--load FOLDER,STEP] [--turn A,B,..] [--shake A,B,..] [--reset A,B,..] [--views] [--frames] [--size WxH] [--time-pictures N] [--time-stages N]`
//!
//! The run starts from the product's scene at rest, and as in the app the
//! picture is turned over at once (`--turn 0`); `--turn` names instead the
//! updates after which a turn starts, and `--shake` those after which the
//! picture is shaken for a moment (a shake asked for during a shake or a turn
//! is not made, as in the app). `--load FOLDER,STEP` starts from a dump's
//! folder as the state after update STEP, with no turn unless one is named.
//! `--reset` returns the model to the scene at rest after the named updates
//! (before a turn named for the same update), as the app's reset does. `--steps` is the number of the last update. `--dump` writes the
//! picture's state (the cells, the geometry rows, the faces, the flow, the
//! bed's columns, the strata and the labels) after the listed updates into
//! `OUTPUT/step-NNNNN/`; `--light EVERY` writes the cells and the bed's
//! columns every EVERY updates. `--views` adds the three views' pictures to
//! each dump, as raw RGBA and as PNG (`image` is Photoreal, the name the
//! baseline gives it); `--frames` draws a Photoreal picture after every
//! update, as a window that draws each update does. `--size WxH` is the pictures' size in
//! pixels (the regression's 1440x960 unless given), and `--time-pictures N`
//! draws N Photoreal pictures of the last state and tells what one costs.
//! `--time-stages N` runs N more updates after the last, each stage in a pass
//! of its own, and tells what each stage costs by the GPU's timestamps at the
//! two ends of its pass (an Apple GPU writes none inside a pass); the passes'
//! own cost, about 10 us each, is in the numbers.
//! Run outside a sandbox: it needs the GPU.
//!
//! Two runs agree bit for bit in the state but for the labels, where they
//! agree only in the films' counts and the cells' words: the bubbles'
//! records are numbered in the order the GPU's threads reach them.
mod common;

use common::{Pictures, headless, picture_name, read, wait, write_png};
use mikage::GpuContext;
use sand_picture::model::{DT, Model, Scene, Stage};
use sand_picture::view::{Mode, REFERENCE_SIZE};
use std::path::{Path, PathBuf};

/// Writes the three views into a dump's folder. Every view is drawn before
/// the buffers are read: the comparison of the buffers then also shows that
/// no view writes to them.
fn write_pictures(gpu: &GpuContext, pictures: &Pictures, folder: &Path) {
    std::fs::create_dir_all(folder).unwrap();
    for mode in Mode::ALL {
        let name = picture_name(mode);
        let pixels = pictures.picture(gpu, mode);
        std::fs::write(folder.join(format!("{name}.bin")), &pixels).unwrap();
        write_png(
            &folder.join(format!("{name}.png")),
            &pixels,
            pictures.size(),
        );
    }
}

/// Writes the picture's state into a dump's folder, or of it only the cells
/// and the bed's columns.
fn write(gpu: &GpuContext, model: &Model, folder: &Path, light: bool) {
    std::fs::create_dir_all(folder).unwrap();
    for (name, buffer) in model.buffers().state() {
        if light && !["cells", "terrain"].contains(&name) {
            continue;
        }
        std::fs::write(folder.join(format!("{name}.bin")), read(gpu, buffer)).unwrap();
    }
}

/// Writes a dump into a model just built: the state after an update. The
/// model's other buffers keep what its preparation of the scene left, and
/// the next update writes them before it reads them.
fn load(gpu: &GpuContext, model: &Model, folder: &Path) {
    for (name, buffer) in model.buffers().state() {
        let bytes = std::fs::read(folder.join(format!("{name}.bin")))
            .unwrap_or_else(|error| panic!("{name}.bin of {}: {error}", folder.display()));
        assert_eq!(bytes.len() as u64, buffer.size(), "{name} size differs");
        gpu.queue.write_buffer(buffer, 0, &bytes);
    }
}

/// The update's stages in order, each with the range of its steps.
fn stages(model: &Model) -> Vec<(Stage, std::ops::Range<usize>)> {
    let mut stages: Vec<(Stage, std::ops::Range<usize>)> = Vec::new();
    for (index, step) in model.steps().iter().enumerate() {
        match stages.last_mut() {
            Some((stage, range)) if *stage == step.stage => range.end = index + 1,
            _ => stages.push((step.stage, index..index + 1)),
        }
    }
    stages
}

/// Runs `count` updates with a pass to each stage and returns each stage's
/// mean time on the GPU (ms an update), by the timestamps at the two ends of
/// its pass. The timestamps of a batch of updates are read back together, so
/// that no readback falls between two updates of a batch.
fn time_stages(gpu: &GpuContext, model: &mut Model, count: u64) -> Vec<f64> {
    assert!(
        gpu.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY),
        "this GPU gives no timestamps"
    );
    let stages = stages(model);
    let marks = 2 * stages.len() as u32;
    let batch = (wgpu::QUERY_SET_MAX_QUERIES / marks).min(32);
    let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("stage timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: marks * batch,
    });
    let resolved = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stage timestamps"),
        size: 8 * u64::from(marks * batch),
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let period = f64::from(gpu.queue.get_timestamp_period());
    let mut sums = vec![0_f64; stages.len()];
    let mut done = 0;
    while done < count {
        let updates = u64::from(batch).min(count - done) as u32;
        for update in 0..updates {
            let mirror = model.advance_motion(&gpu.queue);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            if mirror {
                model.invert(&mut encoder.begin_compute_pass(&Default::default()));
            }
            for (index, (_, range)) in stages.iter().enumerate() {
                let first = update * marks + 2 * index as u32;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &set,
                        beginning_of_pass_write_index: Some(first),
                        end_of_pass_write_index: Some(first + 1),
                    }),
                });
                model.encode_steps(&mut pass, range.clone());
            }
            gpu.queue.submit([encoder.finish()]);
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&set, 0..updates * marks, &resolved, 0);
        gpu.queue.submit([encoder.finish()]);
        let ticks: Vec<u64> = read(gpu, &resolved)
            .chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
            .collect();
        for update in 0..updates as usize {
            for (index, sum) in sums.iter_mut().enumerate() {
                let first = update * marks as usize + 2 * index;
                *sum += ticks[first + 1].saturating_sub(ticks[first]) as f64 * period * 1e-6;
            }
        }
        done += u64::from(updates);
    }
    sums.iter().map(|sum| sum / count as f64).collect()
}

/// The update's stages in order, each with its number of steps.
fn composition(model: &Model) -> String {
    stages(model)
        .iter()
        .map(|(stage, range)| format!("{} {}", stage.label(), range.len()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn numbers(text: &str, option: &str) -> Vec<u64> {
    text.split(',')
        .map(|v| v.parse().unwrap_or_else(|_| panic!("{option} A,B,..")))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let output = PathBuf::from(args.next().expect(
        "OUTPUT --steps N [--dump A,B,..] [--light EVERY] [--load FOLDER,STEP] [--turn A,B,..] [--shake A,B,..] [--reset A,B,..] [--views] [--frames] [--size WxH] [--time-pictures N] [--time-stages N]",
    ));
    let mut steps = 0_u64;
    let mut dump: Vec<u64> = Vec::new();
    let mut light = None;
    let mut loaded: Option<(PathBuf, u64)> = None;
    let mut turns: Option<Vec<u64>> = None;
    let mut shakes: Vec<u64> = Vec::new();
    let mut resets: Vec<u64> = Vec::new();
    let mut sensor_gravity: Option<[f32; 3]> = None;
    let mut views = false;
    let mut frames = false;
    let mut size = REFERENCE_SIZE;
    let mut timed_pictures = 0_u32;
    let mut timed_stages = 0_u64;
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--steps" => steps = value().parse().expect("--steps N"),
            "--dump" => dump = numbers(&value(), "--dump"),
            "--light" => light = Some(value().parse::<u64>().expect("--light EVERY")),
            "--turn" => turns = Some(numbers(&value(), "--turn")),
            "--shake" => shakes = numbers(&value(), "--shake"),
            "--reset" => resets = numbers(&value(), "--reset"),
            "--gravity" => {
                let raw = value();
                let parts: Vec<f32> = raw
                    .split(',')
                    .map(|v| v.parse().expect("--gravity GX,GY,GZ"))
                    .collect();
                assert_eq!(parts.len(), 3, "--gravity GX,GY,GZ");
                sensor_gravity = Some([parts[0], parts[1], parts[2]]);
            }
            "--views" => views = true,
            "--frames" => frames = true,
            "--size" => {
                let value = value();
                let (width, height) = value.split_once('x').expect("--size WxH");
                size = [width, height].map(|v| v.parse().expect("--size WxH"));
            }
            "--time-pictures" => timed_pictures = value().parse().expect("--time-pictures N"),
            "--time-stages" => timed_stages = value().parse().expect("--time-stages N"),
            "--load" => {
                let value = value();
                let (folder, step) = value.rsplit_once(',').expect("--load FOLDER,STEP");
                loaded = Some((folder.into(), step.parse().expect("--load FOLDER,STEP")));
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    std::fs::create_dir_all(&output).unwrap();
    let started = std::time::Instant::now();
    let gpu = headless();
    let mut model = Model::new(&gpu.device, &gpu.queue, Scene::product());
    wait(&gpu);
    println!("model ready after {:.2} s", started.elapsed().as_secs_f64());
    println!(
        "an update is {} steps: {}",
        model.steps().len(),
        composition(&model)
    );
    let start = match &loaded {
        Some((folder, step)) => {
            load(&gpu, &model, folder);
            *step
        }
        None => 0,
    };
    model.set_sensor_gravity(sensor_gravity);
    let turns = turns.unwrap_or(if loaded.is_some() { vec![] } else { vec![0] });
    let pictures =
        (views || frames || timed_pictures > 0).then(|| Pictures::new(&gpu, &model, size));
    // A full dump: the views first, if asked for, then the buffers.
    let dump_all = |model: &Model, step: u64| {
        let folder = output.join(format!("step-{step:05}"));
        if let Some(pictures) = pictures.as_ref().filter(|_| views) {
            write_pictures(&gpu, pictures, &folder);
        }
        write(&gpu, model, &folder, false);
    };
    if dump.contains(&start) {
        dump_all(&model, start);
    }
    let running = std::time::Instant::now();
    for step in start + 1..=steps {
        if resets.contains(&(step - 1)) {
            model.reset(&gpu.device, &gpu.queue);
        }
        if turns.contains(&(step - 1)) {
            model.start_turn();
        }
        if shakes.contains(&(step - 1)) {
            model.start_shake();
        }
        let mirror = model.advance_motion(&gpu.queue);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if mirror {
                model.invert(&mut pass);
            }
            model.encode(&mut pass);
        }
        if let Some(pictures) = pictures.as_ref().filter(|_| frames) {
            pictures.draw(&mut encoder, Mode::default());
        }
        gpu.queue.submit([encoder.finish()]);
        if dump.contains(&step) {
            wait(&gpu);
            dump_all(&model, step);
        } else if light.is_some_and(|every| step % every == 0) {
            wait(&gpu);
            write(&gpu, &model, &output.join(format!("step-{step:05}")), true);
        } else if step % 8 == 0 {
            wait(&gpu);
        }
    }
    wait(&gpu);
    let seconds = running.elapsed().as_secs_f64();
    if let Some(pictures) = pictures.as_ref().filter(|_| timed_pictures > 0) {
        // One picture is drawn first: it writes the whole distance field.
        let draw = |count: u32| {
            for _ in 0..count {
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                pictures.draw(&mut encoder, Mode::Photoreal);
                gpu.queue.submit([encoder.finish()]);
            }
            wait(&gpu);
        };
        draw(1);
        let started = std::time::Instant::now();
        draw(timed_pictures);
        println!(
            "a Photoreal picture of {} x {}: {:.2} ms ({timed_pictures} drawn)",
            size[0],
            size[1],
            1e3 * started.elapsed().as_secs_f64() / f64::from(timed_pictures)
        );
    }
    let run = steps.saturating_sub(start);
    println!(
        "{run} updates ({:.0} s of the picture) in {seconds:.1} s: {:.2} ms an update, readbacks included",
        run as f64 * f64::from(DT),
        1e3 * seconds / run.max(1) as f64
    );
    if timed_stages > 0 {
        let times = time_stages(&gpu, &mut model, timed_stages);
        let total: f64 = times.iter().sum();
        println!(
            "{timed_stages} more updates, a pass to each stage: {total:.2} ms an update on the GPU"
        );
        for ((stage, range), time) in stages(&model).iter().zip(&times) {
            println!(
                "  {:<20} {:>4} steps {time:>7.3} ms {:>5.1} %",
                stage.label(),
                range.len(),
                100. * time / total
            );
        }
    }
}
