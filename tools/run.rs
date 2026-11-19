//! Runs the product's scene without a window and writes the model's buffers
//! after chosen updates:
//! `cargo run --release --example run -- OUTPUT --steps N [--dump A,B,..] [--light EVERY]`
//!
//! As in the app, the picture rests and is turned over at once. `--dump`
//! writes the eight compared buffers, the strata and the labels after the
//! listed updates (0 is the scene before any update) into
//! `OUTPUT/step-NNNNN/`; `--light EVERY` writes the cells and the terrain
//! every EVERY updates. Run outside a sandbox: it needs the GPU.
use mikage::{GpuContext, GpuDescriptor, GpuRequirements};
use sand_picture::model::{Model, Scene};
use std::path::{Path, PathBuf};

fn wait(gpu: &GpuContext) {
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
}

fn read(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    wait(gpu);
    staging.slice(..).get_mapped_range().unwrap().to_vec()
}

fn write(gpu: &GpuContext, model: &Model, folder: &Path, light: bool) {
    std::fs::create_dir_all(folder).unwrap();
    let buffers = model.buffers();
    let extra = [("strata", &buffers.strata), ("labels", &buffers.labels)];
    for (name, buffer) in buffers.named().into_iter().chain(extra) {
        if light && name != "cells" && name != "terrain" {
            continue;
        }
        std::fs::write(folder.join(format!("{name}.bin")), read(gpu, buffer)).unwrap();
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let output = PathBuf::from(
        args.next()
            .expect("OUTPUT --steps N [--dump A,B,..] [--light EVERY]"),
    );
    let mut steps = 0_u64;
    let mut dump: Vec<u64> = Vec::new();
    let mut light = None;
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--steps" => steps = value().parse().expect("--steps N"),
            "--dump" => {
                dump = value()
                    .split(',')
                    .map(|v| v.parse().expect("--dump A,B,.."))
                    .collect();
            }
            "--light" => light = Some(value().parse::<u64>().expect("--light EVERY")),
            other => panic!("unknown argument {other:?}"),
        }
    }
    std::fs::create_dir_all(&output).unwrap();
    let started = std::time::Instant::now();
    let gpu = pollster::block_on(GpuContext::headless(GpuDescriptor {
        requirements: GpuRequirements {
            required_limits: Some(wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            }),
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let mut model = Model::new(&gpu.device, &gpu.queue, &Scene::product());
    wait(&gpu);
    println!("model ready after {:.2} s", started.elapsed().as_secs_f64());
    model.start_turn();
    if dump.contains(&0) {
        write(&gpu, &model, &output.join("step-00000"), false);
    }
    let running = std::time::Instant::now();
    for step in 1..=steps {
        let mirror = model.advance_turn(&gpu.queue);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            if mirror {
                model.invert(&mut pass);
            }
            model.encode(&mut pass);
        }
        gpu.queue.submit([encoder.finish()]);
        let full = dump.contains(&step);
        if full || light.is_some_and(|every| step % every == 0) {
            wait(&gpu);
            write(&gpu, &model, &output.join(format!("step-{step:05}")), !full);
        } else if step % 8 == 0 {
            wait(&gpu);
        }
    }
    wait(&gpu);
    let seconds = running.elapsed().as_secs_f64();
    println!(
        "{steps} updates in {seconds:.1} s ({:.2} ms an update, readbacks included)",
        1e3 * seconds / steps.max(1) as f64
    );
}
