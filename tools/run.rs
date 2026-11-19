//! Runs the model without a window and writes its buffers after chosen
//! updates:
//! `cargo run --release --example run -- OUTPUT --steps N [--dump A,B,..] [--light EVERY] [--load FOLDER,STEP] [--turn A,B,..] [--reset A,B,..] [--views] [--frames]`
//!
//! The run starts from the product's scene at rest, and as in the app the
//! picture is turned over at once (`--turn 0`); `--turn` names instead the
//! updates after which a turn starts. `--load FOLDER,STEP` starts from a
//! dump's folder as the state after update STEP, with no turn unless one is
//! named. `--reset` returns the model to the scene at rest after the named
//! updates (before a turn named for the same update), as the app's reset
//! does. `--steps` is the number of the last update. `--dump` writes the
//! eight compared buffers, the strata and the labels after the listed updates
//! into `OUTPUT/step-NNNNN/`; `--light EVERY` writes the cells and the
//! terrain every EVERY updates. `--views` adds the three views' pictures to
//! each dump, as raw RGBA and as PNG (`image` is Photoreal, the name
//! v2.20's baseline gives it); `--frames` draws a Photoreal picture after
//! every update, as v2.20's window did. Run outside a sandbox: it needs the
//! GPU.
//!
//! Two runs agree bit for bit in the eight buffers and the strata. In the
//! labels they agree only in the films' counts and the cells' words: the
//! bubbles' records are numbered in the order the GPU's threads reach them.
use mikage::{GpuContext, GpuDescriptor, GpuRequirements};
use sand_picture::model::{DT, Model, Scene};
use sand_picture::view::{IMAGE_FORMAT, IMAGE_SIZE, Mode, Views};
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

/// The views and the image they draw into, as a window keeps them: one set
/// for the whole run, since a view may keep marks from picture to picture.
struct Pictures {
    views: Views,
    image: wgpu::Texture,
}
impl Pictures {
    fn new(gpu: &GpuContext, model: &Model) -> Self {
        Self {
            views: Views::new(&gpu.device, model.buffers()),
            image: gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("picture"),
                size: wgpu::Extent3d {
                    width: IMAGE_SIZE[0],
                    height: IMAGE_SIZE[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: IMAGE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
        }
    }

    fn draw(&self, encoder: &mut wgpu::CommandEncoder, mode: Mode) {
        self.views
            .encode(encoder, &self.image.create_view(&Default::default()), mode);
    }

    /// The picture of `mode` now: its pixels as RGBA, the top row first.
    fn picture(&self, gpu: &GpuContext, mode: Mode) -> Vec<u8> {
        let [width, height] = IMAGE_SIZE;
        // A copied row is padded to wgpu's row alignment.
        let row = (4 * width).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picture readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.draw(&mut encoder, mode);
        encoder.copy_texture_to_buffer(
            self.image.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            self.image.size(),
        );
        gpu.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        wait(gpu);
        staging
            .slice(..)
            .get_mapped_range()
            .unwrap()
            .chunks_exact(row as usize)
            .flat_map(|padded| &padded[..4 * width as usize])
            .copied()
            .collect()
    }

    /// Writes the three views into a dump's folder. Every view is drawn
    /// before the buffers are read: the comparison of the buffers then also
    /// shows that no view writes to them.
    fn write(&self, gpu: &GpuContext, folder: &Path) {
        std::fs::create_dir_all(folder).unwrap();
        for mode in Mode::ALL {
            let name = match mode {
                Mode::Simple => "simple",
                Mode::Photoreal => "image",
                Mode::Flow => "flow",
            };
            let pixels = self.picture(gpu, mode);
            std::fs::write(folder.join(format!("{name}.bin")), &pixels).unwrap();
            let file = std::fs::File::create(folder.join(format!("{name}.png"))).unwrap();
            let mut encoder =
                png::Encoder::new(std::io::BufWriter::new(file), IMAGE_SIZE[0], IMAGE_SIZE[1]);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&pixels)
                .unwrap();
        }
    }
}

/// What a full dump holds and a loaded run starts from: the eight compared
/// buffers, the strata and the labels.
fn saved(model: &Model) -> Vec<(&'static str, &wgpu::Buffer)> {
    let buffers = model.buffers();
    let extra = [("strata", &buffers.strata), ("labels", &buffers.labels)];
    buffers.named().into_iter().chain(extra).collect()
}

fn write(gpu: &GpuContext, model: &Model, folder: &Path, light: bool) {
    std::fs::create_dir_all(folder).unwrap();
    for (name, buffer) in saved(model) {
        if light && !["cells", "terrain"].contains(&name) {
            continue;
        }
        std::fs::write(folder.join(format!("{name}.bin")), read(gpu, buffer)).unwrap();
    }
}

/// Writes a dump into a model just built: the state after an update. The
/// model's working buffers keep what its preparation of the scene left.
fn load(gpu: &GpuContext, model: &Model, folder: &Path) {
    for (name, buffer) in saved(model) {
        let bytes = std::fs::read(folder.join(format!("{name}.bin")))
            .unwrap_or_else(|error| panic!("{name}.bin of {}: {error}", folder.display()));
        assert_eq!(bytes.len() as u64, buffer.size(), "{name} size differs");
        gpu.queue.write_buffer(buffer, 0, &bytes);
    }
}

/// The update's stages in order, each with its number of steps.
fn composition(model: &Model) -> String {
    let mut stages = Vec::new();
    for step in model.steps() {
        match stages.last_mut() {
            Some((stage, count)) if *stage == step.stage => *count += 1,
            _ => stages.push((step.stage, 1)),
        }
    }
    stages
        .iter()
        .map(|(stage, count)| format!("{} {count}", stage.label()))
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
        "OUTPUT --steps N [--dump A,B,..] [--light EVERY] [--load FOLDER,STEP] [--turn A,B,..] [--reset A,B,..] [--views] [--frames]",
    ));
    let mut steps = 0_u64;
    let mut dump: Vec<u64> = Vec::new();
    let mut light = None;
    let mut loaded: Option<(PathBuf, u64)> = None;
    let mut turns: Option<Vec<u64>> = None;
    let mut resets: Vec<u64> = Vec::new();
    let mut views = false;
    let mut frames = false;
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--steps" => steps = value().parse().expect("--steps N"),
            "--dump" => dump = numbers(&value(), "--dump"),
            "--light" => light = Some(value().parse::<u64>().expect("--light EVERY")),
            "--turn" => turns = Some(numbers(&value(), "--turn")),
            "--reset" => resets = numbers(&value(), "--reset"),
            "--views" => views = true,
            "--frames" => frames = true,
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
    let gpu = pollster::block_on(GpuContext::headless(GpuDescriptor {
        requirements: GpuRequirements {
            // WebGPU's default limits, as the app asks for.
            required_limits: Some(wgpu::Limits::default()),
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
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
    let turns = turns.unwrap_or(if loaded.is_some() { vec![] } else { vec![0] });
    let pictures = (views || frames).then(|| Pictures::new(&gpu, &model));
    // A full dump: the views first, if asked for, then the buffers.
    let dump_all = |model: &Model, step: u64| {
        let folder = output.join(format!("step-{step:05}"));
        if let Some(pictures) = pictures.as_ref().filter(|_| views) {
            pictures.write(&gpu, &folder);
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
        let mirror = model.advance_turn(&gpu.queue);
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
    let run = steps.saturating_sub(start);
    println!(
        "{run} updates ({:.0} s of the picture) in {seconds:.1} s: {:.2} ms an update, readbacks included",
        run as f64 * f64::from(DT),
        1e3 * seconds / run.max(1) as f64
    );
}
