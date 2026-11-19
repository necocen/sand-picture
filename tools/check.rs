//! The regression check: runs the product's scene without a window and
//! compares what it computes and draws with the hashes in
//! `tests/baseline/hashes.txt`.
//! `cargo run --release --example check [-- --full] [--record FOLDER]`
//!
//! The picture rests, is turned over at once and runs to 5 s, with
//! checkpoints at 0, 1 and 5 s (updates 0, 125 and 625); it is then turned
//! over again and checked a second later; then it is reset, and the same
//! three checkpoints must come out as before. `--full` runs on from there to
//! 15, 30 and 60 s. A checkpoint is the SHA-256 of the eight compared
//! buffers, of the strata and of the three views' pictures; the labels are
//! left out (their records are numbered in the order the GPU's threads reach
//! them). `--record FOLDER` also writes what was found, with the pictures as
//! PNG, as a candidate for a new baseline. Run outside a sandbox: it needs
//! the GPU.
mod common;

use common::{Pictures, headless, picture_name, read, wait, write_png};
use mikage::GpuContext;
use sand_picture::model::{Model, Scene};
use sand_picture::view::{Mode, REFERENCE_SIZE};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const BASELINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/baseline/hashes.txt");

/// A checkpoint's fields by name, each with its SHA-256.
type Fields = BTreeMap<String, String>;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The state and the pictures now. The views are drawn before the buffers
/// are read, so the buffers' hashes also show that no view writes to them.
fn checkpoint(
    gpu: &GpuContext,
    model: &Model,
    pictures: &Pictures,
    record: Option<&Path>,
    name: &str,
) -> Fields {
    let mut fields = Fields::new();
    for mode in Mode::ALL {
        let pixels = pictures.picture(gpu, mode);
        if let Some(folder) = record {
            let file = format!("{}-{}.png", name.replace(' ', "-"), picture_name(mode));
            write_png(&folder.join(file), &pixels, pictures.size());
        }
        fields.insert(picture_name(mode).to_owned(), hash(&pixels));
    }
    let buffers = model.buffers();
    for (field, buffer) in buffers
        .named()
        .into_iter()
        .chain([("strata", &buffers.strata)])
    {
        fields.insert(field.to_owned(), hash(&read(gpu, buffer)));
    }
    fields
}

/// Runs the model's updates up to update `to`.
fn advance(gpu: &GpuContext, model: &mut Model, update: &mut u64, to: u64) {
    while *update < to {
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
        *update += 1;
        if update.is_multiple_of(8) {
            wait(gpu);
        }
    }
    wait(gpu);
}

/// The baseline: `checkpoint<TAB>field<TAB>hash` a line, `#` for comments.
fn baseline() -> BTreeMap<String, Fields> {
    let mut checkpoints: BTreeMap<String, Fields> = BTreeMap::new();
    let text = std::fs::read_to_string(BASELINE).unwrap_or_default();
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let mut parts = line.split('\t');
        if let (Some(name), Some(field), Some(hash)) = (parts.next(), parts.next(), parts.next()) {
            checkpoints
                .entry(name.to_owned())
                .or_default()
                .insert(field.to_owned(), hash.to_owned());
        }
    }
    checkpoints
}

fn main() {
    let mut full = false;
    let mut record: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--full" => full = true,
            "--record" => record = Some(args.next().expect("--record FOLDER").into()),
            other => panic!("unknown argument {other:?}"),
        }
    }
    if let Some(folder) = &record {
        std::fs::create_dir_all(folder).unwrap();
    }
    let gpu = headless();
    let mut model = Model::new(&gpu.device, &gpu.queue, Scene::product());
    let pictures = Pictures::new(&gpu, &model, REFERENCE_SIZE);
    let mut found: Vec<(String, Fields)> = Vec::new();
    let mut failures = 0;
    let mut note = |name: String, fields: Fields| found.push((name, fields));

    // As in the app, the picture is turned over at once.
    model.start_turn();
    let mut update = 0;
    let mut first = Vec::new();
    for to in [0, 125, 625] {
        advance(&gpu, &mut model, &mut update, to);
        let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), &format!("{to}"));
        first.push(fields.clone());
        note(format!("{to}"), fields);
    }
    // The person turns the picture over again.
    model.start_turn();
    advance(&gpu, &mut model, &mut update, 750);
    let name = "turned 750";
    let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), name);
    note(name.to_owned(), fields);
    // A reset: the same three checkpoints as from the start.
    model.reset(&gpu.device, &gpu.queue);
    model.start_turn();
    update = 0;
    for (to, before) in [0, 125, 625].into_iter().zip(&first) {
        advance(&gpu, &mut model, &mut update, to);
        let fields = checkpoint(&gpu, &model, &pictures, None, "after a reset");
        for (field, hash) in &fields {
            if before[field] != *hash {
                println!("after a reset, update {to}: {field} DIFFERS from the first run");
                failures += 1;
            }
        }
    }
    if full {
        for to in [1875, 3750, 7500] {
            advance(&gpu, &mut model, &mut update, to);
            let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), &format!("{to}"));
            note(format!("{to}"), fields);
        }
    }

    let baseline = baseline();
    for (name, fields) in &found {
        let differing: Vec<&str> = fields
            .iter()
            .filter(|(field, hash)| {
                baseline.get(name).and_then(|known| known.get(*field)) != Some(*hash)
            })
            .map(|(field, _)| field.as_str())
            .collect();
        if differing.is_empty() {
            println!("{name}: {} fields the same", fields.len());
        } else {
            println!("{name}: DIFFERS in {}", differing.join(", "));
            failures += differing.len();
        }
    }
    if let Some(folder) = &record {
        let adapter = gpu.capabilities().adapter_info.clone();
        let mut text = format!(
            "# The regression baseline: checkpoint, field, SHA-256. Recorded by `check --record`.\n# {} ({:?}), {}\n",
            adapter.name,
            adapter.backend,
            std::env::consts::OS
        );
        for (name, fields) in &found {
            for (field, hash) in fields {
                text.push_str(&format!("{name}\t{field}\t{hash}\n"));
            }
        }
        std::fs::write(folder.join("hashes.txt"), text).unwrap();
        println!("recorded in {}", folder.display());
    }
    if failures > 0 {
        println!("{failures} differences");
        std::process::exit(1);
    }
}
