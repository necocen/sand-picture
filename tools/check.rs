//! The regression check: runs the product's scene without a window and
//! compares what it computes and draws with the hashes in
//! `tests/baseline/hashes.txt`.
//! `cargo run --release --example check [-- --full] [--record FOLDER]`
//!
//! The picture rests, is turned over at once and runs to 5 s, with
//! checkpoints at 0, 1 and 5 s (updates 0, 125 and 625: at rest, as the turn
//! mirrors the grid, and three seconds after the turn); it is then turned
//! over again and checked a second later, as the grid is mirrored with the
//! sand up; then it is reset and, at rest, shaken, and checked a second into
//! the shake; then it is reset, and the first three checkpoints must come
//! out as before; then it is reset once more and given the state it had one
//! update before the turn mirrored the grid, and must reach 5 s as before:
//! the state alone carries the picture on. And pictures of another size must
//! be the same whether the views were made at that size or brought to it, as
//! a window that is resized brings them.
//! `--full` runs on from there to 15, 30 and 60 s. A checkpoint is the
//! SHA-256 of the buffers of the picture's state and of the three views'
//! pictures; of the labels only the parts two runs agree in are taken (the
//! bubbles' records are numbered in the order the GPU's threads reach
//! them). `--record FOLDER`
//! also writes what was found, with the pictures as PNG, as a candidate for
//! a new baseline. Run outside a sandbox: it needs the GPU.
mod common;

use common::{Pictures, headless, picture_name, read, wait, write_png};
use mikage::GpuContext;
use sand_picture::model::{Model, Scene};
use sand_picture::view::{Mode, REFERENCE_SIZE, Views};
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
    for (field, buffer) in buffers.state() {
        let mut bytes = read(gpu, buffer);
        if field == "labels" {
            // Only what two runs agree in.
            let agreeing = buffers.agreeing_labels();
            bytes = agreeing
                .into_iter()
                .flat_map(|part| &bytes[part])
                .copied()
                .collect();
        }
        fields.insert(field.to_owned(), hash(&bytes));
    }
    fields
}

/// Tells which of `fields` differ from `before`, and how many.
fn differences(what: &str, fields: &Fields, before: &Fields) -> usize {
    let differing: Vec<&str> = fields
        .iter()
        .filter(|(field, hash)| before.get(*field) != Some(*hash))
        .map(|(field, _)| field.as_str())
        .collect();
    if !differing.is_empty() {
        println!("{what}: DIFFERS in {}", differing.join(", "));
    }
    differing.len()
}

/// Runs the model's updates up to update `to`, and says which update
/// mirrored the grid, if one did.
fn advance(gpu: &GpuContext, model: &mut Model, update: &mut u64, to: u64) -> Option<u64> {
    let mut mirrored = None;
    while *update < to {
        let mirror = model.advance_motion(&gpu.queue);
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
        if mirror {
            mirrored = Some(*update);
        }
        if update.is_multiple_of(8) {
            wait(gpu);
        }
    }
    wait(gpu);
    mirrored
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
    let mut mirror = None;
    for to in [0, 125, 625] {
        mirror = mirror.or(advance(&gpu, &mut model, &mut update, to));
        let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), &format!("{to}"));
        first.push(fields.clone());
        note(format!("{to}"), fields);
    }
    let mirror = mirror.expect("a turn mirrors the grid");
    // The person turns the picture over again.
    model.start_turn();
    advance(&gpu, &mut model, &mut update, 750);
    let name = "turned 750";
    let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), name);
    note(name.to_owned(), fields);
    // The person shakes the picture as it rests.
    model.reset(&gpu.device, &gpu.queue);
    model.start_shake();
    let mut shaken = 0;
    advance(&gpu, &mut model, &mut shaken, 125);
    let name = "shaken 125";
    let fields = checkpoint(&gpu, &model, &pictures, record.as_deref(), name);
    note(name.to_owned(), fields);
    // A reset: the same three checkpoints as from the start. On the way the
    // state is kept as it is one update before the grid is mirrored.
    model.reset(&gpu.device, &gpu.queue);
    model.start_turn();
    update = 0;
    let mut kept = Vec::new();
    for (to, before) in [0, 125, 625].into_iter().zip(&first) {
        if kept.is_empty() && to >= mirror {
            advance(&gpu, &mut model, &mut update, mirror - 1);
            kept = model
                .buffers()
                .state()
                .map(|(_, buffer)| read(&gpu, buffer))
                .to_vec();
        }
        advance(&gpu, &mut model, &mut update, to);
        let fields = checkpoint(&gpu, &model, &pictures, None, "after a reset");
        failures += differences(&format!("after a reset, update {to}"), &fields, before);
    }
    // The state alone carries the picture on. The model is reset, so that
    // its other buffers hold what another time left, and given the kept
    // state; its turn, which is not in the buffers, is brought to the same
    // point by advancing a new one without running updates. Its next update
    // mirrors the grid, and at 5 s it must be where the first run was.
    model.reset(&gpu.device, &gpu.queue);
    for ((_, buffer), bytes) in model.buffers().state().into_iter().zip(&kept) {
        gpu.queue.write_buffer(buffer, 0, bytes);
    }
    model.start_turn();
    for _ in 1..mirror {
        model.advance_motion(&gpu.queue);
    }
    update = mirror - 1;
    advance(&gpu, &mut model, &mut update, 625);
    let fields = checkpoint(&gpu, &model, &pictures, None, "from the kept state");
    failures += differences("from the kept state, update 625", &fields, &first[2]);
    // A window that is resized: the views brought to another size draw what
    // views made at that size draw.
    let larger = [2160, 1440];
    let views = Views::new(&gpu.device, model.buffers(), REFERENCE_SIZE);
    views.resize(&gpu.queue, larger);
    let brought = Pictures::of(&gpu, views, larger);
    let made = Pictures::new(&gpu, &model, larger);
    for mode in Mode::ALL {
        if brought.picture(&gpu, mode) != made.picture(&gpu, mode) {
            println!("after a resize: {} DIFFERS", picture_name(mode));
            failures += 1;
        }
    }
    if full {
        // The run goes on from the state that was kept and given back.
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
