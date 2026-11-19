# Sand Picture

A GPU simulation of a moving sand picture: water, two sands and a little air in
a thin tank. Turned over, the sand rests on a raft of bubbles, drains through
its gaps as thin streams for about a quarter of an hour, and builds a striped
landscape below.

This repository is a rewrite, from an empty tree, of the model of the earlier
`sand-picture` repository (its version 2.20): the same computation in a plainer
program. [docs/design.md](docs/design.md) (in Japanese) holds the design, the
order of the work and each stage's results; the earlier repository stays the
reference for the physics.

The model is ported and gives v2.20's results bit for bit. The window does not
show it yet: the app is the next stage.

## Run

```sh
cargo run --release      # the window (macOS, Metal)
trunk serve              # the web build at http://127.0.0.1:8080 (a browser with WebGPU)
```

Rust's version is fixed by `rust-toolchain.toml`; `mise.toml` fixes trunk and
names the tasks (`mise run verify`, `mise run web`).

Keys: **V** cycles the views (Simple, Photoreal, Flow), **H** hides or shows
the control panel, **P** moves the panel between the window's top and bottom.

With `SP_AGENT=1` the native app serves mikage's HTTP port on
`127.0.0.1:3939`, through which tools read the window's picture and state
(`curl -o window.png http://127.0.0.1:3939/screenshot`).

## The model without a window

```sh
cargo run --release --example run -- OUT --steps 7500 --dump 0,7500
```

runs the product's scene for 7,500 updates (60 s of the picture) on the GPU
and writes the model's buffers before the first and after the last update
into `OUT/step-NNNNN/`. `--light EVERY` writes the cells and the bed's columns
every EVERY updates, `--load FOLDER,STEP` starts from such a dump, `--turn
A,B,..` turns the picture over after the named updates, and `--steps 0` only
prints an update's stages. An update is 8 ms of the picture and 608 dispatches
in one compute pass.

`cargo test` needs no GPU; it also runs every shader program through naga.

## Layout

| Path | Holds |
|---|---|
| `src/lib.rs`, `src/view.rs` | The library: the model and the views |
| `src/model.rs`, `src/model/` | The model: its buffers, its kernels and their WGSL (one folder a module), an update as a list of steps |
| `assets/` | The product's scene: a deposit the model laid itself, and its bubbles |
| `tools/run.rs` | The model without a window (`--example run`) |
| `src/bin/sand_picture.rs`, `src/bin/sand_picture/` | The window: keys, the control panel, the frame |
| `index.html`, `Trunk.toml` | The web build |
| `records/` | Experiment records and their output: git-ignored, never committed |
