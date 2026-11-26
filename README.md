# Sand Picture

A GPU simulation of a moving sand picture: water, two sands and a little air in
a thin 240 × 160 mm tank (216 × 144 grid, 1.11 mm cells). Turned over, the sand
rests on a raft of deformable bubbles, drains through its gaps as thin streams
over several minutes, and builds a striped landscape below.

Live web demo (requires a browser with WebGPU): https://suna.necocen.info/

[docs/design.md](docs/design.md) (in Japanese) tells how the picture is
modelled and computed, what is approximated and what is a convention of the
drawing. The rules themselves, with their reasons and sources, are in the
comments of the kernels (`src/model`, `src/view`).

The window runs the product's scene: a deposit the model laid itself rests
and is turned over as the app starts.

## Run

```sh
cargo run --release      # the window (macOS, Metal)
trunk serve              # the web build at http://127.0.0.1:8080 (a browser with WebGPU)
```

Rust's version is fixed by `rust-toolchain.toml`; `mise.toml` fixes trunk and
names the tasks (`mise run verify`, `mise run web`, `mise run check`,
`mise run check-rotation`, `mise run deploy`).

Keys:
- **F**: turn the picture over (180° flip with a 0.5 s gravity ramp)
- **S**: shake left and right for 3 s
- **Space**: pause or resume simulation time
- **R**: reset to the scene at rest, turned over at once
- **A**: toggle automatic turning once sand finishes falling
- **G**: toggle device tilt / accelerometer gravity (supports full 360° four-wall rotation on mobile)
- **`-` / `=`**: decrease or increase the air volume
- **V**: cycle views (Simple, Photoreal, Flow)
- **H**: hide or show the control panel (a click or tap on the picture also brings it back)
- **P**: move the panel between the window's top and bottom

The control panel exposes the same controls, unfolds **Details** (view, air
adjustment, panel position, picture time, speed and time since turn), and on
the web syncs with a Screen Wake Lock (keeps mobile screens awake while running,
releases when paused) and a top-right repository badge (hides with the panel and
moves clear of it on narrow portrait screens).

The picture is drawn at the window's own pixels when the window is larger
than 1440 × 960 pixels, up to a pixel of one grain of the fine sand
(2181 × 1454); a smaller window shows the 1440 × 960 picture scaled down.
Built with `--features fixed-picture` (`cargo run --release --features
fixed-picture`, `trunk build --features fixed-picture`), every window shows
the 1440 × 960 picture scaled to it.

The window opens at once and shows its progress while the model is built: its
materials' table takes a second or two on the CPU, and the first start after
an update also compiles the kernels.

With `SP_AGENT=1` the native app serves mikage's HTTP port on
`127.0.0.1:3939`, through which tools read the window's picture and state
(`curl -o window.png http://127.0.0.1:3939/screenshot`).

## The model without a window

```sh
cargo run --release --example run -- OUT --steps 7500 --dump 0,7500
```

runs the product's scene for 7,500 updates (60 s of the picture) on the GPU
and writes the picture's state (eight buffers: with them a model goes on as
the one that wrote them) before the first and after the last update into
`OUT/step-NNNNN/`. `--light EVERY` writes the cells and the bed's columns
every EVERY updates, `--load FOLDER,STEP` starts from such a dump, `--turn
A,B,..` turns the picture over and `--shake A,B,..` shakes it after the named
updates, and `--steps 0` only prints an update's stages. An update is 8 ms of
the picture and 543 dispatches in one compute pass.

`cargo test` needs no GPU; it also runs every shader program through naga.

## Regression

```sh
cargo run --release --example check             # 0, 1 and 5 s, a second turn, a shake, a reset
cargo run --release --example check -- --full   # on to 60 s
cargo run --release --example rotation_check    # 90° CW/CCW transpose, sideways settle, flips
```

`check` compares the picture's state and the views' pixels with the hashes in
`tests/baseline/hashes.txt`, bit for bit ([tests/README.md](tests/README.md)).

## Layout

| Path | Holds |
|---|---|
| `src/lib.rs` | The library: the model, the views, and WGSL as it is compiled (`wgsl.rs`) |
| `src/view.rs`, `src/view/` | The three views and their WGSL |
| `src/model.rs`, `src/model/` | The model: its buffers (`buffers.rs`; the rows of the two row buffers in `rows.rs`), its kernels and their WGSL (including Lagrangian foam in `foam.rs` / `foam.wgsl`), what each kernel binds (`uses.rs`), an update as a list of steps |
| `assets/` | The product's scene (`scene.bin`, `air.bin`) and OGP preview image (`ogp.png`) |
| `src/model/APHROS_LICENSE.txt` | The licence of Aphros (ETH Zurich, MIT), from which `interface/interface.wgsl` takes `intercept` and `rectangle_fraction` |
| `tools/run.rs`, `tools/check.rs`, `tools/rotation_check.rs` | Headless runner (`--example run`), bit-exact regression check (`--example check`), and 90° rotation stress test (`--example rotation_check`) |
| `tests/` | The regression guide and its baseline of hashes |
| `src/bin/sand_picture.rs`, `src/bin/sand_picture/` | The window: build progress, updates, keys, control panel, tilt/rotation sensor, kept picture |
| `index.html`, `Trunk.toml`, `wrangler.toml` | The web build and Cloudflare Workers static deployment |
| `records/` | Experiment records and their output: git-ignored, never committed |

## Licence

MIT ([LICENSE](LICENSE)).

One part is not this repository's own. `src/model/interface/interface.wgsl` holds routines transcribed from
[Aphros](https://github.com/cselab/aphros) (Copyright (c) 2021 ETH Zurich, MIT): the plane's intercept and area
fraction (`intercept`, `rectangle_fraction`), named in the file's head. Their notice is
`src/model/APHROS_LICENSE.txt`; it goes wherever those routines go. A Web build carries it, with the notices of
the crates it links, as `licenses.txt` (`tools/notices.py`, run by trunk after a build).
