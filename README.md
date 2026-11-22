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

The model is v2.20's but for two things ([docs/design.md](docs/design.md)
lists them: a turn is not preceded by a shake, and the rule that frees the
white's powder), and the window runs it: the product's scene, a deposit the
model laid itself, rests and is turned over as the app starts.

## Run

```sh
cargo run --release      # the window (macOS, Metal)
trunk serve              # the web build at http://127.0.0.1:8080 (a browser with WebGPU)
```

Rust's version is fixed by `rust-toolchain.toml`; `mise.toml` fixes trunk and
names the tasks (`mise run verify`, `mise run web`).

Keys: **F** turns the picture over (tipped front to back in 2 s), **S** shakes
it left and right for 3 s, **Space** stops and runs its time, **R** resets it, **V** cycles the
views (Simple, Photoreal, Flow), **H** hides or shows the control panel, **P**
moves the panel between the window's top and bottom. The panel has the same
controls, the picture's own time and its speed (seconds of the picture per
second: on an M1 Max about 0.5 for the first seconds, 0.35 while the machine is
cool, under 0.3 once it is warm).

The picture is drawn at the window's own pixels when the window is larger
than 1440 x 960 pixels, up to a pixel of one grain of the fine sand
(2727 x 1818); a smaller window shows the 1440 x 960 picture scaled down.
Built with `--features fixed-picture` (`cargo run --release --features
fixed-picture`, `trunk build --features fixed-picture`), every window shows
the 1440 x 960 picture scaled to it, as version 2.20 did.

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
and writes the picture's state (seven buffers: with them a model goes on as
the one that wrote them) before the first and after the last update into
`OUT/step-NNNNN/`. `--light EVERY` writes the cells and the bed's columns
every EVERY updates, `--load FOLDER,STEP` starts from such a dump, `--turn
A,B,..` turns the picture over and `--shake A,B,..` shakes it after the named
updates, and `--steps 0` only
prints an update's stages. An update is 8 ms of the picture and 608 dispatches
in one compute pass.

`cargo test` needs no GPU; it also runs every shader program through naga.

## Regression

```sh
cargo run --release --example check             # 0, 1 and 5 s, a second turn, a shake, a reset
cargo run --release --example check -- --full   # on to 60 s
```

compare the picture's state and the views' pixels with the hashes in
`tests/baseline/hashes.txt`, bit for bit ([tests/README.md](tests/README.md)).

## Layout

| Path | Holds |
|---|---|
| `src/lib.rs` | The library: the model, the views, and WGSL as it is compiled (`wgsl.rs`) |
| `src/view.rs`, `src/view/` | The three views and their WGSL |
| `src/model.rs`, `src/model/` | The model: its buffers (`buffers.rs`; the rows of the two row buffers in `rows.rs`), its kernels and their WGSL (one folder a module), what each kernel binds (`uses.rs`), an update as a list of steps |
| `assets/` | The product's scene: a deposit the model laid itself, and its bubbles |
| `src/model/APHROS_LICENSE.txt` | The licence of Aphros (ETH Zurich, MIT), from which `interface/interface.wgsl` takes its interface normals and plane areas |
| `tools/run.rs`, `tools/check.rs` | The model without a window (`--example run`) and the regression check (`--example check`) |
| `tests/` | The regression guide and its baseline of hashes |
| `src/bin/sand_picture.rs`, `src/bin/sand_picture/` | The window: the build with its progress, the updates, keys, the control panel, the kept picture |
| `index.html`, `Trunk.toml` | The web build |
| `records/` | Experiment records and their output: git-ignored, never committed |

## Licence

MIT ([LICENSE](LICENSE)).

One part is not this repository's own. `src/model/interface/interface.wgsl` holds routines transcribed from
[Aphros](https://github.com/cselab/aphros) (Copyright (c) 2021 ETH Zurich, MIT): the interface normals and the
plane areas, named in the file's head. Their notice is `src/model/APHROS_LICENSE.txt`; it goes wherever those
routines go. A Web build carries it, with the notices of the crates it links, as `licenses.txt`
(`tools/notices.py`, run by trunk after a build).
