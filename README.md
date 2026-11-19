# Sand Picture

A GPU simulation of a moving sand picture: water, two sands and a little air in
a thin tank. Turned over, the sand rests on a raft of bubbles, drains through
its gaps as thin streams for about a quarter of an hour, and builds a striped
landscape below.

This repository is a rewrite, from an empty tree, of the model of the earlier
`sand-picture` repository (its version 2.20): the same computation in a plainer
program. [docs/design.md](docs/design.md) (in Japanese) holds the design and
the order of the work; the earlier repository stays the reference for the
physics until the port is complete.

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

## Layout

| Path | Holds |
|---|---|
| `src/lib.rs`, `src/view.rs` | The library: the views (the model comes with the port) |
| `src/bin/sand_picture.rs`, `src/bin/sand_picture/` | The window: keys, the control panel, the frame |
| `index.html`, `Trunk.toml` | The web build |
| `records/` | Experiment records and their output: git-ignored, never committed |
