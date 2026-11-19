# Regression

`cargo run --release --example check` (`mise run check`) runs the product's
scene without a window and compares what the model computes and what the
views draw with `tests/baseline/hashes.txt`, bit for bit. It needs the GPU, so
run it outside a sandbox. The app and the check use the same model, the same
views and WebGPU's default limits; the check holds no second physics.

What a run does, in the picture's own time (an update is 8 ms):

| Step | Checkpoints |
|---|---|
| The picture rests, is turned over at once and runs | 0 s, 1 s (the shake), 5 s (the turn ends): updates 0, 125, 625 |
| It is turned over again | 6 s: `turned 750` |
| It is reset, as with the app's Reset | 0, 1 and 5 s again: they must equal the first three |
| `--full` (`mise run check-full`) runs on | 15, 30 and 60 s: updates 1875, 3750, 7500 |

A checkpoint is the SHA-256 of twelve things: the eight buffers `cells`,
`geometry`, `faces`, `stream`, `transport`, `terrain`, `members` and
`schedule`, the `strata`, and the raw RGBA pixels of the three views at
1440 x 960 (`image` is Photoreal, then `simple` and `flow`). The views are
drawn before the buffers are read, so a view that wrote to the model would
show. The `labels` buffer is left out: the records of its bubbles are numbered
in the order the GPU's threads reach them, and two runs of one build differ
there while everything else is equal.

## When the check fails

A change that is meant to keep the behaviour (a refactoring, a speed-up that
keeps the order of the arithmetic, a change to a view alone) has to pass. If
it does not, it changes the behaviour and is treated as such: the baseline is
not changed to make a check pass.

GPU floating point depends on the order of operations and on how the compiler
fuses them, and Metal compiles with fast math: a rewrite that is the same
mathematics can change the last bits, and in this flow a last bit is a
different local arrangement a second later. Such a change is judged by
comparing ensembles of runs, as `../sand-picture/tests/README.md` describes
under "Rounding-level changes", not by this check.

## A new baseline

    cargo run --release --example check -- --full --record records/baseline

writes `records/baseline/hashes.txt` and every checkpoint's three pictures as
PNG (`records/` is ignored by git). Look at the pictures, at 1, 5, 15, 30 and
60 s in that order, and copy the hashes to `tests/baseline/hashes.txt` only
then. Pictures are not kept in the repository.

The baseline holds for one environment: the hashes here were recorded on an
Apple M1 Max with macOS 27.2.1, Rust 1.95.0 and wgpu and naga 30.0.1. Before any
of these is raised, record a candidate in the new environment first, with the
code unchanged.

The first baseline (2026-11-19) is v2.20 of `../sand-picture`: its 60 entries
that v2.20's baseline also has (the eight buffers, Photoreal and Simple at
the six times) are equal to it.
