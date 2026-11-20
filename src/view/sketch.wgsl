// Photoreal's sketch of the state, a compute pass before each picture: per
// cell, what the picture draws there as smooth fields (photoreal.wgsl). What
// concerns a whole cell is decided here, once a cell; the picture only
// interpolates. Display only: every simulation buffer is read-only.
//
// The gas of a cell is its sum over the identities, with the two largest
// named: a cell where a film runs holds two bubbles, and the traces of
// further identities that the transport leaves behind are no bubbles.
//
// The sand of a cell is drawn in one of two ways. A packed body of sand at
// rest has a sharp surface, and a cell at that surface holds the body's sand
// in a part of itself: such sand is kept as the share of the cell it fills,
// and the picture draws the body to where that share is a half. A body holds
// the sand at rest beside it and the sand that lies on it, at rest or
// sliding down its slope. Sand that falls, or lies thinner than a body, has
// no surface: it is kept as its volume and veils the water.
struct Sketch {
    // The cell's gas, its two largest identities' gas, and those identities
    // as first + 16 second.
    gas: vec4<f32>,
    // The grains that are not the bed's: their fine and coarse volumes, the
    // share of the cell they fill as packed sand at rest, and the black's
    // share of their cross-sections.
    grains: vec4<f32>,
    // The grains' velocity (m/s), the share of the white that is free powder,
    // and what the 3x3 cells around this one hold: 1 for gas enough to show
    // an outline, 2 for sand that is not the bed's.
    motion: vec4<f32>,
    // How the picture draws the grains: the share of the cell that a body's
    // sand fills, and the other sand's fine and coarse volumes.
    sand: vec4<f32>,
}
@group(0) @binding(2) var<storage, read> geometry: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sketch: array<Sketch>;
@group(0) @binding(5) var<storage, read> terrain: array<vec4<f32>>;

// The exposure of a frame: one refresh of a 60 Hz display. Grains that move
// more than their size in it do not show one by one.
const EXPOSURE: f32 = 1. / 60.;
// The packing of sand at rest, the deposit's (material.rs).
const REST_PACKING: f32 = .585;

// A cell's own part of its sketch.
@compute @workgroup_size(8, 8)
fn sketch_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let i = cell_index(vec2<i32>(id.xy));
    let c = cells[i];
    var total = 0.;
    var first = vec2<f32>(0.);
    var second = vec2<f32>(0.);
    for (var l = 0u; l < GAS_SLOTS; l++) {
        let amount = max(geometry[LAYER_ROW * cfg.size.x * cfg.size.y + GAS_SLOTS * i + l].x, 0.);
        total += amount;
        if amount > first.x {
            second = first;
            first = vec2<f32>(amount, f32(l));
        } else if amount > second.x {
            second = vec2<f32>(amount, f32(l));
        }
    }
    // The grains that are not drawn with the bed: the free grains, and a
    // deposit outside its floor-connected column (right after a turn, before
    // the next update releases it).
    let free = max(vec2<f32>(c.volume.fine, c.volume.coarse), vec2<f32>(0.));
    var grains = free;
    let column = terrain[id.x];
    if column.x <= 0. || f32(id.y) > column.y { grains += max(c.bed.xy, vec2<f32>(0.)); }
    var velocity = vec2<f32>(0.);
    if free.x + free.y > 0. { velocity = (c.momentum[1] + c.momentum[2]) / (free.x + free.y); }
    let still = 1. - clamp(length(velocity) * EXPOSURE / cfg.diameters.x, 0., 1.);
    // The white's free powder scatters by its own cross-section; powder bound
    // to its grit lies in the grit's aggregates and scatters with them.
    let white = max(c.volume.fine, 0.) + max(c.bed.x, 0.);
    var powder = 0.;
    if white > 0. { powder = clamp(max(c.powder.free, 0.) / white, 0., 1.); }
    let sections = grains * vec2<f32>((1. - powder) / cfg.diameters.x + powder / cfg.powder_diameter, 1. / cfg.diameters.y);
    var black = 0.;
    if grains.x + grains.y > 0. { black = sections.y / (sections.x + sections.y); }
    sketch[i] = Sketch(
        vec4<f32>(total, first.x, second.x, first.y + 16. * second.y),
        vec4<f32>(grains, still * min((grains.x + grains.y) / REST_PACKING, 1.), black),
        vec4<f32>(velocity, powder, 0.),
        vec4<f32>(0.));
}

// What a cell's sketch takes from the cells around it, after sketch_cells.
// A cell of a body is more than half packed sand at rest. A body holds this
// cell's sand at rest if the cell or one beside it is the body's, and this
// cell's sand at any speed if it lies on the body: within two cells above
// one of the body's.
@compute @workgroup_size(8, 8)
fn sketch_bodies(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let q = vec2<i32>(id.xy);
    let i = cell_index(q);
    var beside = 0.;
    var under = 0.;
    var gas_around = 0.;
    var sand_around = 0.;
    for (var y = -2; y <= 1; y++) {
        for (var x = -2; x <= 2; x++) {
            let n = q + vec2<i32>(x, y);
            if any(n < vec2<i32>(0)) || any(n >= vec2<i32>(cfg.size)) { continue; }
            let around = sketch[cell_index(n)];
            if abs(x) <= 1 && y >= -1 {
                beside = max(beside, around.grains.z);
                gas_around += around.gas.x;
                sand_around += around.grains.x + around.grains.y;
            }
            if y < 0 && abs(x) <= -y { under = max(under, around.grains.z); }
        }
    }
    let own = sketch[i];
    let sand = own.grains.x + own.grains.y;
    let still = 1. - clamp(length(own.motion.xy) * EXPOSURE / cfg.diameters.x, 0., 1.);
    let held = max(still * smoothstep(.5, .75, beside), smoothstep(.5, .75, under));
    sketch[i].sand = vec4<f32>(held * min(sand / REST_PACKING, 1.), (1. - held) * own.grains.xy, 0.);
    // An outline needs the gas of the cells around a point to weigh a half;
    // sand under a ten-thousandth of the cells veils nothing.
    sketch[i].motion.w = select(0., 1., gas_around > .3) + select(0., 2., sand_around > 1.e-4);
}
