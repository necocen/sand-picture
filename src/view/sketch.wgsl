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
// the sand at rest beside it, and the layer that lies on it. Other sand,
// falling or lying thinner than a body, has no surface: it is kept as its
// volume, and the picture shows its grains one by one.
//
// The layer on a body: sand that slides down a body's slope is a layer a few
// grains deep on its surface. The model's transport carries it as a cloud
// some cells deep, three grains' depth of sand thinning upward over three to
// five cells, the uppermost of it fast and thin. Drawn where the cells hold
// it, that cloud is a drift of single grains in the water over the slope,
// which no slope shows. So the sand of the cells over a body, up to where
// the water bears no more, lies in the picture where it does on a slope: in
// the cell on the body's surface, as the body's. A run of sand that reaches
// higher than SURFACE_LAYER cells is no layer but a stream that falls onto
// the body, and stays where it is.

// A cell's own facts (survey_cells).
struct Facts {
    // The cell's gas, its two largest identities' gas, and those identities
    // as first + 16 second.
    gas: vec4<f32>,
    // The grains that are not the bed's: their fine and coarse volumes, the
    // share of the cell they fill as packed sand at rest, and how far they
    // are from sliding their own width in a frame (1 at rest, 0 beyond).
    grains: vec4<f32>,
    // The grains' velocity (m/s) and the share of the white that is free
    // powder.
    motion: vec4<f32>,
}
// What the picture draws of a cell (sketch_cells).
struct Sketch {
    // As the facts' gas, with what the 3x3 cells around this one hold added
    // to the identities as 256 times: 1 for gas enough to show an outline,
    // 2 for sand that is not the bed's.
    gas: vec4<f32>,
    // The share of the cell that a body's sand fills and the black's share
    // of that sand's cross-sections; the other sand's fine and coarse
    // volumes.
    sand: vec4<f32>,
    // The other sand's velocity (m/s), the share of its white that is free
    // powder, and how far the body's grains blur: the share of a grain's
    // width they slide in a frame, at most one.
    carry: vec4<f32>,
}
@group(0) @binding(2) var<storage, read> geometry: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> facts: array<Facts>;
@group(0) @binding(4) var<storage, read_write> sketch: array<Sketch>;
@group(0) @binding(5) var<storage, read> terrain: array<vec4<f32>>;

// The exposure of a frame: one refresh of a 60 Hz display. Grains that move
// more than their size in it do not show one by one.
const EXPOSURE: f32 = 1. / 60.;
// The packing of sand at rest, the deposit's (material.rs).
const REST_PACKING: f32 = .585;
// The most cells over a body that the layer on it takes up in the model.
const SURFACE_LAYER: i32 = 8;
// The least sand (of a cell's volume) with which the water over a body
// bears the layer on: less shows a grain in one pixel of a thousand.
const LAYER_SAND: f32 = 1.e-4;

@compute @workgroup_size(8, 8)
fn survey_cells(@builtin(global_invocation_id) id: vec3<u32>) {
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
    facts[i] = Facts(
        vec4<f32>(total, first.x, second.x, first.y + 16. * second.y),
        vec4<f32>(grains, still * min((grains.x + grains.y) / REST_PACKING, 1.), still),
        vec4<f32>(velocity, powder, 0.));
}

// A cell of a body is more than half packed sand at rest.
fn body(q: vec2<i32>) -> bool { return facts[cell_index(q)].grains.z > .5; }
// Water that bears a layer's sand: a cell of the grid outside the bodies and
// the bubbles, with sand enough to show.
fn bears(q: vec2<i32>) -> bool {
    if q.y >= i32(cfg.size.y) { return false; }
    let fact = facts[cell_index(q)];
    return fact.gas.x <= .5 && fact.grains.z <= .5 && fact.grains.x + fact.grains.y >= LAYER_SAND;
}
// The cells from a body's surface up to the cell q, if q's sand is of the
// layer on the body: 1 on the surface itself, 0 for no layer. The layer is
// the run of cells that bear sand from the surface up, if it ends within
// SURFACE_LAYER cells.
fn layer_height(q: vec2<i32>) -> i32 {
    if !bears(q) { return 0; }
    var height = 0;
    for (var d = 1; d <= SURFACE_LAYER && d <= q.y; d++) {
        let below = q - vec2<i32>(0, d);
        if body(below) {
            height = d;
            break;
        }
        if !bears(below) { return 0; }
    }
    if height == 0 { return 0; }
    for (var above = 1; height + above <= SURFACE_LAYER + 1; above++) {
        if !bears(q + vec2<i32>(0, above)) { return height; }
    }
    return 0;
}

// After survey_cells: a body holds a cell's sand at rest if the cell or one
// beside it is the body's, and the cell on its surface holds its layer.
@compute @workgroup_size(8, 8)
fn sketch_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let q = vec2<i32>(id.xy);
    let i = cell_index(q);
    let own = facts[i];
    var beside = 0.;
    var gas_around = 0.;
    var sand_around = 0.;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let n = q + vec2<i32>(x, y);
            if any(n < vec2<i32>(0)) || any(n >= vec2<i32>(cfg.size)) { continue; }
            let around = facts[cell_index(n)];
            beside = max(beside, around.grains.z);
            gas_around += around.gas.x;
            sand_around += around.grains.x + around.grains.y;
        }
    }
    // The body's sand in this cell (fine, coarse), the part of it that
    // slides a grain's width in a frame, and the other sand.
    var held = own.grains.w * smoothstep(.5, .75, beside) * own.grains.xy;
    var slid = (1. - own.grains.w) * (held.x + held.y);
    var other = own.grains.xy - held;
    let height = layer_height(q);
    if height == 1 {
        held = vec2<f32>(0.);
        slid = 0.;
        other = vec2<f32>(0.);
        for (var above = 0; above < SURFACE_LAYER && bears(q + vec2<i32>(0, above)); above++) {
            let lying = facts[cell_index(q + vec2<i32>(0, above))].grains;
            held += lying.xy;
            slid += (1. - lying.w) * (lying.x + lying.y);
        }
    } else if height > 1 {
        held = vec2<f32>(0.);
        other = vec2<f32>(0.);
    }
    // A floor-connected bed cell right beside a packed mobile body (for
    // example, while bubbles delay capture of an adjacent column after a
    // mid-drainage turn) joins the body's smooth field so the B-spline does
    // not fall below half and open a dark slit along the bed-body seam.
    // Bed cells with no mobile body beside them have beside <= 0.5 and stay
    // excluded, leaving the bed's own crisp surface untouched.
    let column = terrain[id.x];
    if column.x > 0. && f32(id.y) <= column.y {
        held += smoothstep(.5, .75, beside) * max(cells[i].bed.xy, vec2<f32>(0.));
    }
    let sections = held / cfg.diameters;
    // An outline needs the gas of the cells around a point to weigh a half;
    // sand under a ten-thousandth of the cells shows no grain.
    let reach = select(0., 1., gas_around > .3) + select(0., 2., sand_around > 1.e-4);
    sketch[i] = Sketch(
        vec4<f32>(own.gas.xyz, own.gas.w + 256. * reach),
        vec4<f32>(min((held.x + held.y) / REST_PACKING, 1.),
            sections.y / max(sections.x + sections.y, 1.e-12), other),
        vec4<f32>(own.motion.xyz, slid / max(held.x + held.y, 1.e-12)));
}
