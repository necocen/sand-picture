// Photoreal's distance field, a compute pass before each frame: for every
// gas identity (a layer of the view's texture) and cell, the signed distance
// in cells to that identity's outline, negative inside, its outward normal
// and the cell's gas, from the simulation's read-only geometry buffer.
// The picture draws an outline only where the field, averaged over texels
// within two cells of the pixel, is under a pixel's width (bubble_at and
// fragment, photoreal.wgsl), and a mean of texels at SHOWN or more is not.
// So the field is exact within four cells of a texel under SHOWN and flat
// (+4, the normal up) elsewhere, whatever gas lies there: gas thinner than
// a grain layer, which most cells with gas hold late in a life, shows no
// outline. Most texels are flat and stay so from frame to frame: mark_shown
// and mark_near find the others, and prepare_geometry writes only those.
@group(0) @binding(2) var<storage, read> geometry: array<vec4<f32>>;
@group(0) @binding(3) var distances: texture_storage_2d_array<rgba16float, write>;
// Per cell, the identities whose texel there is under SHOWN.
@group(0) @binding(4) var<storage, read_write> shown: array<u32>;
// Per cell, the identities shown within four cells: at this frame (bits
// 0-8), and at this frame or the last one drawn (bits 9-17).
@group(0) @binding(5) var<storage, read_write> near: array<atomic<u32>>;

// The model's rows of each identity's gas (x: the volume) and of its plane,
// by the model's own constants.
fn layer(i: u32, l: u32) -> vec4<f32> {
    return geometry[LAYER_ROW * cfg.size.x * cfg.size.y + GAS_SLOTS * i + l];
}
fn shape(i: u32, l: u32) -> vec3<f32> {
    return geometry[SHAPE_ROW * cfg.size.x * cfg.size.y + GAS_SLOTS * i + l].xyz;
}
// More than a pixel's width in cells on every grid (a quarter cell at
// most), and less than the half cell that is the least distance of a
// neighbour's outline from a cell's centre: a texel is under it only where
// the centre lies in the identity's gas, or within it of the outline that
// the cell itself holds.
const SHOWN: f32 = .4;
fn shows(i: u32, l: u32) -> bool {
    let amount = layer(i, l).x;
    if amount <= 0. { return false; }
    let s = shape(i, l);
    let inner = inner_intercept(i, l, s);
    if s.z >= 0. && inner < 0. { return true; }
    if amount >= 1. { return false; }
    let outline = min(closest_on_plane(vec2<f32>(0.), s).z,
        closest_on_plane(vec2<f32>(0.), vec3<f32>(-s.xy, -inner)).z);
    return outline < SHOWN * SHOWN;
}
// The identities each cell shows; the last frame's marks of shown gas move
// up, for mark_near to add this frame's.
@compute @workgroup_size(8, 8)
fn mark_shown(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let i = cell_index(vec2<i32>(id.xy));
    var slots = 0u;
    for (var l = 0u; l < GAS_SLOTS; l++) {
        if shows(i, l) { slots |= 1u << l; }
    }
    shown[i] = slots;
    let before = atomicLoad(&near[i]) & ((1u << GAS_SLOTS) - 1u);
    atomicStore(&near[i], before << GAS_SLOTS);
}
// A cell that shows gas marks the cells within four of it: few cells show
// any, and each of them writes its marks.
@compute @workgroup_size(8, 8)
fn mark_near(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let p = vec2<i32>(id.xy);
    let slots = shown[cell_index(p)];
    if slots == 0u { return; }
    for (var y = -4; y <= 4; y++) {
        for (var x = -4; x <= 4; x++) {
            let q = p + vec2<i32>(x, y);
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(cfg.size)) { continue; }
            atomicOr(&near[cell_index(q)], slots | (slots << GAS_SLOTS));
        }
    }
}
// Preserve each identity's inner boundary too: touching bubbles stay
// distinct.
fn inner_intercept(i: u32, l: u32, s: vec3<f32>) -> f32 {
    var alpha = -2.;
    for (var k = 0u; k < l; k++) {
        let other = shape(i, k);
        if layer(i, k).x > 0. && dot(s.xy, other.xy) > 0. {
            alpha = max(alpha, other.z);
        }
    }
    return alpha;
}

// Nearest point on the actual PLIC plane clipped to its cell. The distance
// is capped outside the local neighbourhood, where the optical cap is flat.
fn closest_on_plane(p: vec2<f32>, s: vec3<f32>) -> vec3<f32> {
    let norm2 = dot(s.xy, s.xy);
    if norm2 < 1.e-12 { return vec3<f32>(0., 0., 16.); }
    let origin = s.xy * s.z / norm2;
    let tangent = vec2<f32>(-s.y, s.x) / sqrt(norm2);
    var low = -1.e5;
    var high = 1.e5;
    for (var a = 0u; a < 2u; a++) {
        if abs(tangent[a]) < 1.e-6 {
            if abs(origin[a]) > .5 { return vec3<f32>(0., 0., 16.); }
        } else {
            let t0 = (-.5 - origin[a]) / tangent[a];
            let t1 = (.5 - origin[a]) / tangent[a];
            low = max(low, min(t0, t1));
            high = min(high, max(t0, t1));
        }
    }
    if high <= low { return vec3<f32>(0., 0., 16.); }
    let q = origin + tangent * clamp(dot(p - origin, tangent), low, high);
    return vec3<f32>(q, dot(p - q, p - q));
}

@compute @workgroup_size(8, 8)
fn prepare_geometry(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let p = vec2<i32>(id.xy);
    let i = cell_index(p);
    // A texel that was flat at the last frame and has no shown gas near it
    // is flat; one that the shown gas has left is flat again.
    let marks = atomicLoad(&near[i]);
    if ((marks >> (GAS_SLOTS + id.z)) & 1u) == 0u { return; }
    if ((marks >> id.z) & 1u) == 0u {
        textureStore(distances, p, i32(id.z), vec4<f32>(4., 0., 1., 0.));
        return;
    }
    let s = shape(i, id.z);
    let amount = layer(i, id.z).x;
    let inside = amount > 0. && s.z >= 0. && inner_intercept(i, id.z, s) < 0.;
    var nearest2 = 16.;
    var delta = vec2<f32>(0.);
    var normal = vec2<f32>(0., 1.);
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let q = p + vec2<i32>(x, y);
            if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(cfg.size)) { continue; }
            let j = cell_index(q);
            let gas = layer(j, id.z).x;
            if gas <= 0. || gas >= 1. { continue; }
            let outer = shape(j, id.z);
            let inner = vec3<f32>(-outer.xy, -inner_intercept(j, id.z, outer));
            for (var side = 0u; side < 2u; side++) {
                let plane = select(outer, inner, side == 1u);
                let sample = closest_on_plane(-vec2<f32>(f32(x), f32(y)), plane);
                if sample.z < nearest2 {
                    nearest2 = sample.z;
                    delta = -vec2<f32>(f32(x), f32(y)) - sample.xy;
                    normal = normalize(plane.xy);
                }
            }
        }
    }
    let sign = select(1., -1., inside);
    let distance = sqrt(nearest2);
    if distance > 1.e-5 && distance < 4. { normal = sign * delta / distance; }
    textureStore(distances, p, i32(id.z), vec4<f32>(sign * distance, normal, amount));
}
