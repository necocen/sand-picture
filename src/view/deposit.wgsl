// The floor-connected bed drawn from its strata record (bed/strata.wgsl):
// below the bed surface, interpolated between column centres, the material is
// packed sand whose coarse share comes from the slices laid down there.
// Between two columns each layer is a past surface of the bed, interpolated
// like the present one, so layers that slope across columns stay continuous
// and follow the old slopes.
@group(0) @binding(4) var<storage, read> strata: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> terrain: array<vec4<f32>>;
// The record's layout and the deposit's packing, as bed/strata.wgsl and
// material.rs have them.
const STRATA_RES: u32 = 8u;
const STRATA_HEADER: u32 = 3u;
const BED_PACKING: f32 = .585;

fn strata_base(x: u32) -> u32 { return x * (STRATA_HEADER + cfg.size.y * STRATA_RES); }
fn slices(x: u32) -> u32 {
    return min(u32(ceil(strata[strata_base(x)].x * f32(STRATA_RES))), cfg.size.y * STRATA_RES);
}
// (coarse share, time laid down) at `height` cells above the floor, linear
// between slice centres within the recorded slices.
fn slice_at(x: u32, height: f32) -> vec2<f32> {
    let last = f32(max(slices(x), 1u) - 1u);
    let s = clamp(height * f32(STRATA_RES) - .5, 0., last);
    let k = u32(floor(s));
    let base = strata_base(x) + STRATA_HEADER;
    return mix(strata[base + k], strata[base + min(k + 1u, u32(last))], s - f32(k));
}
fn record(x: u32, k: u32) -> vec2<f32> { return strata[strata_base(x) + STRATA_HEADER + k]; }
// Whether slice k of column x belongs to a block captured at once (at least
// half a cell laid at the same time): its timestamp marks when the column was
// captured, not a past surface rising through the block.
fn in_block(x: u32, k: u32) -> bool {
    let n = slices(x);
    if k >= n { return false; }
    let t = record(x, k).y;
    let half = STRATA_RES / 2u;
    return (k >= half && record(x, k - half).y == t)
        || (k + half < n && record(x, k + half).y == t);
}
// The last slice of column x laid down by `time` (0 if none).
fn slice_by_time(x: u32, time: f32) -> u32 {
    let n = slices(x);
    if n == 0u { return 0u; }
    var lo = 0u;
    var hi = n - 1u;
    while lo < hi {
        let mid = (lo + hi + 1u) / 2u;
        if record(x, mid).y <= time { lo = mid; } else { hi = mid - 1u; }
    }
    return lo;
}
// Time of the slice after k, or never.
fn next_time(x: u32, k: u32) -> f32 {
    if k + 1u >= slices(x) { return 3.4e38; }
    return record(x, k + 1u).y;
}
// Height (cells) and coarse share of column x's surface at `time`, given k,
// its last slice laid down by then: linear between slice centres.
fn surface_at(x: u32, k: u32, time: f32) -> vec2<f32> {
    let n = slices(x);
    if n == 0u { return vec2<f32>(0., .5); }
    let a = record(x, k);
    if time < a.y { return vec2<f32>(0., a.x); }
    if k + 1u >= n { return vec2<f32>(strata[strata_base(x)].x, a.x); }
    let b = record(x, k + 1u);
    var f = 0.;
    if b.y > a.y { f = clamp((time - a.y) / (b.y - a.y), 0., 1.); }
    return vec2<f32>((f32(k) + .5 + f) / f32(STRATA_RES), mix(a.x, b.x, f));
}
// The coarse share of the sand at p as the cells hold it, for a bed whose
// record has no slice there.
fn cell_share(p: vec2<f32>) -> f32 {
    let q = vec2<i32>(floor(p));
    let t = fract(p);
    let c = mix(mix(composition(q), composition(q + vec2<i32>(1, 0)), t.x),
        mix(composition(q + vec2<i32>(0, 1)), composition(q + vec2<i32>(1)), t.x), t.y);
    return c.z / max(1.e-6, c.y + c.z);
}
// The floor-connected bed at p (cell coordinates, cell centres at integers):
// how much of the pixel lies under its surface, and the coarse share of the
// sand laid down there.
struct Bed { inside: f32, share: f32 }
fn bed_at(p: vec2<f32>) -> Bed {
    let x0 = clamp(i32(floor(p.x)), 0, i32(cfg.size.x) - 1);
    let x1 = min(x0 + 1, i32(cfg.size.x) - 1);
    let w = clamp(p.x - f32(x0), 0., 1.);
    let height = p.y + .5;
    let surface = mix(terrain[u32(x0)].x, terrain[u32(x1)].x, w);
    let pixel = f32(cfg.size.x) / f32(cfg.image_size.x);
    let inside = 1. - smoothstep(-.5 * pixel, .5 * pixel, height - surface);
    if inside <= 0. { return Bed(0., 0.); }
    let in0 = height < strata[strata_base(u32(x0))].x;
    let in1 = height < strata[strata_base(u32(x1))].x;
    if !in0 && !in1 { return Bed(inside, cell_share(p)); }
    // The layer through this point is the past surface, interpolated between
    // the two columns, that passes through it: the time T at which
    // mix(H0(T), H1(T), w) = height. It lies between the times at which each
    // column reached this height (now, for a column that has not). Between
    // consecutive slice times of either column the interpolated surface rises
    // linearly, so walk those times in order and solve the crossing segment.
    let a = u32(x0);
    let b = u32(x1);
    // Sand laid as a block at once (the factory's deposit, or a resting slab
    // captured after a turn—possibly on different updates in adjacent columns
    // as bubbles lift off it) has no past surface within the block to follow:
    // every slice of the block carries the capture time, so the time walk
    // below would lock onto the block's ends and smear them vertically.
    // Interpolate its slices horizontally at this height instead.
    let k = min(u32(height * f32(STRATA_RES)), cfg.size.y * STRATA_RES - 1u);
    let block0 = in0 && in_block(a, k);
    let block1 = in1 && in_block(b, k);
    if in0 && in1 && (block0 || block1 || record(a, k).y == record(b, k).y) {
        return Bed(inside, mix(slice_at(a, height).x, slice_at(b, height).x, w));
    }
    if block0 && !in1 { return Bed(inside, slice_at(a, height).x); }
    if block1 && !in0 { return Bed(inside, slice_at(b, height).x); }
    let now = strata[strata_base(a)].y;
    let t0 = select(now, slice_at(a, height).y, in0);
    let t1 = select(now, slice_at(b, height).y, in1);
    let end = max(t0, t1);
    var time = min(t0, t1);
    var k0 = slice_by_time(a, time);
    var k1 = slice_by_time(b, time);
    var gap = mix(surface_at(a, k0, time).x, surface_at(b, k1, time).x, w) - height;
    for (var step = 0u; step < 64u && gap < 0. && time < end; step++) {
        let n0 = next_time(a, k0);
        let n1 = next_time(b, k1);
        let next = min(min(n0, n1), end);
        let rise = mix(surface_at(a, k0, next).x, surface_at(b, k1, next).x, w) - height;
        if rise >= 0. {
            if rise > gap { time += (next - time) * clamp(-gap / (rise - gap), 0., 1.); }
            break;
        }
        time = next;
        if n0 <= next { k0 = slice_by_time(a, next); }
        if n1 <= next { k1 = slice_by_time(b, next); }
        gap = mix(surface_at(a, k0, time).x, surface_at(b, k1, time).x, w) - height;
    }
    return Bed(inside, mix(surface_at(a, k0, time).y, surface_at(b, k1, time).y, w));
}
