// The Photoreal picture, drawn per pixel from the sketch of the state
// (sketch.wgsl): the wet sand over the backdrop (wet_at) and over it the
// gas, each bubble a thin meniscus that reflects and refracts the studio
// (bubble_color). Display only: every simulation buffer is read-only.
//
// Gas and sand are drawn as smooth fields: a quadratic B-spline through the
// cells' values, which has no corner at a cell's edge and no feature finer
// than a cell. A cell is about a gap wide, and neither a bubble's meniscus
// nor a body of sand between the plates shows a finer in-plane feature. A
// field's half level is the outline of what fills the cells, a body of sand
// or a bubble alone: where a straight surface cuts a cell, the level lies
// within a tenth of a cell of it. Bubbles that touch part the gas between
// them (bubble_at).
struct Sketch { gas: vec4<f32>, sand: vec4<f32>, carry: vec4<f32> }
@group(0) @binding(2) var<storage, read> sketch: array<Sketch>;

struct FoamBubble {
    target_area: f32,
    drain_time: f32,
    slot: u32,
    alive: u32,
    centroid: vec2<f32>,
    vel: vec2<f32>,
    bbox_min: vec2<f32>,
    bbox_max: vec2<f32>,
    drain_partner: u32,
    held: u32,
    pad0: u32,
    pad1: u32,
    verts: array<vec2<f32>, FOAM_VERTS>,
}
@group(0) @binding(6) var<storage, read> foam: array<FoamBubble>;

// The weights of the 3x3 cells around a point, and their derivatives.
struct Stencil { cell: vec2<i32>, wx: vec3<f32>, wy: vec3<f32>, dx: vec3<f32>, dy: vec3<f32> }
fn stencil(p: vec2<f32>) -> Stencil {
    let c = floor(p + .5);
    let t = p - c;
    return Stencil(vec2<i32>(c),
        vec3<f32>(.5 * (.5 - t.x) * (.5 - t.x), .75 - t.x * t.x, .5 * (.5 + t.x) * (.5 + t.x)),
        vec3<f32>(.5 * (.5 - t.y) * (.5 - t.y), .75 - t.y * t.y, .5 * (.5 + t.y) * (.5 + t.y)),
        vec3<f32>(t.x - .5, -2. * t.x, t.x + .5),
        vec3<f32>(t.y - .5, -2. * t.y, t.y + .5));
}
fn sketch_at(k: Stencil, i: i32, j: i32) -> Sketch {
    return sketch[cell_index(k.cell + vec2<i32>(i - 1, j - 1))];
}

// What a cell's sketch says of the cells around it (sketch.wgsl).
const GAS_NEAR: u32 = 1u;
const SAND_NEAR: u32 = 2u;
fn near(k: Stencil, what: u32) -> bool {
    return ((u32(sketch[cell_index(k.cell)].gas.w) >> 8u) & what) != 0u;
}

// The sand at a point that is not the bed's: the share of the wet part that
// a body's sand fills, with that share's steepness (a cell), the black's
// share of the body's sand and how far its grains blur; the other sand's
// fine and coarse volumes, of the wet part, with the free powder's share of
// the fine and the velocity (m/s).
struct Sand {
    packed: f32, steepness: f32, black: f32, blur: f32,
    other: vec2<f32>, powder: f32, drift: vec2<f32>,
}
fn sand_at(p: vec2<f32>) -> Sand {
    let k = stencil(p);
    if !near(k, SAND_NEAR) { return Sand(0., 0., 0., 0., vec2<f32>(0.), 0., vec2<f32>(0.)); }
    var share = 0.;
    var share_slope = vec2<f32>(0.);
    var gas = 0.;
    var gas_slope = vec2<f32>(0.);
    var black = 0.;
    var slid = 0.;
    var other = vec2<f32>(0.);
    var powder = 0.;
    var flux = vec2<f32>(0.);
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            let s = sketch_at(k, i, j);
            let w = k.wx[i] * k.wy[j];
            let d = vec2<f32>(k.dx[i] * k.wy[j], k.wx[i] * k.dy[j]);
            share += w * s.sand.x;
            share_slope += d * s.sand.x;
            gas += w * min(s.gas.x, 1.);
            gas_slope += d * min(s.gas.x, 1.);
            black += w * s.sand.x * s.sand.y;
            slid += w * s.sand.x * s.carry.w;
            other += w * s.sand.zw;
            powder += w * s.sand.z * s.carry.z;
            flux += w * (s.sand.z + s.sand.w) * s.carry.xy;
        }
    }
    // The sand lies in the wet part of the cells. Where gas takes a part of
    // them, a body that fills the rest ends at the bubble's outline, not
    // short of it.
    let wet = max(1. - gas, .05);
    let packed = share / wet;
    return Sand(packed, length(share_slope + packed * gas_slope) / wet,
        black / max(share, 1.e-6), slid / max(share, 1.e-6),
        other / wet, powder / max(other.x, 1.e-9), flux / max(other.x + other.y, 1.e-9));
}

// The exposure of a frame: one refresh of a 60 Hz display.
const EXPOSURE: f32 = 1. / 60.;
// A number in [0, 1) fixed to a grain site.
fn site_hash(q: vec2<i32>) -> f32 {
    var h = (u32(q.x) * 73856093u) ^ (u32(q.y) * 19349663u);
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967296.;
}
// The most points a side at which a pixel's grains are sampled, and the
// most along a grain's path in a frame.
const GRAIN_SAMPLES: i32 = 8;
const PATH_SAMPLES: i32 = 4;
// The seconds a pattern of falling grains is carried with them before it is
// laid anew: long enough for a grain to be seen to fall, short enough for
// the grains' velocity to have stayed what it is now. A longer carry would
// multiply every flicker of the velocity into a jump of the pattern.
const CARRY: f32 = .5;
// The time (s) of the state the picture shows: the bed's record keeps it
// (bed/strata.wgsl), from the start or the last turn.
fn clock() -> f32 { return strata[0].y; }
// The grains a pixel shows at p, as (the share of the pixel they cover, the
// black's share of them), where a grain shows at a site with the probability
// `cover` and is black with the probability `black`, and the grains move at
// `velocity` (m/s).
//
// Sand against the glass is seen grain by grain: each site a grain across
// shows one grain or the water behind. A pixel shows the grains under it: it
// is sampled at points no further apart than a site, as many a side as it is
// sites across, rounded up (a pixel no wider than a grain shows the grain it
// lies in).
//
// Grains that move are seen to move: the sites are carried with the grains'
// velocity. A carried pattern shears with the flow, so two patterns half a
// period apart take turns, each fading in after it is laid anew and out
// before the next: every point of a pixel shows one of the two, the first
// for the share of the points that is its turn's weight. In a frame's
// exposure a grain is smeared along its path, and only along it: across
// their path the grains of a falling stream still show one by one.
fn grains_at(p: vec2<f32>, cover: f32, black: f32, velocity: vec2<f32>) -> vec2<f32> {
    let site = cfg.diameters.x / cfg.cell_width;
    let pixel = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let side = clamp(i32(ceil(pixel.x / site)), 1, GRAIN_SAMPLES);
    let drift = velocity / cfg.cell_width;
    let speed = length(drift);
    let along = clamp(i32(ceil(speed * EXPOSURE / site)), 1, PATH_SAMPLES);
    let points = along * side * side;
    // The first pattern's turn and weight; the second is half a period on.
    // Grains carried under a quarter of a site in a period keep the first.
    let turn = fract(clock() / CARRY);
    let first = select(1., 1. - abs(2. * turn - 1.), speed * CARRY > .25 * site);
    // The points that show the first pattern differ from pixel to pixel
    // (interleaved gradient noise), so that no two pixels change together.
    let dither = fract(52.9829189 * fract(dot(floor((p + .5) / pixel), vec2<f32>(.06711056, .00583715))));
    var shown = 0.;
    var dark = 0.;
    for (var k = 0; k < points; k++) {
        let s = k % (side * side);
        let o = ((vec2<f32>(f32(s % side), f32(s / side)) + .5) / f32(side) - .5) * pixel;
        let lag = select(fract(turn + .5), turn, fract(dither + (f32(k) + .5) / f32(points)) < first) * CARRY
            + EXPOSURE * ((f32(k / (side * side)) + .5) / f32(along) - .5);
        let q = vec2<i32>(floor((p + o - drift * lag) / site));
        var grain = 1.;
        if cover < 1. { grain = select(0., 1., site_hash(q + vec2<i32>(7919, 104729)) < cover); }
        shown += grain;
        dark += grain * select(0., 1., site_hash(q) < black);
    }
    return vec2<f32>(shown / f32(points), dark / max(shown, 1.e-6));
}
// The white sand and the black (chromite) in the colours a real picture
// shows under blue light, for grains of which the share `black` is black.
fn sand_color(black: f32) -> vec3<f32> {
    return mix(vec3<f32>(.50, .74, .98), vec3<f32>(.03, .05, .09), black);
}
// The depth of grains a sight line crosses in the gap, in the grains' own
// mean free paths. The grains of a volume (of the wet volume) are spheres of
// the diameter d: n = volume / (pi d^3 / 6) of them a volume, each pi d^2 / 4
// across, so a line across the gap b meets n b pi d^2 / 4 = 1.5 b volume / d
// on average, and none with exp(-depth). (The light a grain takes from a
// beam is twice that, by diffraction: that is how far one sees through the
// sand, not how many grains one sees.)
fn depth(volume: f32, diameter: f32) -> f32 {
    return 1.5 * cfg.gap * max(volume, 0.) / diameter;
}
// The share of the light on a white grain that the grain turns back. The
// white is silica and glass: grains that let the light through and bend it
// little in water. By the ray optics of a sphere of quartz or glass (index
// 1.52 to 1.55 against the water's 1.333) the light a grain scatters keeps
// the mean cosine g = 0.9 of its direction; (1 - g) / 2 is what a grain
// turns of a stream of light going in into the stream coming out.
const TURNED_BACK: f32 = .05;
// How white the white sand of the depth `grains` is, as a share of the
// bed's white. Sand is white where the light meets many grains: a layer
// that absorbs nothing, before a dark backing, sends back k t / (1 + k t)
// of the light on it (k = TURNED_BACK, t its depth). The sand's colour is
// the bed's, a gap full of packed white, and thinner sand has its share of
// it: a sixth of the bed's white for each grain a sight line meets, where
// the sand is thin. So what falls is about as white, summed over the
// picture, as the heap it makes.
fn glow(grains: f32) -> f32 {
    let bed = depth(BED_PACKING, cfg.diameters.x);
    return min(grains * (1. + TURNED_BACK * bed) / (bed * (1. + TURNED_BACK * grains)), 1.);
}
// The wet part of the picture at a point. Three things lie over the
// backdrop, the later hiding the earlier: the sand that falls or lies thin,
// in single grains; the bodies of sand at rest, to where they fill half; and
// the bed, from its record. A pixel shows one pattern of grains for the
// three: where two meet within it, their shares of the pixel weigh the
// pattern's chances.
//
// The bodies and the bed fill the gap, and every site of theirs shows a
// grain. Thin sand shows a black grain at a site with the chance that a
// sight line meets one. Its white grains are glassy, and the sand is less
// white than the grains a sight line meets would cover (glow): the picture
// shows a white grain at a site with the chance that gives the sand its
// white, a grain in six where the sand is thin and every grain where it
// fills the gap. (Every grain at a sixth of the white is the same in the
// mean, and a haze in which no grain shows.)
//
// Only the thin sand's grains are carried. A body's grains stay where the
// glass shows them: its sand creeps, and where it slides a grain's width in
// a frame its grains blur into their mean.
fn wet_at(p: vec2<f32>) -> vec3<f32> {
    let bed = bed_at(p);
    let laid = vec2<f32>(1. - bed.share, bed.share) / cfg.diameters;
    // The chance of a grain at a site, the chance of a black one, the
    // carried sand's velocity weighed by its chance of a grain, the chance
    // of a grain that slides blurred, and the free powder's white.
    var shown = bed.inside;
    var black = bed.inside * laid.y / max(laid.x + laid.y, 1.e-9);
    var flux = vec2<f32>(0.);
    var blurred = 0.;
    var dust = 0.;
    if bed.inside < 1. {
        let sand = sand_at(p);
        let pixel = f32(cfg.size.x) / f32(cfg.image_size.x);
        let body = (1. - bed.inside) * clamp(.5 + (sand.packed - .5) / max(sand.steepness * pixel, 1.e-4), 0., 1.);
        let open = 1. - bed.inside - body;
        // The grit's depths, white and black, and the chance that a sight
        // line meets a grain of it. The white's free powder is too fine to
        // show grain by grain: it whitens the water behind the grains.
        let grit = vec2<f32>(depth(sand.other.x * (1. - sand.powder), cfg.diameters.x), depth(sand.other.y, cfg.diameters.y));
        let met = (1. - exp(-grit.x - grit.y)) / max(grit.x + grit.y, 1.e-9);
        let dark = open * met * grit.y;
        let thin = dark + open * min(glow(grit.x), met * grit.x);
        shown += body + thin;
        black += body * sand.black + dark;
        flux = thin * sand.drift;
        blurred = body * sand.blur;
        dust = open * glow(depth(sand.other.x * sand.powder, cfg.powder_diameter));
    }
    var color = mix(backdrop(p), sand_color(0.), dust);
    if shown > 0. {
        let seen = grains_at(p, shown, black / shown, flux / shown);
        color = mix(color, sand_color(mix(seen.y, black / shown, blurred / shown)), seen.x);
    }
    return color;
}

// A bubble at a point: the distance (cells) to its outline, negative inside,
// and the outline's outward normal.
struct Bubble { distance: f32, normal: vec2<f32> }
// The furthest a bubble's outline lies outside the half level of all the
// gas (cells): where a sharp outline's gas is a third, the least level at
// which a bubble ends.
const SWELL: f32 = .25;
// Each identity's gas is a smooth field. A bubble alone is drawn to the half
// level of the gas. Among others, a point's bubble is the largest identity
// there, and where its outline lies follows from its lead over the second,
// the difference of the two over their sum:
//
// - With no lead the two hold as much: there the film runs, and both end.
// - Toward a film the outline keeps to fuller cells, at the film itself to
//   cells full of gas. So where a film meets the outline of the gas, each
//   bubble turns into the film on a round, and between the two rounds the
//   water or the sand shows, as between three bubbles: the foam's Plateau
//   borders. Surface tension gives the bubbles of a raft these round
//   shoulders (a border under a raft a few millimetres deep is about a
//   millimetre wide, the capillary length squared over the depth); the
//   cells' gas alone would draw a band cut by straight films.
// - Between its films the outline swells past the half level by as much as
//   the rounds take: the level's mean over the lead is the half, so the
//   bubbles of the picture have the gas of the model.
//
// The traces of further identities that the transport leaves in a cell are
// no bubbles and part nothing.
fn bubble_at(p: vec2<f32>) -> Bubble {
    let k = stencil(p);
    if !near(k, GAS_NEAR) { return Bubble(4., vec2<f32>(0., 1.)); }
    let q = p + vec2<f32>(.5);
    var best_sd = 4.;
    var best_n = vec2<f32>(0., 1.);
    for (var bi = 0u; bi < MAX_BUBBLES; bi++) {
        if foam[bi].alive == 0u { continue; }
        let bmin = foam[bi].bbox_min - vec2<f32>(1.5);
        let bmax = foam[bi].bbox_max + vec2<f32>(1.5);
        if q.x < bmin.x || q.x > bmax.x || q.y < bmin.y || q.y > bmax.y {
            continue;
        }
        var min_d2 = 1e20;
        var edge_n = vec2<f32>(0., 1.);
        var crossings = 0i;
        for (var m = 0u; m < FOAM_VERTS; m++) {
            let a = foam[bi].verts[m];
            let b = foam[bi].verts[(m + 1u) % FOAM_VERTS];
            let e = b - a;
            let len2 = dot(e, e);
            let t = select(0., clamp(dot(q - a, e) / len2, 0., 1.), len2 > 1e-10);
            let closest = a + t * e;
            let diff = q - closest;
            let d2 = dot(diff, diff);
            if d2 < min_d2 {
                min_d2 = d2;
                let e_len = sqrt(max(len2, 1e-10));
                let en = vec2<f32>(e.y, -e.x) / e_len;
                // Smooth vertex normal interpolation across edge ends for silky rim shading.
                let a_prev = foam[bi].verts[(m + FOAM_VERTS - 1u) % FOAM_VERTS];
                let b_next = foam[bi].verts[(m + 2u) % FOAM_VERTS];
                let t0 = b - a_prev;
                let t1 = b_next - a;
                let n0 = vec2<f32>(t0.y, -t0.x) / max(length(t0), 1e-6);
                let n1 = vec2<f32>(t1.y, -t1.x) / max(length(t1), 1e-6);
                let interp = mix(n0, n1, t);
                let ilen = length(interp);
                edge_n = select(en, interp / ilen, ilen > 1e-5);
            }
            if (a.y <= q.y && b.y > q.y) || (b.y <= q.y && a.y > q.y) {
                let x_int = a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if q.x < x_int { crossings += 1; }
            }
        }
        let inside = (crossings & 1) != 0;
        let dist = sqrt(min_d2);
        let sd = select(dist, -dist, inside);
        if sd < best_sd {
            best_sd = sd;
            best_n = edge_n;
        }
    }
    return Bubble(clamp(best_sd, -4., 4.), best_n);
}

fn bubble_color(p: vec2<f32>, bubble: Bubble) -> vec3<f32> {
    // Circular cross-section of a wetting meniscus across the 1 mm gap.
    // Within half a gap from the outline it joins a flat front surface.
    let radius = .5 * cfg.gap / cfg.cell_width;
    let radial = clamp(1. + bubble.distance / radius, 0., 1.);
    let normal = vec3<f32>(bubble.normal * radial, sqrt(max(0., 1. - radial * radial)));
    let incident = vec3<f32>(0., 0., -1.);
    let reflection = reflect(incident, normal);
    let f = fresnel(normal.z, WATER_IOR / AIR_IOR);
    var reflectance = 1.;
    var transmission = backdrop(p);
    if f < 1. {
        let ray = refract(incident, normal, WATER_IOR / AIR_IOR);
        // Intersect the back of the same circular meniscus, or its flat
        // centre. A parallel thin sheet cancels the angular deflection and
        // hides the background distortion precisely where the curved rim
        // should show it.
        let start = vec2<f32>(radius + bubble.distance, radius * normal.z);
        let direction = vec2<f32>(dot(ray.xy, bubble.normal), ray.z);
        var travel = (-radius - start.y) / ray.z;
        var back_normal = vec3<f32>(0., 0., -1.);
        if start.x + travel * direction.x > 0. {
            let b = dot(start, direction);
            travel = -b + sqrt(max(0., b * b - dot(start, start) + radius * radius));
            let end = (start + travel * direction) / radius;
            back_normal = normalize(vec3<f32>(bubble.normal * end.x, end.y));
        }
        let exit_ray = refract(ray, -back_normal, AIR_IOR / WATER_IOR);
        let back_f = fresnel(clamp(dot(ray, back_normal), 0., 1.), AIR_IOR / WATER_IOR);
        // Sum the scalar noninterfering Fresnel series. Secondary paths use
        // the first reflection/exit direction; no caustic or interference
        // rendering.
        reflectance = 1. - (1. - f) * (1. - back_f) / (1. - f * back_f);
        transmission = environment(p + ray.xy * travel, exit_ray);
    }
    // The rim reflects the studio's light all the way round. (Grains that
    // stand against a meniscus would show in its reflection in place of the
    // light, and a bubble under white sand would lose its outline there.
    // The picture keeps the outline: it is what shows a bubble.)
    let surface_z = radius * normal.z;
    let plate_z = select(-radius, radius, reflection.z >= 0.);
    let path = abs(plate_z - surface_z) / max(.05, abs(reflection.z));
    let reflected = environment(p + reflection.xy * path, reflection);
    return mix(transmission, reflected, reflectance);
}

@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2. - 1., 0., 1.);
}
fn scene_color(p: vec2<f32>, pixel_width: f32, bubble: Bubble) -> vec3<f32> {
    let coverage = 1. - smoothstep(-.5 * pixel_width, .5 * pixel_width, bubble.distance);
    var color = vec3<f32>(0.);
    if coverage < 1. { color = wet_at(p); }
    if coverage > 0. { color = mix(color, bubble_color(p, bubble), coverage); }
    return color;
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let p = vec2<f32>(position.x * scale.x, f32(cfg.size.y) - position.y * scale.y) - .5;
    let bubble = bubble_at(p);
    var color = vec3<f32>(0.);
    let radius = .5 * cfg.gap / cfg.cell_width;
    // The outline and the narrow band of total reflection inside it, the
    // outer quarter of the meniscus, are integrated over the pixel at four
    // points a side; the locally planar distance has the same interpolated
    // contour and normal. Elsewhere the picture changes slowly across a
    // pixel, and its middle stands for it.
    let side = select(1u, 4u, bubble.distance > -.5 * radius - scale.x && bubble.distance < scale.x);
    /*loop: copies of the whole picture's shading*/for (var y = 0u; y < side; y++) {
        /*loop: copies of the whole picture's shading*/for (var x = 0u; x < side; x++) {
            let offset = ((vec2<f32>(f32(x), f32(y)) + .5) / f32(side) - .5) * scale;
            let sample = Bubble(bubble.distance + dot(bubble.normal, offset), bubble.normal);
            color += scene_color(p + offset, scale.x / f32(side), sample) / f32(side * side);
        }
    }
    // The front pane sees the same finite studio lights as the bubble rims.
    let glass = fresnel(1., 1. / 1.52);
    color = mix(color, environment(p, vec3<f32>(0., 0., 1.)), glass);
    return vec4<f32>(color, 1.);
}
