// The Photoreal picture, drawn per pixel from the sketch of the state
// (sketch.wgsl): the wet sand over the backdrop (wet_at) and over it the
// gas, each bubble a thin meniscus that reflects and refracts the studio
// (bubble_color). Display only: every simulation buffer is read-only.
//
// Gas and sand are drawn as smooth fields: a quadratic B-spline through the
// cells' values, which has no corner at a cell's edge and no feature finer
// than a cell. A cell is about a gap wide, and neither a bubble's meniscus
// nor a body of sand between the plates shows a finer in-plane feature. A
// field's half level is the outline of what fills the cells: where a
// straight surface cuts a cell, the level lies within a tenth of a cell of
// it.
struct Sketch { gas: vec4<f32>, grains: vec4<f32>, motion: vec4<f32>, sand: vec4<f32> }
@group(0) @binding(2) var<storage, read> sketch: array<Sketch>;

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

// The packing of sand at rest, the deposit's (material.rs).
const REST_PACKING: f32 = .585;
// What a cell's sketch says of the cells around it (sketch.wgsl).
const GAS_NEAR: u32 = 1u;
const SAND_NEAR: u32 = 2u;
fn near(k: Stencil, what: u32) -> bool {
    return (u32(sketch[cell_index(k.cell)].motion.w) & what) != 0u;
}

// The sand at a point that is not the bed's: the share a body's sand fills
// with the steepness of that share (a cell), its black share and its speed
// (m/s); the other sand's fine and coarse volumes with the free powder's
// share of the fine and its velocity (m/s); the wet volume.
struct Sand {
    packed: f32, steepness: f32, black: f32, speed: f32,
    veil: vec2<f32>, powder: f32, drift: vec2<f32>,
    wet: f32,
}
fn sand_at(p: vec2<f32>) -> Sand {
    let k = stencil(p);
    if !near(k, SAND_NEAR) { return Sand(0., 0., 0., 0., vec2<f32>(0.), 0., vec2<f32>(0.), 1.); }
    var packed = 0.;
    var gradient = vec2<f32>(0.);
    var black = 0.;
    var slide = 0.;
    var veil = vec2<f32>(0.);
    var powder = 0.;
    var wet = 0.;
    var flux = vec2<f32>(0.);
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            let s = sketch_at(k, i, j);
            let w = k.wx[i] * k.wy[j];
            packed += w * s.sand.x;
            gradient += vec2<f32>(k.dx[i] * k.wy[j], k.wx[i] * k.dy[j]) * s.sand.x;
            black += w * s.sand.x * s.grains.w;
            veil += w * s.sand.yz;
            slide += w * s.sand.x * length(s.motion.xy);
            powder += w * s.sand.y * s.motion.z;
            wet += w * (1. - min(s.gas.x, 1.));
            flux += w * (s.sand.y + s.sand.z) * s.motion.xy;
        }
    }
    return Sand(packed, length(gradient), black / max(packed, 1.e-6), slide / max(packed, 1.e-6),
        veil, powder / max(veil.x, 1.e-9), flux / max(veil.x + veil.y, 1.e-9), wet);
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
// The cross-sections (per volume, 1/m) of fine and coarse volumes of which
// the fine has the free powder's share `powder`: the powder, a fifth of the
// grains' size, whitens the water five times more than as much white.
fn sections(grains: vec2<f32>, powder: f32) -> vec2<f32> {
    return max(grains, vec2<f32>(0.))
        * vec2<f32>((1. - powder) / cfg.diameters.x + powder / cfg.powder_diameter, 1. / cfg.diameters.y);
}
// The wet part of the picture at a point: its colour, and how fast its
// grains dim a ray through the water there (the optical depth a metre).
struct Wet { color: vec3<f32>, turbidity: f32 }
// Three things lie over the backdrop, the later hiding the earlier: the sand
// that falls or lies thin, a veil whose grains show with the probability
// that follows their cross-sections across the gap; the bodies of sand at
// rest, to where they fill half; and the bed, from its record. A pixel shows
// one pattern of grains for the three: where two meet within it, their
// shares of the pixel weigh the pattern's chances.
//
// Only the veil's grains are carried. A body's grains stay where the glass
// shows them: its sand creeps, and where it slides a grain's width in a
// frame its grains blur into their mean.
fn wet_at(p: vec2<f32>) -> Wet {
    let bed = bed_at(p);
    let laid = sections(vec2<f32>(1. - bed.share, bed.share), 0.);
    // The chance of a grain at a site, the chance of a black one, the
    // veil's velocity weighed by its chance of a grain, and the chance of a
    // grain that slides blurred.
    var shown = bed.inside;
    var black = bed.inside * laid.y / max(laid.x + laid.y, 1.e-9);
    var flux = vec2<f32>(0.);
    var blurred = 0.;
    var grains = 0.;
    if bed.inside < 1. {
        let sand = sand_at(p);
        let pixel = f32(cfg.size.x) / f32(cfg.image_size.x);
        let body = (1. - bed.inside) * clamp(.5 + (sand.packed - .5) / max(sand.steepness * pixel, 1.e-4), 0., 1.);
        let veil = sections(sand.veil, sand.powder);
        let depth = 3. * cfg.gap * (veil.x + veil.y) / max(sand.wet, .02);
        let thin = (1. - bed.inside - body) * (1. - exp(-depth));
        shown += body + thin;
        black += body * sand.black + thin * veil.y / max(veil.x + veil.y, 1.e-9);
        flux = thin * sand.drift;
        blurred = body * clamp(sand.speed * EXPOSURE / cfg.diameters.x, 0., 1.);
        grains = sand.veil.x + sand.veil.y + REST_PACKING * body;
    }
    var color = backdrop(p);
    if shown > 0. {
        let seen = grains_at(p, shown, black / shown, flux / shown);
        color = mix(color, sand_color(mix(seen.y, black / shown, blurred / shown)), seen.x);
    }
    return Wet(color, 3. * mix(grains, BED_PACKING, bed.inside) / cfg.diameters.x);
}

// A bubble at a point: the distance (cells) to its outline, negative inside,
// and the outline's outward normal.
struct Bubble { distance: f32, normal: vec2<f32> }
// The gas is the sum of the cells' gas as a smooth field, its half level the
// outline of all the gas. Inside, the identities part it into bubbles: each
// identity's gas is a smooth field of its own, the largest at a point is the
// point's bubble, and where the two largest are equal a film stands between
// two bubbles. The identities' traces in a cell, which the sketch leaves
// out, part nothing.
fn bubble_at(p: vec2<f32>) -> Bubble {
    let k = stencil(p);
    if !near(k, GAS_NEAR) { return Bubble(4., vec2<f32>(0., 1.)); }
    var value: array<f32, GAS_SLOTS>;
    var slope: array<vec2<f32>, GAS_SLOTS>;
    var total = 0.;
    var gradient = vec2<f32>(0.);
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            let gas = sketch_at(k, i, j).gas;
            if gas.x > 0. {
                let w = k.wx[i] * k.wy[j];
                let d = vec2<f32>(k.dx[i] * k.wy[j], k.wx[i] * k.dy[j]);
                total += w * gas.x;
                gradient += d * gas.x;
                let a = u32(gas.w) % 16u;
                let b = u32(gas.w) / 16u;
                value[a] += w * gas.y;
                slope[a] += d * gas.y;
                value[b] += w * gas.z;
                slope[b] += d * gas.z;
            }
        }
    }
    let steepness = max(length(gradient), 1.e-3);
    var bubble = Bubble(clamp((.5 - total) / steepness, -4., 4.), -gradient / steepness);
    var first = 0u;
    var second = 1u;
    if value[1] > value[0] {
        first = 1u;
        second = 0u;
    }
    for (var l = 2u; l < GAS_SLOTS; l++) {
        if value[l] > value[first] {
            second = first;
            first = l;
        } else if value[l] > value[second] {
            second = l;
        }
    }
    // A film shows where two identities hold the gas between them: where the
    // second is a trace, or many share the gas, no film is resolved, and its
    // meniscus is kept a gap away.
    let resolved = smoothstep(.5, .8, (value[first] + value[second]) / total)
        * smoothstep(.1, .25, value[second] / total);
    let apart = slope[first] - slope[second];
    let steep = max(length(apart), 1.e-3);
    let film = (value[second] - value[first]) / steep - (1. - resolved) * cfg.gap / cfg.cell_width;
    if film > bubble.distance { bubble = Bubble(film, -apart / steep); }
    return bubble;
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
    // A reflected ray sees the grains around it in the same shallow water
    // slab: those half-way along its path to the plate dim it and show.
    let surface_z = radius * normal.z;
    let plate_z = select(-radius, radius, reflection.z >= 0.);
    let path = abs(plate_z - surface_z) / max(.05, abs(reflection.z));
    var reflected = environment(p + reflection.xy * path, reflection);
    // Where little is reflected the grains change less than can be seen.
    if path > 0. && reflectance > .25 {
        let around = wet_at(p + .5 * reflection.xy * path);
        reflected = mix(reflected, around.color, 1. - exp(-path * cfg.cell_width * around.turbidity));
    }
    return mix(transmission, reflected, reflectance);
}

@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2. - 1., 0., 1.);
}
fn scene_color(p: vec2<f32>, pixel_width: f32, bubble: Bubble) -> vec3<f32> {
    let coverage = 1. - smoothstep(-.5 * pixel_width, .5 * pixel_width, bubble.distance);
    var color = vec3<f32>(0.);
    if coverage < 1. { color = wet_at(p).color; }
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
