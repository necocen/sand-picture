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

// The sand at a point that is not the bed's: the share packed sand at rest
// fills with the steepness of that share (a cell) and its black share; the
// other sand's fine and coarse volumes with the free powder's share of the
// fine; the wet volume; the grains' velocity (m/s).
struct Sand { packed: f32, steepness: f32, black: f32, veil: vec2<f32>, powder: f32, wet: f32, velocity: vec2<f32> }
fn sand_at(p: vec2<f32>) -> Sand {
    let k = stencil(p);
    if !near(k, SAND_NEAR) { return Sand(0., 0., 0., vec2<f32>(0.), 0., 1., vec2<f32>(0.)); }
    var packed = 0.;
    var gradient = vec2<f32>(0.);
    var black = 0.;
    var veil = vec2<f32>(0.);
    var powder = 0.;
    var wet = 0.;
    var flux = vec2<f32>(0.);
    var grains = 0.;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            let s = sketch_at(k, i, j);
            let w = k.wx[i] * k.wy[j];
            packed += w * s.sand.x;
            gradient += vec2<f32>(k.dx[i] * k.wy[j], k.wx[i] * k.dy[j]) * s.sand.x;
            black += w * s.sand.x * s.grains.w;
            veil += w * s.sand.yz;
            powder += w * s.sand.y * s.motion.z;
            wet += w * (1. - min(s.gas.x, 1.));
            let amount = w * (REST_PACKING * s.sand.x + s.sand.y + s.sand.z);
            flux += amount * s.motion.xy;
            grains += amount;
        }
    }
    return Sand(packed, length(gradient), black / max(packed, 1.e-6), veil, powder / max(veil.x, 1.e-9), wet,
        flux / max(grains, 1.e-9));
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
// The most points a side at which a pixel's grains are sampled.
const GRAIN_SAMPLES: i32 = 8;
// The share of the pixel at p that black grains cover, where grains of the
// black share `black` move at `speed` (m/s). Sand against the glass is seen
// grain by grain: each site a grain across shows one grain, black with the
// black's share of the grains' cross-sections. A pixel shows the grains
// under it: it is sampled at points no further apart than a site, as many a
// side as it is sites across, rounded up (a pixel no wider than a grain
// shows the grain it lies in). Grains that move more than their size in a
// frame are smeared into the mean share.
fn black_cover(p: vec2<f32>, black: f32, speed: f32) -> f32 {
    let site = cfg.diameters.x / cfg.cell_width;
    let pixel = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let side = clamp(i32(ceil(pixel.x / site)), 1, GRAIN_SAMPLES);
    var grains = 0.;
    for (var k = 0; k < side * side; k++) {
        let o = ((vec2<f32>(f32(k % side), f32(k / side)) + .5) / f32(side) - .5) * pixel;
        grains += select(0., 1., site_hash(vec2<i32>(floor((p + o) / site))) < black);
    }
    grains /= f32(side * side);
    return mix(grains, black, clamp(speed * EXPOSURE / cfg.diameters.x, 0., 1.));
}
// The white sand and the black (chromite) in the colours a real picture
// shows under blue light.
fn grains_color(p: vec2<f32>, black: f32, speed: f32) -> vec3<f32> {
    return mix(vec3<f32>(.50, .74, .98), vec3<f32>(.03, .05, .09), black_cover(p, black, speed));
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
// Over the backdrop the sand that moves or lies thin, as a veil whose depth
// follows its grains' cross-sections across the gap; over it the bodies of
// sand at rest, to where they fill half; over all the bed, from its record.
fn wet_at(p: vec2<f32>) -> Wet {
    let bed = bed_at(p);
    var color = backdrop(p);
    var grains = 0.;
    if bed.inside < 1. {
        let sand = sand_at(p);
        let speed = length(sand.velocity);
        let veil = sections(sand.veil, sand.powder);
        let depth = 3. * cfg.gap * (veil.x + veil.y) / max(sand.wet, .02);
        if depth > 0. {
            color = mix(color, grains_color(p, veil.y / (veil.x + veil.y), speed), 1. - exp(-depth));
        }
        let pixel = f32(cfg.size.x) / f32(cfg.image_size.x);
        let cover = clamp(.5 + (sand.packed - .5) / max(sand.steepness * pixel, 1.e-4), 0., 1.);
        if cover > 0. { color = mix(color, grains_color(p, sand.black, speed), cover); }
        grains = sand.veil.x + sand.veil.y + REST_PACKING * cover;
    }
    if bed.inside > 0. {
        let laid = sections(vec2<f32>(1. - bed.share, bed.share), 0.);
        color = mix(color, grains_color(p, laid.y / (laid.x + laid.y), 0.), bed.inside);
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
    if coverage <= 0. { return wet_at(p).color; }
    let air = bubble_color(p, bubble);
    if coverage >= 1. { return air; }
    return mix(wet_at(p).color, air, coverage);
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let p = vec2<f32>(position.x * scale.x, f32(cfg.size.y) - position.y * scale.y) - .5;
    let bubble = bubble_at(p);
    var color = vec3<f32>(0.);
    let radius = .5 * cfg.gap / cfg.cell_width;
    if bubble.distance > -.5 * radius - scale.x && bubble.distance < scale.x {
        // Integrate the outline and the narrow band of total reflection
        // inside it, the outer quarter of the meniscus, over the pixel. The
        // locally planar distance has the same interpolated contour/normal.
        // Further in, the meniscus changes slowly across a pixel.
        /*loop: sixteen copies of the whole picture's shading*/for (var y = 0u; y < 4u; y++) {
            /*loop: sixteen copies of the whole picture's shading*/for (var x = 0u; x < 4u; x++) {
                let offset = ((vec2<f32>(f32(x), f32(y)) + .5) * .25 - .5) * scale;
                let sample = Bubble(bubble.distance + dot(bubble.normal, offset), bubble.normal);
                color += .0625 * scene_color(p + offset, .25 * scale.x, sample);
            }
        }
    } else {
        color = scene_color(p, scale.x, bubble);
    }
    // The front pane sees the same finite studio lights as the bubble rims.
    let glass = fresnel(1., 1. / 1.52);
    color = mix(color, environment(p, vec3<f32>(0., 0., 1.)), glass);
    return vec4<f32>(color, 1.);
}
