// The Photoreal picture, drawn per pixel after the distance field and the
// layer mask: the wet sand, grain by grain over the backdrop (wet_color), and
// over it each bubble as a thin meniscus that reflects and refracts the
// studio (bubble_color). Display only: every simulation buffer is read-only.
@group(0) @binding(2) var distances: texture_2d_array<f32>;
@group(0) @binding(3) var optical_sampler: sampler;
@group(0) @binding(6) var<storage, read> layers: array<u32>;

fn sample_composition(p: vec2<f32>) -> vec4<f32> {
    let q = vec2<i32>(floor(p));
    let t = fract(p);
    return mix(mix(composition(q), composition(q + vec2<i32>(1, 0)), t.x),
        mix(composition(q + vec2<i32>(0, 1)), composition(q + vec2<i32>(1)), t.x), t.y);
}
fn sample_powder_share(p: vec2<f32>) -> f32 {
    let q = vec2<i32>(floor(p));
    let t = fract(p);
    return mix(mix(powder_share(q), powder_share(q + vec2<i32>(1, 0)), t.x),
        mix(powder_share(q + vec2<i32>(0, 1)), powder_share(q + vec2<i32>(1)), t.x), t.y);
}
// The inverse diameters (1/m) that set the two species' optical depth per
// unit volume. The white's grains and its free powder each scatter by their
// cross-section, so the powder, a fifth of the grains' size, whitens the
// water five times more than as much white.
fn inverse_diameters(p: vec2<f32>) -> vec2<f32> {
    let share = sample_powder_share(p);
    return vec2<f32>((1. - share) / cfg.diameters.x + share / cfg.powder_diameter, 1. / cfg.diameters.y);
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
// The speed (m/s) of the grains of the cell at p, the deposit's at rest.
fn grain_speed(p: vec2<f32>) -> f32 {
    let c = cells[cell_index(vec2<i32>(floor(p + .5)))];
    let solid = max(c.volume.fine, 0.) + max(c.volume.coarse, 0.) + max(c.bed.x, 0.) + max(c.bed.y, 0.);
    if solid <= 0. { return 0.; }
    return length(c.momentum[1] + c.momentum[2]) / solid;
}
// The most points a side at which a pixel's grains are sampled.
const GRAIN_SAMPLES: i32 = 8;
// The share of the pixel at p that black grains cover. Sand against the
// glass is seen grain by grain: each site a grain across shows one grain,
// black with the black's share of the projected area. A pixel shows the
// grains under it, so it is sampled at as many points a side as it is sites
// across (one at least: a pixel finer than a grain shows the grain it lies
// in). Grains that move more than their size in a frame are smeared into the
// mean share.
fn black_cover(p: vec2<f32>, black: f32) -> f32 {
    let site = cfg.diameters.x / cfg.cell_width;
    let pixel = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let side = clamp(i32(round(pixel.x / site)), 1, GRAIN_SAMPLES);
    var grains = 0.;
    for (var k = 0; k < side * side; k++) {
        let o = ((vec2<f32>(f32(k % side), f32(k / side)) + .5) / f32(side) - .5) * pixel;
        grains += select(0., 1., site_hash(vec2<i32>(floor((p + o) / site))) < black);
    }
    grains /= f32(side * side);
    let smear = clamp(grain_speed(p) * EXPOSURE / cfg.diameters.x, 0., 1.);
    return mix(grains, black, smear);
}
fn wet_color(p: vec2<f32>) -> vec3<f32> {
    let c = display_composition(p);
    let wet = max(1.e-6, c.x + c.y + c.z);
    let depths = 3. * cfg.gap * max(c.yz, vec2<f32>(0.)) * inverse_diameters(p) / wet;
    let depth = depths.x + depths.y;
    // The white sand and the black (chromite) in the colours a real picture
    // shows under blue light.
    let sand = mix(vec3<f32>(.50, .74, .98), vec3<f32>(.03, .05, .09), black_cover(p, depths.y / max(1.e-6, depth)));
    return mix(backdrop(p), sand, 1. - exp(-depth));
}

struct Bubble { distance: f32, normal: vec2<f32> }
fn bubble_at(p: vec2<f32>) -> Bubble {
    // A bubble's outline in a gap b shows no in-plane feature finer than its
    // meniscus: each label's distance is averaged over a 3x3 stencil one gap
    // across, which rounds the cells' interface segments and drops gas
    // slivers thinner than the gap (display only). Only the layers near the
    // pixel's cell are sampled (layers.wgsl), in the same ascending order.
    let step = cfg.gap / cfg.cell_width;
    var bubble = Bubble(4., vec2<f32>(0., 1.));
    var near = layers[cell_index(vec2<i32>(floor(p + .5)))];
    loop {
        if near == 0u { break; }
        let l = countTrailingZeros(near);
        near &= near - 1u;
        var field = vec3<f32>(0.);
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let w = f32((2 - abs(x)) * (2 - abs(y))) / 16.;
                let uv = (p + .5 + step * vec2<f32>(f32(x), f32(y))) / vec2<f32>(cfg.size);
                field += w * textureSampleLevel(distances, optical_sampler, uv, i32(l), 0.).xyz;
            }
        }
        if field.x < bubble.distance {
            bubble = Bubble(field.x, field.yz);
        }
    }
    bubble.normal /= max(1.e-6, length(bubble.normal));
    return bubble;
}

fn bubble_color(p: vec2<f32>, bubble: Bubble) -> vec3<f32> {
    // Circular cross-section of a wetting meniscus across the 1 mm gap.
    // Within half a gap from the VOF contour it joins a flat front surface.
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
    // A reflected ray sees surrounding grains in the same shallow water slab.
    let surface_z = radius * normal.z;
    let plate_z = select(-radius, radius, reflection.z >= 0.);
    let path = abs(plate_z - surface_z) / max(.05, abs(reflection.z));
    let reflected_p = p + reflection.xy * path;
    let material = sample_composition(p + .5 * reflection.xy * path);
    let occlusion = 1. - exp(-3. * path * cfg.cell_width
        * dot(max(material.yz, vec2<f32>(0.)), inverse_diameters(p + .5 * reflection.xy * path)));
    let reflected = mix(environment(reflected_p, reflection), wet_color(reflected_p), occlusion);
    return mix(transmission, reflected, reflectance);
}

@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2. - 1., 0., 1.);
}
fn scene_color(p: vec2<f32>, pixel_width: f32, bubble: Bubble) -> vec3<f32> {
    let coverage = 1. - smoothstep(-.5 * pixel_width, .5 * pixel_width, bubble.distance);
    var color = wet_color(p);
    if coverage > 0. { color = mix(color, bubble_color(p, bubble), coverage); }
    return color;
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    let p = vec2<f32>(position.x * scale.x, f32(cfg.size.y) - position.y * scale.y) - .5;
    let bubble = bubble_at(p);
    var color = scene_color(p, scale.x, bubble);
    let radius = .5 * cfg.gap / cfg.cell_width;
    if bubble.distance > -radius - scale.x && bubble.distance < scale.x {
        // Integrate narrow Fresnel/refraction highlights from the surrounding
        // light. The locally planar distance has the same interpolated
        // contour/normal.
        color = vec3<f32>(0.);
        for (var y = 0u; y < 4u; y++) {
            for (var x = 0u; x < 4u; x++) {
                let offset = ((vec2<f32>(f32(x), f32(y)) + .5) * .25 - .5) * scale;
                let sample = Bubble(bubble.distance + dot(bubble.normal, offset), bubble.normal);
                color += .0625 * scene_color(p + offset, .25 * scale.x, sample);
            }
        }
    }
    // The front pane sees the same finite studio lights as the bubble rims.
    let glass = fresnel(1., 1. / 1.52);
    color = mix(color, environment(p, vec3<f32>(0., 0., 1.)), glass);
    return vec4<f32>(color, 1.);
}
