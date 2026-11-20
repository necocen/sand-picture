// Flow view: the cell-mean velocity of all that moves (the sum of the phase
// momenta α u of water, both grain kinds and gas over the sum of their
// volumes) as an optical-flow colour wheel. The hue is the direction (right
// red, up chartreuse, left cyan, down violet); the brightness is the speed.
// Seen brightness goes as the square root of the speed, up to FLOW_FULL: a
// stream at 16 mm/s is four times as bright as the water rising between the
// streams at 1 mm/s, and twice as bright as a plume at 4 mm/s. (A scale by
// the logarithm, which also showed the slab's creep, made everything from
// 1 mm/s on nearly as bright as a stream.) Sand and gas add a dark
// silhouette where nothing moves. Display only: no physics reads it.
const FLOW_FULL: f32 = 2.5e-2;  // 25 mm/s, the fastest streams at full brightness
const SAND_SHADE: f32 = 0.05;   // linear grey of still sand
const GAS_SHADE: f32 = 0.02;    // linear grey of still gas

struct Motion { momentum: vec2<f32>, volume: f32 }
fn motion(p: vec2<i32>) -> Motion {
    let c = cells[cell_index(p)];
    let volume = max(c.volume.water, 0.) + max(c.volume.fine, 0.) + max(c.volume.coarse, 0.) + max(c.volume.gas, 0.);
    return Motion(c.momentum[0] + c.momentum[1] + c.momentum[2] + c.momentum[3], volume);
}
// Hue h in [0, 1) to a fully saturated colour.
fn hue_colour(h: f32) -> vec3<f32> {
    let k = abs(fract(vec3<f32>(h) + vec3<f32>(0., 2. / 3., 1. / 3.)) * 6. - 3.) - 1.;
    return clamp(k, vec3<f32>(0.), vec3<f32>(1.));
}
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2. - 1., 0., 1.);
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = position.xy * vec2<f32>(cfg.size) / vec2<f32>(cfg.image_size);
    // Velocity interpolated between the four nearest cell centres, each
    // weighted by what it holds, so an empty neighbour pulls nothing.
    let sample = vec2<f32>(pixel.x, f32(cfg.size.y) - pixel.y) - 0.5;
    let base = floor(sample);
    let t = sample - base;
    var momentum = vec2<f32>(0.);
    var volume = 0.;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let w = select(1. - t.x, t.x, i == 1) * select(1. - t.y, t.y, j == 1);
            let m = motion(vec2<i32>(base) + vec2<i32>(i, j));
            momentum += w * m.momentum;
            volume += w * m.volume;
        }
    }
    var colour = vec3<f32>(0.);
    if volume > 0.01 {
        let velocity = momentum / volume;
        let speed = length(velocity);
        let level = sqrt(clamp(speed / FLOW_FULL, 0., 1.));
        let hue = fract(atan2(velocity.y, velocity.x) / 6.2831853);
        // The wheel is drawn in display space; the target encodes from
        // linear.
        colour = pow(hue_colour(hue) * level, vec3<f32>(2.2));
    }
    let x = min(u32(pixel.x), cfg.size.x - 1u);
    let y = cfg.size.y - 1u - min(u32(pixel.y), cfg.size.y - 1u);
    let c = composition(vec2<i32>(i32(x), i32(y)));
    let sand = clamp((c.y + c.z) / 0.3, 0., 1.);
    let gas = clamp(c.w, 0., 1.);
    let shade = vec3<f32>(max(SAND_SHADE * sand, GAS_SHADE * gas));
    return vec4<f32>(colour + shade, 1.);
}
