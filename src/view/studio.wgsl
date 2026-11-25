// A quiet backing and broad surrounding light. Distances are metres; colors
// are linear radiance. Water and bubbles see the same stationary scene.
fn world_position(p: vec2<f32>) -> vec2<f32> {
    return (p + .5 - .5 * vec2<f32>(cfg.size)) * cfg.cell_width;
}
fn rectangle(p: vec2<f32>, half_size: vec2<f32>, softness: f32) -> f32 {
    let edge = smoothstep(half_size - softness, half_size + softness, abs(p));
    return (1. - edge.x) * (1. - edge.y);
}
// How far behind the tank the backing stands (m). A meniscus turns a ray by
// tens of degrees: with the dark card this far back, a ray turned by thirty
// degrees or more passes its edge and sees the bright surround, and a
// bubble's rim, half a gap wide, shows as a line as wide.
const BACKING_DISTANCE: f32 = .3;
fn backing(q: vec2<f32>) -> vec3<f32> {
    // Grazing rays see the bright diffuser around the finite dark backing.
    // Keep the same 4 cm margin around the tank whether landscape or portrait.
    let half_tank = .5 * vec2<f32>(cfg.size) * cfg.cell_width;
    let card = rectangle(q, half_tank + vec2<f32>(.04, .04), .008);
    return mix(vec3<f32>(1.8, 1.85, 1.9), vec3<f32>(.004, .012, .035), card);
}
fn backdrop(p: vec2<f32>) -> vec3<f32> {
    return backing(world_position(p));
}
fn environment(p: vec2<f32>, ray: vec3<f32>) -> vec3<f32> {
    let q = world_position(p);
    if ray.z < -.001 {
        return backing(q + ray.xy * (BACKING_DISTANCE / -ray.z));
    }
    var light = mix(vec3<f32>(.018, .025, .032), vec3<f32>(.13, .16, .19), .5 + .5 * ray.y);
    if ray.z > .001 {
        let front = q + ray.xy * (.4 / ray.z);
        let key = rectangle(front - vec2<f32>(-.29, .15), vec2<f32>(.11, .28), .015);
        let fill = rectangle(front - vec2<f32>(.28, .02), vec2<f32>(.045, .20), .012);
        light += key * vec3<f32>(3.8, 3.6, 3.2) + fill * vec3<f32>(.65, .82, 1.);
    }
    return light;
}
