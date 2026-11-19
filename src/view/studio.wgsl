// A quiet backing and broad surrounding light. Distances are metres; colors
// are linear radiance. Water and bubbles see the same stationary scene.
fn world_position(p: vec2<f32>) -> vec2<f32> {
    return (p + .5 - .5 * vec2<f32>(cfg.size)) * cfg.cell_width;
}
fn rectangle(p: vec2<f32>, half_size: vec2<f32>, softness: f32) -> f32 {
    let edge = smoothstep(half_size - softness, half_size + softness, abs(p));
    return (1. - edge.x) * (1. - edge.y);
}
fn backing(q: vec2<f32>) -> vec3<f32> {
    // Grazing rays see the bright diffuser around the finite dark backing.
    // The bubble centres still transmit the same quiet background as the
    // water.
    let card = rectangle(q, vec2<f32>(.19, .14), .008);
    return mix(vec3<f32>(1.8, 1.85, 1.9), vec3<f32>(.004, .012, .035), card);
}
fn backdrop(p: vec2<f32>) -> vec3<f32> {
    return backing(world_position(p));
}
fn environment(p: vec2<f32>, ray: vec3<f32>) -> vec3<f32> {
    let q = world_position(p);
    if ray.z < -.001 {
        return backing(q + ray.xy * (.08 / -ray.z));
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
