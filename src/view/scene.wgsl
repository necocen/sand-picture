// The first fragment of every view's shader. Rendering owns this
// configuration (sizes in cells and pixels, lengths in metres). No writable
// simulation binding is present.
struct Config {
    size: vec2<u32>,
    image_size: vec2<u32>,
    diameters: vec2<f32>,
    cell_width: f32,
    gap: f32,
    // The white's powder (material.rs), which scatters as its own small
    // grains.
    powder_diameter: f32,
}
@group(0) @binding(0) var<uniform> cfg: Config;

fn cell_index(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(clamp(p, vec2<i32>(0), vec2<i32>(cfg.size) - 1));
    return q.y * cfg.size.x + q.x;
}
