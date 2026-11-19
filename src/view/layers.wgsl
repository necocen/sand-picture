// Photoreal's layer mask, a compute pass after the distance field: per cell,
// the gas layers whose distance differs from the flat cap (+4) on some texel
// within two cells, the texels a pixel of this cell can sample for its
// outline (bubble_at). A layer at +4 on all of them averages to exactly +4
// and is never the nearest, so the picture skips it.
@group(0) @binding(1) var distances: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> layers: array<u32>;

@compute @workgroup_size(8, 8)
fn mark_layers(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= cfg.size) { return; }
    let p = vec2<i32>(id.xy);
    let last = vec2<i32>(cfg.size) - 1;
    var mask = 0u;
    for (var l = 0u; l < GAS_SLOTS; l++) {
        for (var y = -2; y <= 2; y++) {
            for (var x = -2; x <= 2; x++) {
                let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), last);
                if textureLoad(distances, q, i32(l), 0).x != 4. { mask |= 1u << l; }
            }
        }
    }
    layers[cell_index(p)] = mask;
}
