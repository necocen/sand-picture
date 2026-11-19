// The simulation's cells as the views read them: the layout of the model's
// Cell (shared.wgsl), bound read-only.
struct PhaseVolumes { water: f32, fine: f32, coarse: f32, gas: f32 }
struct GrainStress { pressure: f32, normal: f32, shear: f32, load: f32, suction: f32 }
struct CellPowder { mobile: f32, bed: f32, free: f32 }
struct Cell { @align(16) volume: PhaseVolumes, momentum: array<vec2<f32>, 4>, stress: GrainStress, powder: CellPowder, bed: vec4<f32> }
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
// Water, fine, coarse and gas of cell p, mobile and deposited together.
fn composition(p: vec2<i32>) -> vec4<f32> {
    let c = cells[cell_index(p)];
    return vec4<f32>(c.volume.water + c.bed.w, c.volume.fine + c.bed.x,
        c.volume.coarse + c.bed.y, c.volume.gas);
}
// The share of a cell's white grains, free and in the deposit, that shows as
// the white's powder: its free powder, dispersed as single grains. Powder
// bound to its grit, in the free grains or in the deposit, lies in the
// grit's aggregates and scatters with them as white grains of the grit's
// size: it adds no cross-section of its own and does not veil the black
// grains.
fn powder_share(p: vec2<i32>) -> f32 {
    let c = cells[cell_index(p)];
    let white = max(c.volume.fine, 0.) + max(c.bed.x, 0.);
    if white <= 0. { return 0.; }
    return clamp(max(c.powder.free, 0.) / white, 0., 1.);
}
