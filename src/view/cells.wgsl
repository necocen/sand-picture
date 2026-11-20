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
