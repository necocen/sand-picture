// Ideal instantaneous 180-degree turn in the plane of the tank. Scalars and
// the symmetric stress tensor rotate unchanged; vectors change sign.
struct Config {nx:u32,ny:u32,n:u32,table_side:u32,dt:f32,h:f32,gravity:f32,substeps:u32}
// stress: pressure, normal, shear, load; rest: suction, the powder (mobile,
// bed) and padding.
struct Cell {volume:vec4<f32>,momentum:array<vec2<f32>,4>,stress:vec4<f32>,rest:vec4<f32>,bed:vec4<f32>}
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(4) var<storage,read_write> geometry:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> potential:array<f32>;
// The relabelling's word of each cell (labels.wgsl, label_cell_word), from
// LABEL_CELL_WORDS on.
@group(0) @binding(16) var<storage,read_write> labels:array<u32>;
fn rotated(value:Cell)->Cell {
    var result=value;
    for(var phase=0u;phase<4u;phase++){result.momentum[phase]=-value.momentum[phase];}
    return result;
}
@compute @workgroup_size(128) fn invert_cells(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=(cfg.n+1u)/2u{return;}let j=cfg.n-1u-i;
    let a=rotated(cells[i]);let b=rotated(cells[j]);cells[i]=b;cells[j]=a;
    let row_i=geometry[SOLVE_APERTURE_ROW*cfg.n+i];let row_j=geometry[SOLVE_APERTURE_ROW*cfg.n+j];
    geometry[SOLVE_APERTURE_ROW*cfg.n+i]=row_j.yxwz;geometry[SOLVE_APERTURE_ROW*cfg.n+j]=row_i.yxwz;
    let value=geometry[SOLVE_STATE_ROW*cfg.n+i];
    geometry[SOLVE_STATE_ROW*cfg.n+i]=geometry[SOLVE_STATE_ROW*cfg.n+j];geometry[SOLVE_STATE_ROW*cfg.n+j]=value;
    // The plates' traction on the grains is a vector.
    let held=geometry[HOLD_ROW*cfg.n+i];let other=geometry[HOLD_ROW*cfg.n+j];
    geometry[HOLD_ROW*cfg.n+i]=vec4<f32>(-other.xy,other.zw);geometry[HOLD_ROW*cfg.n+j]=vec4<f32>(-held.xy,held.zw);
    for(var slot=0u;slot<GAS_SLOTS;slot++){
        let left=LAYER_ROW*cfg.n+GAS_SLOTS*i+slot;let right=LAYER_ROW*cfg.n+GAS_SLOTS*j+slot;
        let before=geometry[left];geometry[left]=geometry[right];geometry[right]=before;
    }
    // The cells' words stay in place, as their anchored identities always
    // have, but the gas has moved: they forget which slots hold gas until the
    // next relabelling says it again.
    labels[LABEL_CELL_WORDS+i]&=(1u<<GAS_SLOTS)-1u;labels[LABEL_CELL_WORDS+j]&=(1u<<GAS_SLOTS)-1u;
}
@compute @workgroup_size(128) fn invert_stream(@builtin(global_invocation_id) id:vec3<u32>){
    let n=(cfg.nx+1u)*(cfg.ny+1u);let i=id.x;if i>=(n+1u)/2u{return;}
    let j=n-1u-i;let before=potential[i];potential[i]=potential[j];potential[j]=before;
}
