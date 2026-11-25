// Ideal instantaneous 180-degree and 90-degree rotations in the plane of the
// tank. Scalars rotate unchanged; vectors rotate with the frame; the 2D
// symmetric stress tensor keeps its trace/load while flipping (normal, shear)
// under 90-degree turns.
struct Config {nx:u32,ny:u32,n:u32,table_side:u32,dt:f32,h:f32,gravity:f32,substeps:u32}
// stress: pressure, normal, shear, load; rest: suction, the powder (mobile,
// bed, free).
struct Cell {volume:vec4<f32>,momentum:array<vec2<f32>,4>,stress:vec4<f32>,rest:vec4<f32>,bed:vec4<f32>}
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(2) var<storage,read_write> scratch:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> geometry:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> arrival:array<Cell>;
@group(0) @binding(7) var<storage,read_write> potential:array<f32>;
@group(0) @binding(8) var<storage,read_write> fcg:array<f32>;
// The relabelling's word of each cell (labels.wgsl, label_cell_word), from
// LABEL_CELL_WORDS on.
@group(0) @binding(16) var<storage,read_write> labels:array<u32>;
fn rotated(value:Cell)->Cell {
    var result=value;
    for(var phase=0u;phase<4u;phase++){result.momentum[phase]=-value.momentum[phase];}
    return result;
}
fn rotated_quarter(value:Cell,cw:bool)->Cell {
    var result=value;
    // Release floor-connected bed into mobile packed sand so the new floor
    // builds a fresh deposit while the old dune slumps freely.
    result.volume=vec4<f32>(
        value.volume.x+value.bed.w,
        value.volume.y+value.bed.x,
        value.volume.z+value.bed.y,
        value.volume.w,
    );
    result.rest=vec4<f32>(value.rest.x,value.rest.y+value.rest.z,0.,value.rest.w);
    result.bed=vec4<f32>(0.);
    for(var phase=0u;phase<4u;phase++){
        let m=value.momentum[phase];
        result.momentum[phase]=select(vec2<f32>(-m.y,m.x),vec2<f32>(m.y,-m.x),cw);
    }
    result.stress=vec4<f32>(value.stress.x,-value.stress.y,-value.stress.z,value.stress.w);
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
    // The sand that had given way to a draw lies under the water after the
    // turn, and the sand that hangs now was laid as the bed: none of it has
    // given way, and the last walk's supply is of the picture before.
    geometry[SET_IN_ROW*cfg.n+i]=vec4<f32>(0.);geometry[SET_IN_ROW*cfg.n+j]=vec4<f32>(0.);
}
@compute @workgroup_size(128) fn invert_stream(@builtin(global_invocation_id) id:vec3<u32>){
    let n=(cfg.nx+1u)*(cfg.ny+1u);let i=id.x;if i>=(n+1u)/2u{return;}
    let j=n-1u-i;let before=potential[i];potential[i]=potential[j];potential[j]=before;
}
// Stage a 90-degree rotation from `(old_nx, old_ny) = (cfg.ny, cfg.nx)` into
// `arrival` (cells) and `scratch` rows 0..3 (geometry) using the NEW stride `cfg.nx`.
fn stage_rotate_cells(i:u32,cw:bool){
    if i>=cfg.n{return;}
    let old_nx=cfg.ny;let old_ny=cfg.nx;
    let ox=i%old_nx;let oy=i/old_nx;
    let nx_p=select(old_ny-1u-oy,oy,cw);
    let ny_p=select(ox,old_nx-1u-ox,cw);
    let dst=ny_p*cfg.nx+nx_p;
    arrival[dst]=rotated_quarter(cells[i],cw);
    let ap=geometry[SOLVE_APERTURE_ROW*cfg.n+i];
    scratch[dst]=select(ap.wzxy,ap.zwyx,cw);
    scratch[cfg.n+dst]=geometry[SOLVE_STATE_ROW*cfg.n+i];
    let h=geometry[HOLD_ROW*cfg.n+i];
    scratch[2u*cfg.n+dst]=select(vec4<f32>(-h.y,h.x,h.z,h.w),vec4<f32>(h.y,-h.x,h.z,h.w),cw);
}
@compute @workgroup_size(128) fn rotate_cells_cw(@builtin(global_invocation_id) id:vec3<u32>){stage_rotate_cells(id.x,true);}
@compute @workgroup_size(128) fn rotate_cells_ccw(@builtin(global_invocation_id) id:vec3<u32>){stage_rotate_cells(id.x,false);}
@compute @workgroup_size(128) fn rotate_cells_commit(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    cells[i]=arrival[i];
    geometry[SOLVE_APERTURE_ROW*cfg.n+i]=scratch[i];
    geometry[SOLVE_STATE_ROW*cfg.n+i]=scratch[cfg.n+i];
    geometry[HOLD_ROW*cfg.n+i]=scratch[2u*cfg.n+i];
    geometry[SET_IN_ROW*cfg.n+i]=vec4<f32>(0.);
    labels[LABEL_CELL_WORDS+i]&=(1u<<GAS_SLOTS)-1u;
}
fn stage_rotate_stream(i:u32,cw:bool){
    let vn=(cfg.nx+1u)*(cfg.ny+1u);if i>=vn{return;}
    let old_vx=cfg.ny+1u;let old_vy=cfg.nx+1u;let new_vx=cfg.nx+1u;
    let ox=i%old_vx;let oy=i/old_vx;
    let nx_p=select(old_vy-1u-oy,oy,cw);
    let ny_p=select(ox,old_vx-1u-ox,cw);
    fcg[ny_p*new_vx+nx_p]=potential[i];
}
@compute @workgroup_size(128) fn rotate_stream_cw(@builtin(global_invocation_id) id:vec3<u32>){stage_rotate_stream(id.x,true);}
@compute @workgroup_size(128) fn rotate_stream_ccw(@builtin(global_invocation_id) id:vec3<u32>){stage_rotate_stream(id.x,false);}
@compute @workgroup_size(128) fn rotate_stream_commit(@builtin(global_invocation_id) id:vec3<u32>){
    let vn=(cfg.nx+1u)*(cfg.ny+1u);let i=id.x;if i>=vn{return;}
    potential[i]=fcg[i];
}
