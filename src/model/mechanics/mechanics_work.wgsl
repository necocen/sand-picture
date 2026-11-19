// The grain constraint's working values of a cell, written and read within
// one flow cycle of an update: the weights, the pressure mode, the suction
// the return found, and the grains' force and velocity with their bounds.
struct GrainWork {
    weights:vec2<f32>, pressure_mode:f32, suction:f32, force:vec2<f32>,
    velocity:vec2<f32>, velocity_bound:vec2<f32>, force_bound:vec2<f32>,
}
@group(0) @binding(6) var<storage,read_write> grain_works:array<GrainWork>;
fn grain_work(i:u32)->GrainWork {return grain_works[i];}
fn put_grain_weights(i:u32,weights:vec2<f32>) {grain_works[i].weights=weights;}
fn put_pressure_mode(i:u32,mode:f32) {grain_works[i].pressure_mode=mode;}
fn put_grain_force(i:u32,value:vec2<f32>) {grain_works[i].force=value;}
fn put_grain_force_bound(i:u32,value:vec2<f32>) {grain_works[i].force_bound=value;}
fn put_grain_velocity(i:u32,value:vec2<f32>) {grain_works[i].velocity=value;}
fn put_grain_velocity_bound(i:u32,value:vec2<f32>) {grain_works[i].velocity_bound=value;}
fn put_returned_stress(i:u32,value:vec4<f32>) {scratch[RETURNED_STRESS_ROW*cfg.n+i]=value;}
fn put_returned_suction(i:u32,value:f32) {grain_works[i].suction=value;}
fn returned_stress(i:u32)->vec4<f32> {return scratch[RETURNED_STRESS_ROW*cfg.n+i];}
fn returned_suction(i:u32)->f32 {return grain_works[i].suction;}
