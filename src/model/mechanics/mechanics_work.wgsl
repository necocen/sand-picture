// Binding 6 is shared with transport payload storage (64 bytes/cell).
// Only these accessors interpret its mechanics fields. Store components in the
// original order, leaving unused bytes untouched; transport replaces them later.
struct GrainWork {
    weights:vec2<f32>, pressure_mode:f32, force:vec2<f32>,
    velocity:vec2<f32>, velocity_bound:vec2<f32>, force_bound:vec2<f32>,
}
fn grain_work(i:u32)->GrainWork {
    let c=compression[i];
    return GrainWork(c.volume.xy,c.volume.z,c.momentum[0],c.momentum[1],c.momentum[2],c.momentum[3]);
}
fn put_grain_weights(i:u32,weights:vec2<f32>) {
    compression[i].volume.x=weights.x;compression[i].volume.y=weights.y;
}
fn put_pressure_mode(i:u32,mode:f32) {compression[i].volume.z=mode;}
fn put_grain_force(i:u32,value:vec2<f32>) {compression[i].momentum[0]=value;}
fn put_grain_force_bound(i:u32,value:vec2<f32>) {compression[i].momentum[3]=value;}
fn put_grain_velocity(i:u32,value:vec2<f32>) {compression[i].momentum[1]=value;}
fn put_grain_velocity_bound(i:u32,value:vec2<f32>) {compression[i].momentum[2]=value;}
// Binding 2 holds returned stress here, and outgoing capacity during transport.
fn put_returned_stress(i:u32,value:vec4<f32>) {next[i]=value;}
fn put_returned_suction(i:u32,value:f32) {compression[i].stress.x=value;}
fn returned_stress(i:u32)->vec4<f32> {return next[i];}
fn returned_suction(i:u32)->f32 {return compression[i].stress.x;}
