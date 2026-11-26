// The carry's schedule, a shader module of its own (carry_schedule.rs). In
// TransportPrepare, once the groups are fixed, reset, reduce and finish
// take the update's largest Courant number and choose its subdivisions;
// advance ends each subdivision slot of the carry.
struct Config {nx:u32,ny:u32,n:u32,table_side:u32,dt:f32,h:f32,gravity:f32,substeps:u32}
struct Face {data:array<vec4<f32>,10>}
struct Plan {maximum:atomic<u32>,steps:u32,current:u32,required:u32,sum_steps:u32,updates:u32,max_required:u32,maximum_speed:atomic<u32>,dispatch:vec3<u32>,padding:u32}
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> faces:array<Face>;
@group(0) @binding(2) var<storage,read_write> plan:Plan;
@compute @workgroup_size(1) fn reset(){atomicStore(&plan.maximum,0u);plan.current=0u;}
@group(0) @binding(3) var<storage,read_write> cells:array<array<vec4<f32>,6>>;
@group(0) @binding(4) var<storage,read_write> scratch:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> members:array<atomic<u32>>;
fn group_member(i:u32,k:u32)->u32{
    if k==0u{return i;}var j=atomicLoad(&members[i]);
    for(var n=1u;n<k;n++){j=atomicLoad(&members[cfg.n+j]);}return j;
}

var<workgroup> maxima:array<u32,128>;
var<workgroup> speeds:array<u32,128>;
// The largest Courant number, as float bits (which order as nonnegative
// floats do): for a group the common volume leaving it over its free
// inventory, for a face its common and gas velocities (words 4 and 1 of
// Face), times dt/h.
@compute @workgroup_size(128) fn reduce(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let i=id.x;var amount=0.;var speed=0.;
    if i<cfg.n {
        let x=i%cfg.nx;let y=i/cfg.nx;
        let map=scratch[GROUP_MAP_ROW*cfg.n+i];let owner=u32(map.x);
        if owner==i{
            let count=1u+u32(map.y);var free=0.;var outgoing=0.;
            for(var member=0u;member<count;member++){
                let j=group_member(i,member);
                free+=scratch[GROUP_MAP_ROW*cfg.n+j].z;
                let px=j%cfg.nx;let py=j/cfg.nx;
                let ids=array<u32,4>(2u*j,2u*(j-1u),2u*j+1u,2u*(j-cfg.nx)+1u);
                let neighbors=array<u32,4>(j+1u,j-1u,j+cfg.nx,j-cfg.nx);
                let valid=array<bool,4>(px+1u<cfg.nx,px>0u,py+1u<cfg.ny,py>0u);
                for(var k=0u;k<4u;k++){if valid[k]{
                    if u32(scratch[GROUP_MAP_ROW*cfg.n+neighbors[k]].x)==owner{continue;}
                    outgoing+=max(select(-1.,1.,k%2u==0u)*faces[ids[k]].data[4].w,0.);
                }}
            }
            if free>1e-12&&outgoing>0.{amount=min(outgoing/free,1e6);}
        }
        if x+1u<cfg.nx{speed=max(speed,max(abs(faces[2u*i].data[4].w),abs(faces[2u*i].data[1].w)));}
        if y+1u<cfg.ny{speed=max(speed,max(abs(faces[2u*i+1u].data[4].w),abs(faces[2u*i+1u].data[1].w)));}
        amount=max(amount,speed);
    }
    maxima[lane]=bitcast<u32>(amount*cfg.dt/cfg.h);speeds[lane]=bitcast<u32>(speed);workgroupBarrier();
    for(var width=64u;width>0u;width/=2u){if lane<width{
        maxima[lane]=max(maxima[lane],maxima[lane+width]);
        speeds[lane]=max(speeds[lane],speeds[lane+width]);
    }workgroupBarrier();}
    if lane==0u{atomicMax(&plan.maximum,maxima[0]);atomicMax(&plan.maximum_speed,speeds[0]);}
}
@compute @workgroup_size(1) fn finish(){
    // One subdivision more than the Courant number's whole part, so that
    // each stays below one; within the grid's budget, and with no floor (a
    // quiet update carries once).
    plan.required=u32(floor(clamp(bitcast<f32>(atomicLoad(&plan.maximum)),0.,1e6)))+1u;
    plan.steps=min(cfg.substeps,plan.required);
    plan.dispatch=vec3<u32>((cfg.n+63u)/64u,1u,1u);
    plan.sum_steps+=plan.steps;plan.updates+=1u;plan.max_required=max(plan.max_required,plan.required);
}
// Ends a slot. The carry kernels compare `current` with `steps` themselves;
// `dispatch` only records the reconstruction's workgroups for the slot.
@compute @workgroup_size(1) fn advance(){plan.current+=1u;plan.dispatch.x=select(0u,(cfg.n+63u)/64u,plan.current<plan.steps);}
