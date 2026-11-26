// The local dilation rate is the wet-volume average of relative grain
// divergence. Pressure P is stored per whole cell: intrinsic wet pressure is
// P/W. This face force is the transpose of D(A*delta_grain)/W.
fn wet_pressure(i:u32)->f32 {
    // A trace of wet material left in a gas-filled cell has no grain skeleton:
    // the stale P of the grains that left would be divided by almost nothing.
    let wet=free_wet(cells[i]);if wet<=1e-6{return 0.;}
    return active_pressure(i)/wet;
}
fn relative_pressure_force(f:u32)->f32 {
    let p=point(f/2u);let q=p+axis(f%2u);
    if !valid(q)||faces[f].coarse_momentum.z<=0.{return 0.;}
    return faces[f].coarse_momentum.z*(wet_pressure(f/2u)-wet_pressure(at(q)))/cfg.h;
}
// Change of grain-relative velocity per unit common velocity. Evaluate the
// liquid factor explicitly so the exactly dry limit remains zero. Where the
// grains cannot move relative to the water, every phase moves with the common
// velocity and the contact pressure does its work there, as the shear load
// does: sand pressing on a bubble's meniscus moves the interface, and the
// bubble's lift loads the sand.
fn pressure_common_response(f:u32)->f32 {
    let face=faces[f];let c=face.composition.xyz;let solid=c.y+c.z;
    if phase_locked(f)&&solid>0.{return 1.;}
    if c.x<=0.||solid<=0.||face.response.w<=0.{return 0.;}
    if coarse_locked(f){
        let pair=c.x*(c.y/(c.x+c.y));
        return -(pair/solid)*face.water_momentum.y;
    }
    return 1.*c.x*(dot(c.yz/solid,face.response.yz)-face.response.x)/face.response.w;
}
// On a face where grains slip, pressure does work on delta_grain, not common
// translation. The common generalized load is contrast*force; forming
// share*force-force would lose the dry-side scale. Shear retains its existing
// absolute-velocity work.
fn grain_load(f:u32,weight:f32,shear:ShearLoad)->vec4<f32> {
    let face=faces[f];let c=face.composition.xyz;let solid=c.y+c.z;
    let contrast=pressure_common_response(f);
    let force=weight*face.velocity.w+contrast*relative_pressure_force(f)
        +shear.common_force;
    if phase_locked(f)||c.x==0.{return vec4<f32>(face.velocity.xyz,force);}
    return vec4<f32>(preferred(f,shear.velocity),force);
}

fn normal_pressure_mode(p:vec2<i32>)->bool {
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);
        if valid(base)&&valid(base+axis(a)){
            let f=2u*at(base)+a;
            if faces[f].coarse_momentum.z>0.{return true;}
        }
    }}return false;
}
// The pressure that pushes is the load part; dilatancy suction holds the
// grains (friction_limit, the yield) without pushing the neighbours.
fn active_pressure(i:u32)->f32 {
    if grain_work(i).pressure_mode==0.{return 0.;}
    return cells[i].stress.load;
}

// Sum of the two material amounts and their momentum component. The factor
// one half in the staggered dual volume cancels in normalized pair forces.
fn face_material(p:vec2<i32>,a:u32,phase:u32)->vec2<f32> {
    let q=p+axis(a);if !valid(p)||!valid(q){return vec2<f32>(0.);}
    let l=cells[at(p)];let r=cells[at(q)];
    let vl=max(mobile_volumes(l)[phase],0.);let vr=max(mobile_volumes(r)[phase],0.);
    var momentum=0.;if vl>0.{momentum+=l.momentum[phase][a];}
    if vr>0.{momentum+=r.momentum[phase][a];}
    return vec2<f32>(vl+vr,momentum);
}
fn viscous_laplacian(p:vec2<i32>,a:u32,phase:u32)->f32 {
    let old=face_history(p,a,phase);var result=0.;
    let current=face_material(p,a,phase);if current.x<=0.{return 0.;}
    for(var d=0u;d<2u;d++){for(var side=0u;side<2u;side++){
        let q=p+select(-axis(d),axis(d),side==1u);
        if !valid(q)||!valid(q+axis(a)){result-=old;continue;}
        let neighbor=face_material(q,a,phase);if neighbor.x<=0.{continue;}
        // Harmonic conductance 2 A B/(A+B), divided by receiver amount A.
        // Work with donor momentum, avoiding a huge trace velocity times a
        // tiny conductance. Pair forces are opposite and dissipative.
        let total=current.x+neighbor.x;
        result+=2.*(neighbor.y/total-(neighbor.x/total)*old);
    }}
    return result;
}

@compute @workgroup_size(128) fn project_faces(@builtin(global_invocation_id) id:vec3<u32>){
    let f=id.x;if f>=2u*cfg.n{return;}
    // The sand's (shared) velocity and the gas's; they differ on a gliding face.
    let relative=corrected_relative(f);let velocity=wet_velocity(f);let current=volume_current(f);
    if faces[f].response.w>0.{
        faces[f].support.z=(velocity-dot(faces[f].composition.xyz,faces[f].volume.xyz))/faces[f].response.w;
    }
    faces[f].volume=vec4<f32>(vec3<f32>(velocity)+relative,current);
    faces[f].velocity=vec4<f32>(relative,gas_velocity(f));
    // A changing aperture preserves a velocity trace for the local predictor.
    // Reusing old volume current / new area would invent a velocity impulse.
    scratch[FACE_PREDICTION_ROW*cfg.n+f]=vec4<f32>(velocity,0.,0.,0.);
}
// A bubble smaller than the gap moves with the flow and its own slip through
// the water (gas_transport.wgsl, bubble_slip), whatever the faces' gas carries
// for its neighbours.
@compute @workgroup_size(128) fn apply_momentum(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.n{return;}let p=point(i);let volume=mobile_volumes(cells[i]);
    let slip=bubble_slip(i);let slips=slip.x!=0.||slip.y!=0.;
    for(var phase=0u;phase<4u;phase++){
        var velocity=vec2<f32>(0.);var amount=vec2<f32>(0.);
        for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
            let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
            let index=2u*at(base)+a;let f=faces[index];var weight=face_gas(index);
            if phase<3u{weight=f.coarse_momentum.z*f.composition.w*f.composition[phase];}
            if weight>0.{
                var speed=f.velocity.w;if phase<3u{speed=f.volume[phase];}else if slips{speed=stream_velocity(index);}
                velocity[a]+=weight*speed;amount[a]+=weight;
            }
        }}
        for(var a=0u;a<2u;a++){if amount[a]>1e-30{velocity[a]/=amount[a];}else{velocity[a]=0.;}}
        if phase==3u&&slips{velocity+=slip;}
        cells[i].momentum[phase]=volume[phase]*velocity;
    }
}
// Geometric volume-of-fluid transport of independently owned gas volume.
