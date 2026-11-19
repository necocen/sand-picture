// The velocity at which the suspension arrives on the bed, and the kernel of
// the capture (exchange_from_source), which runs once an update, after the
// first flow solve.

// The velocity (m/s) at which `phase` (0 water, 1 fine, 2 coarse) arrives in
// cell i, from the projected face velocities: across, the mean of the two
// side faces that pass grains, weighted by the phase they hold; down, the
// upper face's alone. The lower face belongs to the bed, which stands:
// averaged in, it would halve the arriving speed.
fn incident_velocity(i:u32,phase:u32)->vec2<f32>{
    let p=point(i);var velocity=vec2<f32>(0.);var weight=0.;
    for(var side=0u;side<2u;side++){
        let base=p-select(axis(0u),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(0u)){continue;}
        let f=faces[2u*at(base)];let amount=f.composition.w*f.composition[phase];
        if amount>0.&&f.support.w==0.{velocity.x+=amount*f.volume[phase];weight+=amount;}
    }
    if weight>0.{velocity.x/=weight;}
    if valid(p+axis(1u)){
        let f=faces[2u*i+1u];
        if f.composition.w*f.composition[phase]>0.{velocity.y=f.volume[phase];}
    }
    return velocity;
}
// The arriving grains' volume times velocity in cell i, both species.
fn arriving_momentum(i:u32)->vec2<f32>{
    let mobile=grains(arrival[i]);
    return mobile.x*incident_velocity(i,1u)+mobile.y*incident_velocity(i,2u);
}
@compute @workgroup_size(128) fn bed_exchange(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x<cfg.nx{exchange_from_source(id.x);}
}
