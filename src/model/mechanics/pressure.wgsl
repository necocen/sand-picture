// The grain pressure relaxed within the first projection. One return per
// projection moves each cell's pressure to its local Darcy-dilatancy balance
// against the neighbours' previous pressures: a change would travel one cell
// per projection, and the sand's strength against a push would scale with
// h^2/dt. Red-black sweeps of the same local return, after return_grains,
// carry the change across the sand within the update.

// The response of a face's compression speed to its relative pressure force
// with the common velocity held fixed: the linear part of
// relative_at_velocity. Trace grains have none. The grains enter as their
// shares, so that no product of two grain amounts is formed.
fn compression_response(f:u32)->f32{
    let face=faces[f];let c=face.composition.xyz;let solid=c.y+c.z;let wet=face.composition.w;
    if phase_locked(f)||c.x<=0.||solid<TRACE_VOLUME||wet<=0.{return 0.;}
    if coarse_locked(f){
        let mobile=c.x+c.y;
        return (c.x/solid)*(c.y/mobile)*face.water_momentum.w/(wet*solid);
    }
    if face.response.w<=0.{return 0.;}
    let r=face.response.xyz;let g=face.grain_response.xyz;let share=c.yz/solid;
    return (c.x/solid)*(share.x*(r.x*g.y-r.y*g.x)+share.y*(r.x*g.z-r.z*g.x))/(face.response.w*wet);
}
// Its response through the common flow. The pressure pushes the common
// motion by contrast times the force (grain_load) against the face's weight,
// at least composition.w/response.w, and that motion changes the grains'
// relative velocity by contrast again. The flow the push drives through the
// face must come back around its two cells: in series with the face, each
// cell's other faces in parallel (the flow solve's own reduction, one level
// deep: a push that moves a bubble along the sand returns through the gas
// beside it, a push into the slab's base only through the packed sand
// behind). Where the grains stand on the plates and water and gas move past
// them (contrast near -1, the raft's top) this path is 40 to 160 times the
// slip path: left out, the pressure there alternates and grows from update
// to update; unreduced, it lets the raft's buoyancy push a dome into the
// slab. A gliding face's wet phases stand, and no pressure pushes its gas.
fn common_compression_response(f:u32)->f32{
    let face=faces[f];let c=face.composition.xyz;let solid=c.y+c.z;let wet=face.composition.w;
    if phase_locked(f)||glides(f)||c.x<=0.||solid<TRACE_VOLUME||wet<=0.||face.response.w<=0.{return 0.;}
    let own=solve_conductance(f);if own<=0.{return 0.;}
    let i=f/2u;let j=at(point(i)+axis(f%2u));
    let behind=vec2<f32>(other_conductance(i,f),other_conductance(j,f));
    if behind.x<=0.||behind.y<=0.{return 0.;}
    let contrast=pressure_common_response(f);
    return contrast*contrast*face.response.w/wet/(1.+own/behind.x+own/behind.y);
}
// The flow solve's conductance of a face (prepare_stream), zero when closed.
fn solve_conductance(f:u32)->f32{
    let x=faces[f].coarse_momentum.x;return select(0.,1./x,x>0.);
}
// A cell's faces other than f, in parallel.
fn other_conductance(i:u32,f:u32)->f32{
    let p=point(i);var total=0.;
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(a)){continue;}
        let g=2u*at(base)+a;if g!=f{total+=solve_conductance(g);}
    }}
    return total;
}
// The couplings (s + c) a^2/h^2 of the cell's two owned faces, and the
// pressure the flow was solved with and its wet factor. Every cell writes
// them, grains or not: a neighbour's pressure change is measured against
// them.
fn cache_pressure_couplings(i:u32){
    let p=point(i);let c=cells[i];let wet=free_wet(c);
    for(var a=0u;a<2u;a++){
        let f=2u*i+a;var coupling=0.;
        if valid(p+axis(a)){
            let face=faces[f];let aperture=face.coarse_momentum.z;
            if face.composition.y+face.composition.z>0.&&aperture>0.{
                coupling=(compression_response(f)+common_compression_response(f))*aperture*aperture/(cfg.h*cfg.h);
            }
        }
        scratch[PRESSURE_COUPLING_ROW*cfg.n+f]=vec4<f32>(coupling,0.,0.,0.);
    }
    var factor=0.;if wet>0.&&!(grain_work(i).pressure_mode==0.){factor=1./wet;}
    scratch[PRESSURE_ITERATE_ROW*cfg.n+i]=vec4<f32>(c.stress.pressure,factor,0.,0.);
    scratch[PRESSURE_MATERIAL_ROW*cfg.n+i]=vec4<f32>(0.);
}
// The local problem of a cell whose return applies (dilation rate at the
// solved flow, pressure response, shear compliance, trial strength), its
// material (grain and wet volume, shear-rate norm), its first iterate and
// its trial stress.
fn cache_pressure_problem(i:u32,theta:f32,alpha:f32,beta:f32,trial:vec2<f32>,material:vec3<f32>,pressure:f32){
    scratch[PRESSURE_PROBLEM_ROW*cfg.n+i]=vec4<f32>(theta,alpha,beta,scaled_norm(trial));
    scratch[PRESSURE_MATERIAL_ROW*cfg.n+i]=vec4<f32>(material,1.);
    scratch[PRESSURE_ITERATE_ROW*cfg.n+i].z=max(pressure,0.);
    scratch[PRESSURE_TRIAL_ROW*cfg.n+i]=vec4<f32>(trial,0.,0.);
}
fn pressure_iterate(i:u32)->f32{return scratch[PRESSURE_ITERATE_ROW*cfg.n+i].z;}
// Change of the wet pressure on the faces since the flow solve.
fn pressure_shift(i:u32)->f32{
    let base=scratch[PRESSURE_ITERATE_ROW*cfg.n+i];return (base.z-base.x)*base.y;
}
fn coupled_last(i:u32,p:vec2<i32>)->f32{
    let own=pressure_shift(i);var coupled=0.;
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;let coupling=scratch[PRESSURE_COUPLING_ROW*cfg.n+f].x;if coupling==0.{continue;}
        let other=select(at(base),at(base+axis(a)),side==1u);
        coupled+=coupling*(own-pressure_shift(other));
    }}
    return coupled;
}
// The cell of colour `color` (x + y even or odd) that sweep thread t serves:
// every row holds half its cells of each colour, alternating.
fn colour_cell(t:u32,color:u32)->u32{
    let across=(cfg.nx+1u)/2u;let y=t/across;let x=2u*(t%across)+(color+y)%2u;
    if y>=cfg.ny||x>=cfg.nx{return cfg.n;}
    return y*cfg.nx+x;
}
// One sweep of one colour, with the flow held fixed. Cells of one colour
// share no face and read the other colour's newest pressure (Gauss-Seidel);
// the change is over-relaxed by OMEGA (Grid::pressure_omega) and projected
// to p >= 0. The last pair is plain Gauss-Seidel and writes the returned
// stress, that of a local solve.
fn relax_pressure(t:u32,color:u32,last:bool){
    let i=colour_cell(t,color);if i>=cfg.n{return;}
    let p=point(i);
    let problem=scratch[PRESSURE_PROBLEM_ROW*cfg.n+i];let material=scratch[PRESSURE_MATERIAL_ROW*cfg.n+i];
    var pressure=0.;var returned=0.;let old=pressure_iterate(i);let mu=grain_friction(grains(cells[i]));
    if material.w>0.{
        let own=pressure_shift(i);var coupled=0.;
        for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
            let base=p-select(axis(a),vec2<i32>(0),side==1u);
            if !valid(base)||!valid(base+axis(a)){continue;}
            let f=2u*at(base)+a;let coupling=scratch[PRESSURE_COUPLING_ROW*cfg.n+f].x;if coupling==0.{continue;}
            let other=select(at(base),at(base+axis(a)),side==1u);
            coupled+=coupling*(own-pressure_shift(other));
        }}
        let theta=problem.x+coupled/material.y;
        if problem.z>0.&&(problem.y>0.||theta==0.){
            let state=joint_return(material.x,material.y,theta,problem.y,problem.z,old,problem.w,mu);
            pressure=state.x;returned=state.y;
        }else{
            pressure=particle_pressure(material.x,material.y,material.z,theta,problem.y,old);
            returned=shear_return(problem.w,material.x,material.y,pressure,problem.z,mu);
        }
    }
    if !last{scratch[PRESSURE_ITERATE_ROW*cfg.n+i].z=max(old+OMEGA*(pressure-old),0.);return;}
    scratch[PRESSURE_ITERATE_ROW*cfg.n+i].z=max(pressure,0.);
    if material.w==0.{return;}
    let theta_last=problem.x+coupled_last(i,p)/material.y;
    let darcy=dilatancy_suction(i,theta_last);
    if darcy>0.{returned=shear_return(problem.w,material.x,material.y,pressure+darcy,problem.z,mu);}
    if pressure+darcy<=0.{put_returned_stress(i,vec4<f32>(0.));put_returned_suction(i,0.);return;}
    var tau=scratch[PRESSURE_TRIAL_ROW*cfg.n+i].xy;if problem.w>0.{tau*=returned/problem.w;}
    // The load is measured from the law's memory, the stored pressure the
    // update began with, as the second return measures it. The sweeps' own
    // iterate already carries the dilatancy demand of the earlier sweeps:
    // measured from it, the demand of a sheared packing that cannot draw
    // water is booked as contact pressure, kept, and ratchets.
    let load=load_part(material.x,theta_last,problem.y,cells[i].stress.pressure,pressure);
    let suction=(pressure-load)+darcy;
    put_returned_stress(i,vec4<f32>(load,tau,load));put_returned_suction(i,suction);
}
@compute @workgroup_size(128) fn pressure_red(@builtin(global_invocation_id) id:vec3<u32>){relax_pressure(id.x,0u,false);}
@compute @workgroup_size(128) fn pressure_black(@builtin(global_invocation_id) id:vec3<u32>){relax_pressure(id.x,1u,false);}
@compute @workgroup_size(128) fn pressure_red_last(@builtin(global_invocation_id) id:vec3<u32>){relax_pressure(id.x,0u,true);}
@compute @workgroup_size(128) fn pressure_black_last(@builtin(global_invocation_id) id:vec3<u32>){relax_pressure(id.x,1u,true);}
