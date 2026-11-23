// The mechanics of the faces and of the grains' stress around the update's
// two flow solves: predict rebuilds each face, prepare_grain_stress and
// prepare_stream give the solve its weights and right-hand side,
// return_grains returns each cell's stress, hold_friction ends the update.

// The grain fractions (fine, coarse) of a face that count in its cells'
// grain velocity (prepare_velocity) and stress force (shear_force): none
// below TRACE_VOLUME, whose velocity is leftover momentum over a vanishing
// amount.
fn traced_grains(face:Face)->vec2<f32>{
    let g=face.composition.yz;return select(vec2<f32>(0.),g,g>=vec2<f32>(TRACE_VOLUME));
}
// Per cell: the weights of its grain velocity per axis, its stress force
// and the bound of that force's absolute row, both per bulk volume (per
// grain volume a nearly empty cell's are unbounded), and its pressure mode.
@compute @workgroup_size(128) fn prepare_grain_stress(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.n{return;}let p=point(i);var weights=vec2<f32>(0.);
    for(var a=0u;a<2u;a++){for(var d=-1;d<=1;d++){for(var side=0u;side<2u;side++){
        let base=p+d*axis(1u-a)-select(axis(a),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(a)){continue;}
        let f=faces[2u*at(base)+a];let g=traced_grains(f);
        weights[a]+=transverse_weight(d)*f.coarse_momentum.z*f.composition.w*(g.x+g.y);
    }}}
    let force=stress_force(p);var normalized=vec2<f32>(0.);var bound=vec2<f32>(0.);
    let amount_bound=(grain_amount(p-axis(0u))+grain_amount(p+axis(0u))+grain_amount(p-axis(1u))+grain_amount(p+axis(1u)))/(2.*cfg.h)+GAP_SHEAR*grain_amount(p);
    normalized=force;bound=vec2<f32>(amount_bound);
    put_grain_weights(i,weights);put_pressure_mode(i,select(0.,1.,normal_pressure_mode(p)));
    put_grain_force(i,normalized);
    put_grain_force_bound(i,bound);
}
struct ShearLoad { velocity:vec3<f32>,common_force:f32,relative:f32,bound:f32 }
// force * mobility / amount; split exponents keep a trace amount in range.
fn response_product(force:f32,mobility:f32,amount:f32)->f32 {
    if force==0.||mobility==0.||amount<=0.{return 0.;}
    let f=frexp(force);let m=frexp(mobility);let a=frexp(amount);
    return ldexp((f.fract*m.fract)/a.fract,f.exp+m.exp-a.exp);
}
fn bulk_velocity_response(f:u32,force:f32)->vec3<f32> {
    let face=faces[f];let amount=face.composition.w*(face.composition.y+face.composition.z);
    var result=vec3<f32>(0.);
    for(var phase=0u;phase<3u;phase++){
        result[phase]=response_product(force,face.grain_response[phase],amount);
    }
    return result;
}
// A face's share of its cells' stress force and of that force's bound: the
// transpose of prepare_velocity's aperture-weighted mean, so that force and
// velocity do the same work; a closed face takes exactly none.
fn shear_force(f:u32)->ShearLoad {
    let face=faces[f];let p=point(f/2u);let a=f%2u;
    let traced=traced_grains(face);
    let solid=face.composition.y+face.composition.z;
    let amount=face.composition.w*solid;var result:ShearLoad;
    if amount<=0.||traced.x+traced.y<=0.{return result;}
    var bulk=vec2<f32>(0.);
    for(var d=-1;d<=1;d++){
        let w=transverse_weight(d);if w<=0.{continue;}
        for(var side=0u;side<2u;side++){
            let q=p+d*axis(1u-a)+select(vec2<i32>(0),axis(a),side==1u);
            if !valid(q){continue;}let c=grain_work(at(q));let total=c.weights[a];
            if total>0.{
                let share=w*phase_ratio(face.coarse_momentum.z*face.composition.w*(traced.x+traced.y),total);
                bulk+=share*vec2<f32>(c.force[a],c.force_bound[a]);
            }
        }
    }
    result.velocity=bulk_velocity_response(f,bulk.x);
    result.common_force=grain_translation_response(f)*bulk.x;
    if coarse_locked(f){result.relative=response_product(bulk.x,face.water_momentum.w,amount);}
    let mobility=max(dot(face.composition.yz/solid,face.grain_response.yz),0.);
    result.bound=response_product(bulk.y,mobility,amount);
    return result;
}
// Coulomb friction of the grains on the plates and the tank's walls, which
// press on them with the wet grain pressure p = P/W and hold a traction tau
// per area, |tau| <= mu p. A cell's layer is stuck or sliding (HOLD_ROW).
// Stuck, it keeps its traction, changed by the contacts' stiffness times the
// slip, tau' = tau - (mu p/SLIP) u dt: in the flow solve, the held force and
// a drag of that stiffness. Loaded past mu p and creeping faster than CREEP
// it slides, with the kinetic mu p as the secant drag mu p/|u| at its
// previous speed, which cannot reverse the slip within the update, until it
// slows to less than SLIP per update.
const KINETIC_FRICTION:f32=.32;
// Static friction exceeds the kinetic, as for any dry contact (the peak over
// the residual wall friction of sand on glass).
const STATIC_FRICTION:f32=.4;
// The slip (m) that mobilises the static friction: a small part of a grain.
const SLIP:f32=1e-6;
// The creep (m/s) of a layer loaded past its friction before it slides.
const CREEP:f32=2e-5;
// Per surface area: the held force (Pa) and the drag (Pa s/m).
struct Traction {force:vec2<f32>,drag:f32}
// K = 1 - sin(33 deg) for sand at rest.
const LATERAL_STRESS_RATIO:f32=.455;
// The plates and the tank's wall hold a packed body by K sigma_v: the
// submerged weight of the packed column above the cell, which saturates at
// Janssen's rho' phi g b/(2 mu K) as the friction above carries the rest
// (51 Pa at BED_PACKING, half the sand's volume being black). It is there
// whether or not a bubble presses from below: the law's contact pressure is
// the raft's push and goes when the bubble leaves. A body's last layer is
// held as its body is: it is the same packing over a part of its cell
// (dense_volume), and without the plates' hold a layer hanging under its
// body would fall as loose sand as soon as its face had breached a little.
fn overburden_stress(i:u32)->f32{
    let p=point(i);let c=cells[i];let g=grains(c);let solid=g.x+g.y;let wet=dense_volume(i);
    if wet<=0.||solid<CRITICAL_PACKING*wet{return 0.;}
    let saturation=dot(g,RHO.yz-vec2<f32>(RHO.x))*cfg.gravity*PLATE_GAP/(2.*KINETIC_FRICTION*LATERAL_STRESS_RATIO);
    var weight=0.;
    for(var y=p.y+1;y<i32(cfg.ny)&&weight<saturation;y++){
        let cj=cells[at(vec2<i32>(p.x,y))];let gj=grains(cj);let sj=gj.x+gj.y;let wj=max(cj.volume.water,0.)+sj;
        if wj<=0.||sj<CRITICAL_PACKING*wj{break;}
        weight+=dot(gj,RHO.yz-vec2<f32>(RHO.x))*cfg.gravity*cfg.h;
    }
    return LATERAL_STRESS_RATIO*min(weight,saturation);
}
// The friction limit of a cell's layer (Pa): mu times the larger of its wet
// contact pressure plus dilatancy suction and of K sigma_v.
fn friction_limit(i:u32,c:Cell,mu:f32)->f32{
    let wet=dense_volume(i);let g=grains(c);
    if wet<=.01||g.x+g.y<=.005{return 0.;}
    // An isolated 1-3 cell speck floating in open water (no wall, bed, bubble,
    // or packed body of >= 3 neighbors around it) cannot arch against the glass
    // plates and must fall under gravity.
    let p=point(i);
    if p.x>0&&p.x<i32(cfg.nx)-1&&p.y>0&&p.y<i32(cfg.ny)-1{
        var supported=false;var packed_nbrs=0u;
        for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
            if dx==0&&dy==0{continue;}
            let cj=cells[at(p+vec2<i32>(dx,dy))];
            let gj=grains(cj);let sj=gj.x+gj.y;let wj=max(cj.volume.water,0.)+sj;
            if cj.bed.z>0.||cj.volume.gas>0.10{supported=true;}
            if wj>0.&&sj>=FREEZING_PACKING*wj{packed_nbrs+=1u;}
        }}
        if !supported&&packed_nbrs<3u{return 0.;}
    }
    // A layer whose support flows away slides to fill the voids on its load
    // alone: the suction resists the shear of a packing that must dilate,
    // not a filling the kinematic model imposes. Only in the slow flow an
    // orifice admits (voids no faster than the packing breaches): above a
    // free fall the layer keeps its dilatancy strength.
    let voids=void_speed(i);
    let suction=select(c.stress.suction,0.,voids>CREEP&&voids<breaching_speed(i));
    let load=select(c.stress.pressure,0.,free_face(i));
    // Tipped toward flat, the grains' submerged weight across the gap lies
    // on the lower plate: half of it on each plate's friction.
    let across=.5*dot(g,RHO.yz-vec2<f32>(RHO.x))/wet*plate_gravity()*PLATE_GAP;
    return mu*(max(max(load+suction,0.)/wet,overburden_stress(i))+across);
}
// A packed cell beside open water through which the grains it releases fall
// away (body_faces) is breaching and bears no normal effective stress: the
// plates hold it by K sigma_v and the suction, not by the law's contact
// pressure, in a static packing the memory of its last compaction. A
// funnel's wall, whose grains come to rest on its floor, keeps it and stands.
fn free_face(i:u32)->bool{
    let p=point(i);let left=p-axis(0u);let right=p+axis(0u);
    if valid(right)&&open_water(cells[at(right)])&&scratch[BODY_ROW*cfg.n+at(right)].z>0.{return true;}
    if valid(left)&&open_water(cells[at(left)])&&scratch[BODY_ROW*cfg.n+at(left)].w>0.{return true;}
    return false;
}
fn limited(tau:vec2<f32>,limit:f32)->vec2<f32>{
    let size=length(tau);if size>limit{return tau*(limit/size);}
    return tau;
}
fn grain_speed(c:Cell)->vec2<f32>{
    let g=grains(c);if g.x+g.y<=0.{return vec2<f32>(0.);}
    return (c.momentum[1]+c.momentum[2])/(g.x+g.y);
}
// Grains that flow out from under a layer leave voids that rise and spread
// sideways: in the kinematic model of silo flow (Nedderman and Tuzun 1979)
// the downward speed diffuses upward with a length K, which in a thin silo
// between smooth plates is 0.26 b whatever the grain size or the discharge
// (Hung, Aussillous and Capart 2018). K/h must stay below 1/2.
const VOID_SPREAD:f32=.26*PLATE_GAP;
// The speed at which voids reach a layer: the downward grain flux of the
// three cells below, spread by K per row and shared by the layer's grains.
// The plates cannot hold a layer whose support flows away. A layer on a
// bubble's inside is held by the bubble, whatever a joint beside it drains;
// a throat only shares the bubbles' rims and does undermine the body above.
// A link of the plug descent follows its voids in the measure of its supply
// (SET_IN_ROW, the last walk's): the walk brings its sand down by that share
// of what is asked of it, and a layer that slid faster would be stretched
// with nothing to fill it, lose its dilatancy suction (friction_limit) and
// loosen. Sand that is no link stands on nothing packed: the carry moves it.
fn void_speed(i:u32)->f32{
    let p=point(i);if p.y==0{return 0.;}
    let own=grains(cells[i]);let solid=own.x+own.y;if solid<=0.{return 0.;}
    if bubble_inside(cells[at(p-axis(1u))]){return 0.;}
    let k=VOID_SPREAD/cfg.h;var speed=0.;
    for(var dx=-1;dx<=1;dx++){
        let c=cells[at(inside(p+vec2<i32>(dx,-1)))];let g=grains(c);
        let down=max(-grain_speed(c).y,0.)*min((g.x+g.y)/solid,1.);
        speed+=select(k,1.-2.*k,dx==0)*down;
    }
    let set_in=geometry[SET_IN_ROW*cfg.n+i];
    if set_in.z>0.{return speed*set_in.y;}
    return 0.;
}
fn traction(i:u32)->Traction{
    let c=cells[i];let state=geometry[HOLD_ROW*cfg.n+i];
    let limit=friction_limit(i,c,select(KINETIC_FRICTION,STATIC_FRICTION,state.z==0.));
    if limit<=0.{return Traction(vec2<f32>(0.),0.);}
    if state.z==0.{return Traction(limited(state.xy,limit),limit*cfg.dt/SLIP);}
    return Traction(vec2<f32>(0.),limit/max(max(length(grain_speed(c)),CREEP),void_speed(i)));
}
// Contact area of the solid surfaces per volume at a face: the two plates,
// and the tank's wall where the face runs along it.
fn surface_area(p:vec2<i32>,a:u32)->f32{
    let normal=1u-a;let size=vec2<i32>(i32(cfg.nx),i32(cfg.ny));
    if p[normal]==0||p[normal]==size[normal]-1{return GAP_SHEAR+1./cfg.h;}
    return GAP_SHEAR;
}
// The friction at a face per grain volume: x the force, y the drag.
fn face_friction(p:vec2<i32>,a:u32,i:u32,j:u32,grain_share:f32)->vec2<f32>{
    if grain_share<=.01{return vec2<f32>(0.);}
    let l=traction(i);let r=traction(j);let area=surface_area(p,a)/grain_share;
    let friction=area*.5*vec2<f32>(l.force[a]+r.force[a],l.drag+r.drag);
    // The flow solve in f32 takes a drag up to about 1e10 against the
    // water's; the held force shrinks with it and drives no faster slip.
    if friction.y>1e10{return friction*(1e10/friction.y);}
    return friction;
}
// Grains that the plates held still at the end of the last update.
fn held_still(i:u32)->bool{return geometry[HOLD_ROW*cfg.n+i].z==0.;}
// Ends the update: HOLD_ROW = (the traction held after this update's slip,
// 1 if sliding, the speed, which tells a slowing layer).
@compute @workgroup_size(128) fn hold_friction(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let c=cells[i];let state=geometry[HOLD_ROW*cfg.n+i];
    let limit=friction_limit(i,c,select(KINETIC_FRICTION,STATIC_FRICTION,state.z==0.));
    let speed=grain_speed(c);let size=length(speed);
    var held=vec2<f32>(0.);var sliding=size>SLIP/cfg.dt;
    if limit>0.{
        let trial=limited(state.xy,limit)-limit*cfg.dt/SLIP*speed;held=limited(trial,limit);
        if state.z==0.{sliding=length(trial)>limit&&size>CREEP;}
        else{sliding=size>SLIP/cfg.dt||size>state.w;}
    }
    if void_speed(i)>CREEP{sliding=true;}
    geometry[HOLD_ROW*cfg.n+i]=vec4<f32>(held,select(0.,1.,sliding),size);
}
fn gas_narrowed(q:vec2<i32>)->bool{
    return cells[at(q)].volume.gas>.05||cells[at(q+axis(1u))].volume.gas>.05;
}
// The width (m) of the slot around a face crossed vertically: the wet
// apertures along its row to the first face narrowed by gas, or the tank's
// wall, on each side within 5 mm; -1 with open water or a bed beside it.
fn slot_width(f:u32)->f32{
    let p=point(f/2u);let reach=i32(ceil(.005/cfg.h));
    var near=false;
    for(var k=-reach;k<=reach;k++){
        let q=p+vec2<i32>(k,0);
        if valid(q)&&valid(q+axis(1u))&&gas_narrowed(q){near=true;break;}
    }
    if !near{return -1.;}
    var width=free_wet_aperture(f);var sides=0u;
    if width<.999&&!gas_narrowed(p){return -1.;}
    for(var side=0;side<2;side++){
        let step=select(-1,1,side==1);
        for(var k=1;k<=reach;k++){
            let q=p+vec2<i32>(step*k,0);
            if !valid(q)||!valid(q+axis(1u)){sides++;break;}
            let open=free_wet_aperture(2u*at(q)+1u);
            if open<.999{
                if !gas_narrowed(q){return -1.;}
                width+=open;sides++;break;
            }
            width+=open;
        }
    }
    if sides<2u{return -1.;}
    return width*cfg.h;
}
// Dense sand leaves a joint between bubbles no faster than a submerged
// hopper discharges (Wilson et al. 2014): grain volume
// C phi v_t (W - 2.4 d)(b - 2.4 d) per species through a slot W wide in the
// gap b, C = 0.6, v_t one grain's terminal speed; grains jam below 2.4 d.
// Each species takes the drag per grain volume (rho_s - rho_w) g/u that
// balances its submerged weight at u = C v_t (W - 2.4 d)(b - 2.4 d)/(W b).
// The sand above counts as dense from a wet packing of 0.4, fully from 0.55.
fn joint_drag(f:u32)->vec2<f32>{
    if f%2u==0u{return vec2<f32>(0.);}
    let q=point(f/2u)+axis(1u);if !valid(q){return vec2<f32>(0.);}
    let above=cells[at(q)];let wet=free_wet(above);let solid=grains(above).x+grains(above).y;
    if wet<=0.{return vec2<f32>(0.);}
    let dense=clamp((solid/wet-.4)/.15,0.,1.);if dense<=0.{return vec2<f32>(0.);}
    let slot=slot_width(f);if slot<=0.{return vec2<f32>(0.);}
    let d=GRAIN_DIAMETER;
    let speed=.6*vec2<f32>(white_speed(white_powder(above)),TERMINAL_SPEED.y)*max(vec2<f32>(slot)-2.4*d,vec2<f32>(0.))*(PLATE_GAP-2.4*d)/(slot*PLATE_GAP);
    // (rho_s - rho_w) g over a speed proportional to the settling speed is a
    // drag coefficient, free of gravity: both at the standard gravity.
    return min(dense*(RHO.yz-RHO.x)*STANDARD_GRAVITY/max(speed,vec2<f32>(1e-12)),vec2<f32>(1e10));
}
// Rebuilds each face before a flow solve: composition, closure mode, the
// unconstrained velocities u* = M f and the responses to a common pressure
// force and to a force on the grains.
@compute @workgroup_size(128) fn predict(@builtin(global_invocation_id) id:vec3<u32>) {
    let f=id.x;if f>=2u*cfg.n{return;}let i=f/2u;let a=f%2u;let p=point(i);let other=p+axis(a);
    var out:Face;out.displacement=faces[f].displacement;
    out.previous_displacement=faces[f].previous_displacement;out.coarse_momentum.z=free_wet_aperture(f);
    if !valid(other)||free_area(f)<=0.{out.support.w=1.;faces[f]=out;return;}let j=at(other);let l=cells[i];let r=cells[j];
    // The transport's inventories carry signed roundoff: clamped at zero.
    let total=max(fractions(l)+fractions(r),vec3<f32>(0.));
    let amount=total.x+total.y+total.z;
    if amount<=0.{faces[f]=out;return;}let c=total/amount;
    var present=select(vec3<f32>(0.),vec3<f32>(1.),c>vec3<f32>(0.));
    let friction=face_friction(p,a,i,j,c.y+c.z);let plate=vec2<f32>(friction.y)+joint_drag(f);
    let film=film_narrowness(out.coarse_momentum.z);
    let wl=max(fractions(l),vec3<f32>(0.));let wr=max(fractions(r),vec3<f32>(0.));
    let sl=wl.x+wl.y+wl.z;let sr=wr.x+wr.y+wr.z;
    let pl=select(0.,(wl.y+wl.z)/max(sl,1e-30),sl>0.);let pr=select(0.,(wr.y+wr.z)/max(sr,1e-30),sr>0.);
    var lookup=c;
    // Switched off: the table is read at the face's own composition; at the
    // packed side's (intrinsic_resistance_lookup) the fine grid breaks.
    if false&&max(pl,pr)>=CRITICAL_PACKING&&min(pl,pr)<CRITICAL_PACKING{lookup=select(wl/sl,wr/sr,pr>pl);}
    let powder=face_powder(l,r);
    let inverse_contact=grain_contact_inverse(f,c);var m=response_between(c,lookup,powder,inverse_contact,plate,film);
    // Faces that no grain crosses lock their phases to the common flow (a
    // plug's side on a bubble, a bubble's inside on a packing, the top and
    // bottom of a film between bubbles): free to slip, grains and water would
    // counter-flow there with a momentum the transport never carries away.
    var mode=grain_face_mode(f,c);if (a==0u&&plug_wall(i,j))||bubble_on_packing(f)||(a==1u&&(raft_film(i)||raft_film(j))){mode=2.;}out.water_momentum.z=mode;
    var old=vec3<f32>(0.);for(var phase=0u;phase<3u;phase++){old[phase]=face_history(p,a,phase);}
    // Inertia, and the weight relative to the water's. The water's own weight
    // is a gradient the pressure takes whole; the stream solve weighs a force
    // over the mobile dual volume and reads it over the free area, which
    // differ on a face touching a bed cell, so carried whole the weight would
    // curl over every step of the deposit and drive the water up its slopes.
    var force=RHO*present*old/cfg.dt-(RHO-vec3<f32>(RHO.x))*present*select(-cfg.lateral,cfg.gravity,a==1u);let lap=viscous_laplacian(p,a,0u);
    force+=vec3<f32>(9.4000000000000000e-4*present.x*lap/(cfg.h*cfg.h),present.yz*friction.x);
    var u=m*force;var rp=m*present;var rg=m*vec3<f32>(0.,present.y,present.z);
    var relative_mobility=0.;
    if mode==2. {
        let scalar=common_resistance_lookup(c,lookup,powder,plate,film)+dot(RHO,c)/cfg.dt;
        u=vec3<f32>(dot(c,force)/scalar);rp=vec3<f32>(1./scalar);rg=rp*(c.y+c.z);
    }else if mode==1. {
        let r=coarse_constraint(c,inverse_contact,powder,plate,film);let difference=force.y-force.x;
        let mean_velocity=(dot(c,force)-r.pair*r.coupling*difference)*r.common_inverse;
        let relative=difference*r.relative_inverse-r.coupling*mean_velocity;
        u=vec3<f32>(mean_velocity)+r.direction*relative;
        rp=(vec3<f32>(1.)-r.direction*r.coupling)*r.common_inverse;
        let grain_common=(c.y+c.z-r.pair*r.coupling)*r.common_inverse;
        rg=vec3<f32>(grain_common)+r.direction*(r.relative_inverse-r.coupling*grain_common);
        relative_mobility=(r.pair/(c.y+c.z))*r.relative_inverse;
        out.water_momentum.x=difference*r.relative_inverse;
        out.water_momentum.y=r.coupling;out.water_momentum.w=r.relative_inverse;
        out.response.w=r.common_inverse;out.velocity.w=mean_velocity;
    }else if c.x>0.&&c.y+c.z>0. {
        let effective=rg-rp*(dot(c,rg)/dot(c,rp));
        relative_mobility=max(dot(c.yz/(c.y+c.z),effective.yz),0.);
        if c.x<c.y+c.z {
            // The same Schur block as c_w/c_s times the liquid's conditional
            // self response (C*M is symmetric): on the dry side this form
            // keeps the small c_w factor exactly.
            let conditional=m[0].x-c.x*rp.x*(rp.x/dot(c,rp));
            relative_mobility=(c.x/(c.y+c.z))*max(0.,conditional);
        }
    }
    out.composition=vec4<f32>(c,.5*amount);
    // A locked face keeps relative_mobility exactly zero: a rounding-sized
    // positive value there becomes a huge pressure in the return.
    out.response=vec4<f32>(rp,out.response.w);out.grain_response=vec4<f32>(rg,relative_mobility);
    out.velocity=vec4<f32>(u,out.velocity.w);
    if mode!=1.{out.response.w=dot(c,rp);out.velocity.w=dot(c,u);}
    faces[f]=out;
}
fn preferred(f:u32,shear:vec3<f32>)->vec3<f32> {
    let i=f/2u;let a=f%2u;let p=point(i);let q=p+axis(a);if !valid(q){return vec3<f32>(0.);}
    let face=faces[f];let solid=face.composition.w*(face.composition.y+face.composition.z);
    var u=face.velocity.xyz;
    if solid>0. {
        let stress=relative_pressure_force(f);
        // Zero stress has exactly zero response: no zero over a tiny amount.
        if stress!=0.{u+=bulk_velocity_response(f,stress);}
        u+=shear;
    }
    return u;
}
// The phases' velocities relative to the common velocity `flow`, formed
// through antisymmetric phase pairs: their weighted sum is zero exactly, and
// a trace of liquid is not the difference of two nearly equal velocities.
fn relative_at_velocity(f:u32,flow:f32)->vec3<f32> {
    let face=faces[f];let c=face.composition.xyz;
    if phase_locked(f){return vec3<f32>(0.);}
    if coarse_locked(f){
        let mobile=c.x+c.y;let relative=face.volume.w-face.water_momentum.y*flow;
        return vec3<f32>(-c.y/mobile,c.x/mobile,0.)*relative;
    }
    if face.response.w<=0.{return vec3<f32>(0.);}
    let v=face.volume.xyz-vec3<f32>(flow);let r=face.response.xyz;
    let fine_water=(r.x*v.y-r.y*v.x)/face.response.w;
    let coarse_water=(r.x*v.z-r.z*v.x)/face.response.w;
    let fine_coarse=(r.z*v.y-r.y*v.z)/face.response.w;
    return vec3<f32>(-c.y*fine_water-c.z*coarse_water,
        c.x*fine_water+c.z*fine_coarse,c.x*coarse_water-c.y*fine_coarse);
}
fn corrected_relative(f:u32)->vec3<f32>{return relative_at_velocity(f,wet_velocity(f));}
// The common velocity a return reads: the solved flow's, or for the second
// return (`predictor`) the first projection's, kept in FACE_PREDICTION_ROW.
fn contact_velocity(f:u32,predictor:bool)->f32 {
    if predictor{return scratch[FACE_PREDICTION_ROW*cfg.n+f].x;}
    return wet_velocity(f);
}
fn grain_sample(f:u32,predictor:bool)->vec3<f32>{
    let flow=contact_velocity(f,predictor);return vec3<f32>(flow)+relative_at_velocity(f,flow);
}

// A wet part whose grains are caged (FREEZING_PACKING or above): its pores
// are far below a bubble's size, which must push such sand aside.
fn packed_wet(c:Cell)->bool{
    let wet=free_wet(c);let g=grains(c);return wet>0.&&g.x+g.y>=FREEZING_PACKING*wet;
}
// A bubble's inside and a packing (a cell at least half wet sand, caged)
// share a face that nothing crosses: gas cannot enter the pores, grains
// cannot enter the gas. The face moves only with the packing. While the
// plates hold the packing still the flow solve closes it (entry_resistance),
// or the raft's gas would creep into the slab's base by the transport's
// upwind share; while the packing slides the bubble follows its sand, as a
// bubble narrower than b/(mu_s K), about 5.5 mm, pushes the slab's base
// aside into a pocket. A column's cells keep their own rules (plug_wall).
// Returns the packing's cell of such a face, cfg.n for any other face.
fn bubble_packing(f:u32)->u32{
    let i=f/2u;let j=at(point(i)+axis(f%2u));
    if column_cell(i)||column_cell(j){return cfg.n;}
    let l=cells[i];let r=cells[j];
    if bubble_inside(l)&&free_wet(r)>=.5&&packed_wet(r){return j;}
    if bubble_inside(r)&&free_wet(l)>=.5&&packed_wet(l){return i;}
    return cfg.n;
}
fn bubble_on_packing(f:u32)->bool{return bubble_packing(f)<cfg.n;}
// Gas over half the cell and less than a grain layer of sand: a cell whose
// top the slab's base dips into meets the packing above sand to sand.
fn bubble_inside(c:Cell)->bool{
    let g=grains(c);return c.volume.gas>.5&&(g.x+g.y)*cfg.h<GRAIN_LAYER;
}
// The weight the pores' entry pressure adds to a face in the stream solve.
// Gas cannot enter wet packed sand: the meniscus would need about
// 2 sigma/r_throat, kilopascals, against tens of pascals of load. A bubble
// advances into caged sand only as fast as that sand moves away: the common
// flow that would carry gas into it (the gas moved that way last update)
// takes the sand's plate friction.
fn entry_resistance(f:u32)->f32{
    let i=f/2u;let p=point(i);let a=f%2u;let j=at(p+axis(a));
    // Gliding, the gas enters no pores: it moves along the sand.
    if glides(f){return 0.;}
    let packing=bubble_packing(f);
    if packing<cfg.n{return select(0.,1e10,held_still(packing));}
    // A plug's side against a bubble is closed to the mixture, as far as the
    // solve holds a face closed. Not its ends: the plug slides through the
    // chain's faces, which the transport closes to gas alone; closed too,
    // they would leave plug and body no outlet while the exit passes sand.
    if a==0u&&plug_wall(i,j){return 1e10;}
    let l=cells[i];let r=cells[j];let gas=face_history(p,a,3u);
    var receiver=i;
    if gas>0.&&l.volume.gas>r.volume.gas{receiver=j;}
    else if gas<0.&&r.volume.gas>l.volume.gas{receiver=i;}
    else{return 0.;}
    if !packed_wet(cells[receiver]){return 0.;}
    let face=faces[f];let solid=face.composition.y+face.composition.z;
    let friction=face_friction(p,a,i,j,solid);
    return face.composition.w*solid*friction.y;
}
// The stream solve's input per face: coarse_momentum.x the weight of its
// common velocity over the free area squared (times dt/rho_w), .y the
// current it prefers, .w the bound of the shear response.
@compute @workgroup_size(128) fn prepare_stream(@builtin(global_invocation_id) id:vec3<u32>) {
    let f=id.x;if f>=2u*cfg.n{return;}let face=faces[f];let p=point(f/2u);let a=f%2u;
    if !valid(p+axis(a)){faces[f].coarse_momentum.x=0.;faces[f].coarse_momentum.y=0.;return;}
    var weight=0.;if face.response.w>0.{weight=face.composition.w/face.response.w;}
    // face.volume keeps the preferred velocities until project_faces sets
    // the final ones: no cell repeats the shear scatter for its faces.
    let shear=shear_force(f);let load=grain_load(f,weight,shear);let u=load.xyz;faces[f].volume=vec4<f32>(u,0.);
    if coarse_locked(f){
        let solid=face.composition.w*(face.composition.y+face.composition.z);
        let normal=response_product(relative_pressure_force(f),face.water_momentum.w,solid);
        // w holds the unprojected relative drive until project_faces.
        faces[f].volume.w=face.water_momentum.x+normal+shear.relative;
    }
    faces[f].coarse_momentum.w=shear.bound;
    var force=load.w;let gas_fraction=face_gas(f);var gas_weight=0.;var gas_force=0.;
    if gas_fraction>0. {
        // Prototype A: Lagrangian closed-curve bubbles (`foam_step`) already solve
        // buoyancy, Young-Laplace surface tension, and plate meniscus drag. Here
        // the gas acts as an immersed moving body at its true displacement velocity
        // `u_gas`, with Brinkman impedance inside the bubble so water flows cleanly
        // around stationary bubbles instead of jetting through them (while leaving
        // open wet throats beside bubbles unobstructed).
        let u_gas=face_history(p,a,3u);
        let gas_cover=max(0.,1.-face.coarse_momentum.z);
        let block=smoothstep(.45,.85,max(gas_fraction,gas_cover));
        let impedance=mix(RHO.x/cfg.dt,2e7,block);
        gas_weight=gas_fraction*impedance;
        gas_force=gas_weight*u_gas;
    }
    // On a gliding face the sand strip stands and the gas strip moves at the
    // face's mean velocity times glide_ratio: seen by the mean, its weight
    // is the gas's times that ratio squared and its force the gas's times it.
    if glides(f){let ratio=glide_ratio(f);weight=gas_weight*ratio*ratio;force=gas_force*ratio;}
    else{weight+=gas_weight;force+=gas_force;}
    weight+=boundary_shear(f)+entry_resistance(f);
    let area=free_area(f);
    if area<=0.{faces[f].coarse_momentum.x=0.;faces[f].coarse_momentum.y=0.;return;}
    faces[f].coarse_momentum.x=(weight/area)/area*cfg.dt/997.;
    faces[f].coarse_momentum.y=select(0.,area*(force/max(weight,1e-30)),weight>0.);
}

@compute @workgroup_size(128) fn prepare_grain_velocity(@builtin(global_invocation_id) id:vec3<u32>){prepare_velocity(id.x,false);}
@compute @workgroup_size(128) fn prepare_contact_velocity(@builtin(global_invocation_id) id:vec3<u32>){prepare_velocity(id.x,true);}
// A cell's grain velocity per axis and the bound of its shear response: the
// means over its faces with the weights of prepare_grain_stress.
fn prepare_velocity(i:u32,predictor:bool) {
    if i>=cfg.n{return;}let p=point(i);var sum=vec2<f32>(0.);var bound=vec2<f32>(0.);let weights=grain_work(i).weights;
    for(var a=0u;a<2u;a++){for(var d=-1;d<=1;d++){for(var side=0u;side<2u;side++){
        let base=p+d*axis(1u-a)-select(axis(a),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;let face=faces[f];let g=traced_grains(face);let solid=g.x+g.y;
        if solid<=0.{continue;}
        sum[a]+=transverse_weight(d)*face.coarse_momentum.z*face.composition.w*dot(g,grain_sample(f,predictor).yz);
        if weights[a]>0.{
            let weight=transverse_weight(d)*face.coarse_momentum.z*face.composition.w*solid/weights[a];
            bound[a]+=weight*face.coarse_momentum.w;
        }
    }}}
    for(var a=0u;a<2u;a++){if weights[a]>0.{sum[a]/=weights[a];}}
    put_grain_velocity(i,sum);
    put_grain_velocity_bound(i,bound);
}
fn grain_velocity(p:vec2<i32>)->vec2<f32> {
    return reflected(p,grain_work(at(inside(p))).velocity);
}
fn velocity_bound(p:vec2<i32>)->vec2<f32> {
    return grain_work(at(inside(p))).velocity_bound;
}
fn holds_grains(q:vec2<i32>)->bool {
    let w=grain_work(at(inside(q))).weights;return w.x+w.y>0.;
}
// The grain velocity at q for the strain rate of the cell at p. Where q
// holds no grains above a trace they end between p and q, and their velocity
// extends unchanged: a skeleton does not shear against a phase not there.
fn edge_velocity(q:vec2<i32>,p:vec2<i32>)->vec2<f32> {
    return select(grain_velocity(p),grain_velocity(q),holds_grains(q));
}
fn edge_bound(q:vec2<i32>,p:vec2<i32>)->vec2<f32> {
    return select(velocity_bound(p),velocity_bound(q),holds_grains(q));
}
@compute @workgroup_size(128) fn grain_constraint(@builtin(global_invocation_id) id:vec3<u32>){return_grains(id.x,false);}
@compute @workgroup_size(128) fn contact_constraint(@builtin(global_invocation_id) id:vec3<u32>){return_grains(id.x,true);}
// A cell's stress return at the solved flow (`predictor`: at the first
// projection's velocity, before the second solve), from the grains' relative
// dilation rate `div`, its response `diagonal` to the cell's pressure, the
// in-plane strain rate and the shear `compliance`.
fn return_grains(i:u32,predictor:bool) {
    if i>=cfg.n{return;}let c=cells[i];let solid=grains(c).x+grains(c).y;let p=point(i);
    // The first return also caches its local problem for the pressure sweeps.
    if !predictor{cache_pressure_couplings(i);}
    // Trace grains take no part: the return's couplings scale as one over
    // the grains, and at 1e-18 of a cell their products leave the float range.
    let wet=dense_volume(i);put_returned_suction(i,0.);if solid<TRACE_VOLUME||wet<=0.{put_returned_stress(i,vec4<f32>(0.));return;}
    var div=0.;var diagonal=0.;var du=mat2x2<f32>();
    for(var a=0u;a<2u;a++) {
        for(var side=0u;side<2u;side++) {
            let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
            let f=2u*at(base)+a;let face=faces[f];let sg=face.composition.yz;
            let amount=sg.x+sg.y;
            let geometric=face.coarse_momentum.z/wet;
            if amount>=TRACE_VOLUME{div+=select(-1.,1.,side==1u)*geometric*compression_speed(f,predictor)/cfg.h;}
            if amount>=TRACE_VOLUME {
                // Twice the unprojected edge diagonal bounds the conditional
                // and the common-coupled responses' graph; incompressibility
                // only lowers the mobility.
                let contrast=pressure_common_response(f);
                let mobility=2.*(face.grain_response.w+amount*face.response.w*contrast*contrast);
                let solid_face=face.composition.w*amount;
                if mobility>0.&&solid_face>0.{diagonal+=mobility*(phase_ratio(solid,solid_face))*(geometric*geometric)/(cfg.h*cfg.h);}
            }
        }
        du[a]=(edge_velocity(p+axis(a),p)-edge_velocity(p-axis(a),p))/(2.*cfg.h);
    }
    // The sand slips at the plates: the glass's friction (0.32 sliding, 0.4
    // held) is at or below the grains' own, so a layer moving past the
    // plates crosses the gap as a plug instead of shearing across it. The
    // yield and the dilatancy see the in-plane strain alone (the plates'
    // traction is the flow solve's): with the gap shear 2u/b in the yield,
    // sliding sand would hold 0.6 P on the plates.
    let strain=vec2<f32>(du[0].x-du[1].y,du[0].y+du[1].x);
    if grain_work(i).pressure_mode==0.{diagonal=0.;}
    var tau=vec2<f32>(c.stress.normal,c.stress.shear);
    // The compliance: the absolute-row bound of the adjoint shear response
    // per grain-volume stress, which the pressure projection can only lower.
    // The bound of the slip at the plates (2/b) stays beside the in-plane
    // modes': without it the trial step grows about six-fold.
    let x=edge_bound(p-axis(0u),p)+edge_bound(p+axis(0u),p);
    let y=edge_bound(p-axis(1u),p)+edge_bound(p+axis(1u),p);
    let local=GAP_SHEAR*velocity_bound(p);
    let compliance=max(max((x.x+y.y)/(2.*cfg.h),(x.y+y.x)/(2.*cfg.h)),max(local.x,local.y));
    if compliance>0.{tau+=solid*strain/compliance;}
    // Together where the cell has a shear compliance and a volume response
    // (or no dilation rate at all), else the pressure first: the same laws.
    let trial=scaled_norm(tau);var pressure=0.;var returned=0.;let mu=grain_friction(grains(c));
    if compliance>0.&&(diagonal>0.||div==0.){
        let state=joint_return(solid,wet,div,diagonal,compliance,c.stress.pressure,trial,mu);
        pressure=state.x;returned=state.y;
    }else{
        pressure=particle_pressure(solid,wet,scaled_norm(strain),div,diagonal,c.stress.pressure);
        returned=shear_return(trial,solid,wet,pressure,compliance,mu);
    }
    if !predictor{cache_pressure_problem(i,div,diagonal,compliance,tau,vec3<f32>(solid,wet,scaled_norm(strain)),pressure);}
    // The law's pressure above the load is unmet dilation demand: a skeleton
    // that shears and cannot dilate. It is pore-water suction, booked with
    // the Darcy suction of the dilation that does happen; only the load is
    // kept as contact pressure, or a creeping cell's demand would ratchet.
    let darcy=dilatancy_suction(i,div);
    if darcy>0.{returned=shear_return(trial,solid,wet,pressure+darcy,compliance,mu);}
    if pressure+darcy<=0.{put_returned_stress(i,vec4<f32>(0.));return;}
    if trial>0.{tau*=returned/trial;}
    let load=load_part(solid,div,diagonal,c.stress.pressure,pressure);
    let suction=(pressure-load)+darcy;
    put_returned_stress(i,vec4<f32>(load,tau,load));put_returned_suction(i,suction);
}
@compute @workgroup_size(128) fn copy_stress(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.n{return;}put_stress(i,returned_stress(i),returned_suction(i));
    remember_contact(i);
}
@compute @workgroup_size(128) fn cache_contact(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x<cfg.n{remember_contact(id.x);}
}
// The record of a stress solve from which transfer_contact projects the
// pressure after the geometry changes: it lasts into the next update.
fn remember_contact(i:u32){
    let p=point(i);var aperture=vec4<f32>(0.);
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);
        if valid(base)&&valid(base+axis(a)){aperture[2u*a+side]=faces[2u*at(base)+a].coarse_momentum.z;}
    }}
    geometry[SOLVE_APERTURE_ROW*cfg.n+i]=aperture;
    geometry[SOLVE_STATE_ROW*cfg.n+i]=vec4<f32>(cells[i].stress.pressure,free_wet(cells[i]),0.,0.);
}
