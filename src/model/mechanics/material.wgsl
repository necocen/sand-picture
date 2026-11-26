// The material laws of the mechanics, without entry points: the gap's
// resistances from the material table and a face's reduced responses, read
// by predict, and the grains' pressure, dilation and shear laws of the
// stress returns.

// The gap's six factors for a composition c (water, fine, coarse), whose
// dissipation is c_w liquid (u_w - sum c_s entrained_s u_s)^2
// + sum c_s plate_s u_s^2 + c_f c_c exchange (u_f - u_c)^2.
struct GapResistance {liquid:f32,entrained:vec2<f32>,plate:vec2<f32>,exchange:f32}
// A water passage narrower than the plate gap, a lamella between bubbles,
// takes Poiseuille's drag of its narrower dimension, (b/w)^2 the gap's: the
// real lamella drains to a soap film, and with the gap's drag a cell-wide
// one would be an open channel through the raft, on which the slab could
// not hang. On the water alone: the sieve holds a narrow passage's grains,
// and the push of packed sand on a bubble through its faces must not weaken.
fn film_narrowness(aperture:f32)->f32{
    let width=aperture*cfg.h;
    if width<=0.||width>=PLATE_GAP{return 1.;}
    return min((PLATE_GAP/width)*(PLATE_GAP/width),1e4);
}
// The factors at the grain fractions cs, from the material table. Its
// POWDER_LEVELS planes run over the share of a white whose powder is all
// free: a plane holds the dilute drag of its share's mean settling speed,
// the packed drag of its share's Sauter mean, and their blend from
// FREEZING_PACKING to LOOSE_PACKING. Bound powder settles with its grit but
// packs with the rest, so the plane is read at the free share in a
// suspension, at the whole share from contact, between them by that blend.
// `plate` adds each species' plate drag per grain volume (face_friction,
// joint_drag); `film` multiplies the water's.
fn gap_resistance(cs:vec2<f32>,powder:PowderShare,plate:vec2<f32>,film:f32)->GapResistance {
    let solid=cs.x+cs.y;
    if solid<=0.{
        let t0=table[0];let t1=table[1];
        return GapResistance(t0.x*film,t0.yz,t1.xy+plate,t0.w);
    }
    let share=cs.x/solid;
    let last=f32(cfg.table_side-1u);
    let uv=clamp(vec2<f32>(sqrt(solid/.8),share)*last,vec2<f32>(0.),vec2<f32>(last));
    let lo=min(vec2<u32>(uv),vec2<u32>(cfg.table_side-2u));let t=uv-vec2<f32>(lo);
    let read=mix(powder.free,powder.total,smoothstep(FREEZING_PACKING,LOOSE_PACKING,solid));
    let w=clamp(read,0.,1.)*f32(POWDER_LEVELS-1u);let level=min(u32(w),POWDER_LEVELS-2u);let tl=w-f32(level);
    let plane=2u*cfg.table_side*cfg.table_side;let stride=2u*cfg.table_side;
    let j0=level*plane+2u*(lo.y*cfg.table_side+lo.x);let k0=j0+plane;
    let r0=mix(
        mix(mix(table[j0],table[j0+2u],t.x),mix(table[j0+stride],table[j0+stride+2u],t.x),t.y),
        mix(mix(table[k0],table[k0+2u],t.x),mix(table[k0+stride],table[k0+stride+2u],t.x),t.y),
        tl,
    );
    let j1=j0+1u;let k1=k0+1u;
    let r1=mix(
        mix(mix(table[j1],table[j1+2u],t.x),mix(table[j1+stride],table[j1+stride+2u],t.x),t.y),
        mix(mix(table[k1],table[k1+2u],t.x),mix(table[k1+stride],table[k1+stride+2u],t.x),t.y),
        tl,
    );
    return GapResistance(r0.x*film,r0.yz,r1.xy+plate,r0.w);
}
// The resistance of the phases moving at one velocity, which annuls the
// exchange exactly: no large interphase entries are subtracted.
fn common_resistance(c:vec3<f32>,powder:PowderShare,plate:vec2<f32>,film:f32)->f32 {
    let r=gap_resistance(c.yz,powder,plate,film);let slip=1.-dot(c.yz,r.entrained);
    return c.x*r.liquid*slip*slip+dot(c.yz,r.plate);
}
// With inertia: (R + diag(rho)/dt)^-1, an identity row for an absent phase.
// H = A + r.liquid * u * w^T where A is block-diagonal (water inertia scalar
// plus 2x2 grain plate+inertia+exchange block) and u*w^T is rank-1. Inverting
// by Cramer's rule cancels O(r.liquid^3) ~ 1e36 terms in f32 when a narrow
// bubble film (film_narrowness up to 1e4) meets packed powder, yielding a
// negative or zero determinant (NaN). Exact Sherman-Morrison inversion keeps
// every intermediate strictly positive and achieves full f32 precision.
fn hydraulic_response_lookup(c:vec3<f32>,lookup:vec3<f32>,powder:PowderShare,plate:vec2<f32>,film:f32)->mat3x3<f32> {
    let r=gap_resistance(lookup.yz,powder,plate,film);let mass=RHO/cfg.dt;
    let p0=select(0.,1.,c.x>0.);let p1=select(0.,1.,c.y>0.);let p2=select(0.,1.,c.z>0.);
    let a00_inv=select(1.,cfg.dt/RHO.x,c.x>0.);
    let d1=select(1.,r.plate.x+mass.y,c.y>0.);
    let d2=select(1.,r.plate.y+mass.z,c.z>0.);
    let k12=select(0.,c.z*r.exchange,c.y>0.);
    let k21=select(0.,c.y*r.exchange,c.z>0.);
    let sc=max(max(d1,d2),max(k12,k21));
    let sd1=d1/sc;let sd2=d2/sc;let sk12=k12/sc;let sk21=k21/sc;
    let inv_det_s=1./(sd1*sd2+sd1*sk21+sd2*sk12);
    let ag00=((sd2+sk21)*inv_det_s)/sc;let ag01=(sk12*inv_det_s)/sc;
    let ag10=(sk21*inv_det_s)/sc;let ag11=((sd1+sk12)*inv_det_s)/sc;
    if r.liquid<=0.||p0==0.{
        return mat3x3<f32>(vec3<f32>(a00_inv,0.,0.),vec3<f32>(0.,ag00,ag10),vec3<f32>(0.,ag01,ag11));
    }
    let u1=-p1*c.x*r.entrained.x;let u2=-p2*c.x*r.entrained.y;
    let w1=-p1*c.y*r.entrained.x;let w2=-p2*c.z*r.entrained.y;
    let q0=a00_inv;let q1=ag00*u1+ag01*u2;let q2=ag10*u1+ag11*u2;
    let z0=a00_inv;let z1=w1*ag00+w2*ag10;let z2=w1*ag01+w2*ag11;
    let sg=w1*q1+w2*q2;
    let inv_r=1./r.liquid;let denom=inv_r+(a00_inv+sg);let alpha=1./denom;
    let m00=a00_inv*((inv_r+sg)/denom);
    let m01=-alpha*q0*z1;let m02=-alpha*q0*z2;
    let m10=-alpha*q1*z0;let m11=ag00-alpha*q1*z1;let m12=ag01-alpha*q1*z2;
    let m20=-alpha*q2*z0;let m21=ag10-alpha*q2*z1;let m22=ag11-alpha*q2*z2;
    return mat3x3<f32>(vec3<f32>(m00,m10,m20),vec3<f32>(m01,m11,m21),vec3<f32>(m02,m12,m22));
}
fn common_resistance_lookup(c:vec3<f32>,lookup:vec3<f32>,powder:PowderShare,plate:vec2<f32>,film:f32)->f32 {
    let r=gap_resistance(lookup.yz,powder,plate,film);let slip=1.-dot(c.yz,r.entrained);
    return c.x*r.liquid*slip*slip+dot(c.yz,r.plate);
}

// A face whose coarse grains cannot pass: the phase dissipation restricted
// to u = U*1 + V*(-p,q,0), p=c_f/(c_w+c_f), q=c_w/(c_w+c_f); the coarse
// phase shares the common U. The reduced equations are
// B*U + A*L*V = c.force - pressure, L*U + D*V = force_f-force_w, with
// A=c_w*c_f/(c_w+c_f); the Schur complement is summed from positive
// dissipation terms.
struct CoarseConstraint {
    direction:vec3<f32>, pair:f32,
    common_inverse:f32, relative_inverse:f32, coupling:f32,
}
fn coarse_constraint(c:vec3<f32>,inverse_contact:f32,powder:PowderShare,plate:vec2<f32>,film:f32)->CoarseConstraint {
    let r=gap_resistance(c.yz,powder,plate,film);let mobile=c.x+c.y;
    let p=c.y/mobile;let q=c.x/mobile;let pair=c.x*p;
    let slip=1.-dot(c.yz,r.entrained);let z=1.+c.x*r.entrained.x;
    let base_d=p*r.liquid*z*z+q*(r.plate.x+c.z*r.exchange)+(p*RHO.x+q*RHO.y)/cfg.dt;
    var inverse_d=select(0.,1./base_d,base_d>0.);var contact_fraction=0.;
    if inverse_contact>=0.{
        let added=c.z*q;let denominator=base_d*inverse_contact+added;
        if denominator>0.{inverse_d=inverse_contact/denominator;contact_fraction=added/denominator;}
        else{inverse_d=0.;contact_fraction=1.;}
    }
    let l=-r.liquid*slip*z+r.plate.x+(RHO.y-RHO.x)/cfg.dt;
    let coupling=l*inverse_d;let direction=vec3<f32>(-p,q,0.);
    let v=vec3<f32>(1.)-direction*coupling;
    let fluid=v.x-dot(c.yz*r.entrained,v.yz);
    let difference=v.y-v.z;
    let schur=c.x*r.liquid*fluid*fluid+dot(c.yz*r.plate,v.yz*v.yz)
        +c.y*c.z*r.exchange*difference*difference+dot(c*RHO,v*v)/cfg.dt
        +pair*(l*inverse_d)*l*contact_fraction;
    return CoarseConstraint(direction,pair,select(0.,1./schur,schur>0.),inverse_d,coupling);
}

@group(0) @binding(8) var<storage,read_write> transport:array<Transport>;

fn mobile_volumes(c:Cell)->vec4<f32> {return vec4<f32>(c.volume.water,c.volume.fine,c.volume.coarse,c.volume.gas);}
fn put_mobile_volumes(i:u32,v:vec4<f32>) {cells[i].volume=PhaseVolumes(v.x,v.y,v.z,v.w);}

fn grain_stress(c:Cell)->vec4<f32> {return vec4<f32>(c.stress.pressure,c.stress.normal,c.stress.shear,c.stress.load);}
fn put_stress(i:u32,value:vec4<f32>,suction:f32) {cells[i].stress=GrainStress(value.x,value.y,value.z,value.w,suction);}
// The load part of a returned pressure: with the dilation at the old
// pressure removed (free_theta = theta - alpha*old), -free_theta/alpha just
// stops the compression the flow imposes, and that much pushes the
// neighbours. The pressure above it meets a sheared packing's dilatancy
// demand: pore-water suction, an effective stress that leaves their total
// stress alone. A cell without compliance carries its pressure whole.
fn load_part(solid:f32,theta:f32,alpha:f32,old_bulk:f32,pressure_bulk:f32)->f32 {
    if alpha<=1e-30||solid<=0.{return pressure_bulk;}
    let free_theta=theta-alpha*(old_bulk/solid);
    return clamp(-free_theta/alpha*solid,0.,pressure_bulk);
}

fn face_history(p:vec2<i32>,a:u32,phase:u32)->f32 {
    let q=p+axis(a);if !valid(p)||!valid(q){return 0.;}
    let l=cells[at(p)];let r=cells[at(q)];let vl=max(0.,mobile_volumes(l)[phase]);let vr=max(0.,mobile_volumes(r)[phase]);
    var momentum=0.;if vl>0.{momentum+=l.momentum[phase][a];}if vr>0.{momentum+=r.momentum[phase][a];}
    if vl+vr<=0.{return 0.;}return momentum/(vl+vr);
}

// Boyer's curve is measured under an imposed pressure and reaches a
// suspension free to expand (phi_eq -> 0) as it vanishes. A packed cell
// whose contact pressure happens to be zero (a free face, the foot of a
// column above a joint) still bears its own layer: the curve's pressure is
// floored at one grain layer's submerged weight, (rho_s - rho_w) g d =
// 1.8 Pa for the fine sand, so that shear alone dilates a packing to no
// looser than 0.55 at a gap shear of 0.4/s instead of fluidising it.
const DILATION_PRESSURE_FLOOR:f32=1.8;
// The equilibrium packing at the floor pressure for a shear rate.
fn equilibrium_floor(gamma:f32)->f32 {
    let root_pressure=sqrt(DILATION_PRESSURE_FLOOR);
    return CRITICAL_PACKING*root_pressure/(root_pressure+sqrt(9.4000000000000000e-4)*sqrt(gamma));
}
// A finite-strain Darcy-Reynolds law: the dilation (+) or compaction rate
// gamma (1 - phi_eq/phi)/strain, with Boyer's steady packing curve and
// Athani's strain scales (0.48 dilating, 0.28 compacting), not their exact
// fit. The packing itself changes only through the transport.
fn dilation_rate(phi:f32,gamma:f32,pressure_per_grain:f32)->f32 {
    if gamma<=0.{return 0.;}
    let root_pressure=sqrt(max(phi*pressure_per_grain,DILATION_PRESSURE_FLOOR));
    let equilibrium=CRITICAL_PACKING*root_pressure/(root_pressure+sqrt(9.4000000000000000e-4)*sqrt(gamma));
    let strain=select(.28,.48,phi>=equilibrium);
    return gamma*(1.-equilibrium/phi)/strain;
}
fn particle_pressure(solid:f32,wet:f32,gamma:f32,theta:f32,compliance:f32,old_bulk:f32)->f32 {
    // Viscous compaction yield: loose grains cannot carry static pressure
    // (Baumgarten and Kamrin 2019, eqs 31-32 with K4=0, on Boyer's curve).
    let phi=solid/wet;let old=old_bulk/solid;
    if phi<CRITICAL_PACKING {
        let gap=CRITICAL_PACKING-phi;let admissible=.00094*gamma*phi/(gap*gap);
        if compliance<=1e-30{return solid*min(old,admissible);}
        if theta-compliance*old+compliance*admissible<=0.{return solid*admissible;}
    }
    if compliance<=1e-30{return old_bulk;}
    let free_theta=theta-compliance*old;
    if free_theta>=dilation_rate(phi,gamma,0.){return 0.;}
    // The largest possible dilation, gamma/.48, bounds the monotone root
    // without an arbitrary stress or velocity ceiling.
    var lo=0.;var hi=min(max(0.,(gamma/.48-free_theta)/compliance),1e30);
    if gamma>0.{
        let strain=select(.28,.48,free_theta>=0.);
        let equilibrium_target=phi*(1.-strain*free_theta/gamma);
        if equilibrium_target>0.&&equilibrium_target<CRITICAL_PACKING{
            // Positive compliance increases dilation: its pressure is below
            // the exact zero-compliance inverse, in dilute material too.
            let gap=CRITICAL_PACKING-equilibrium_target;
            hi=min(hi,9.4000000000000000e-4*(gamma-strain*free_theta)*equilibrium_target/(gap*gap));
        }
    }
    for(var iteration=0u;iteration<12u;iteration++){
        let mid=.5*(lo+hi);
        if free_theta+compliance*mid<dilation_rate(phi,gamma,mid){lo=mid;}else{hi=mid;}
    }
    return solid*(.5*(lo+hi));
}

// Whole-cell particle shear in Baumgarten and Kamrin's phase partition. On
// Boyer's steady packing curve, adding the liquid stress
// eta0*(1+2.5*phi)*gamma recovers the measured total suspension stress.
// Pressure and dilatancy stay a separate, non-associative update.
fn particle_shear(solid:f32,wet:f32,pressure:f32,gamma:f32,mu:f32)->f32 {
    if pressure<=0.||solid<=0.||wet<=0.{return 0.;}
    let phi=solid/wet;let per_grain=pressure/solid;let viscous=9.4000000000000000e-4*gamma;
    let denom=.005*(pressure/wet)+viscous;
    let contact=mu+select(0.,.38*viscous/max(denom,1e-30),viscous>0.&&denom>0.);
    // The volume stays outside the constitutive arithmetic: merged, the two
    // square roots of bulk stresses can underflow in their product.
    let ratio=select(0.,viscous/max(per_grain,1e-30),viscous>0.&&per_grain>0.);
    return pressure*(contact+2.5*sqrt(phi)*sqrt(ratio));
}
// The radial return of the shear norm beyond the static limit mu P, under
// the compliance's absolute-row bound: one scalar root, no global sweeps.
fn shear_return(trial:f32,solid:f32,wet:f32,pressure:f32,compliance:f32,mu:f32)->f32 {
    if pressure<=0.||trial<=0.{return 0.;}
    let static_limit=mu*pressure;
    if trial<=static_limit{return trial;}
    if compliance<=1e-30{return static_limit;}
    var lo=static_limit;
    let free_gamma=compliance*((trial-static_limit)/solid);
    var hi=min(trial,particle_shear(solid,wet,pressure,free_gamma,mu));
    for(var iteration=0u;iteration<12u;iteration++) {
        let stress=.5*(lo+hi);let gamma=compliance*((trial-stress)/solid);
        if stress<particle_shear(solid,wet,pressure,gamma,mu){lo=stress;}else{hi=stress;}
    }
    return .5*(lo+hi);
}

// The same dilation and shear laws solved together, at one corrected shear
// rate. t = sqrt(J)/(1+sqrt(J)) bounds the scalar unknown; the radial return
// first eliminates the pressure and the shear rate (joint_state).
fn joint_state(phi:f32,trial:f32,beta:f32,t:f32,mu:f32)->vec3<f32> {
    let z=1.-t;let tt=t*t;let zz=z*z;let eta_beta=.00094*beta;
    let friction=mu*zz+.38*tt*zz/(.005*zz+tt)+2.5*phi*t*z;
    let denominator=eta_beta*friction+phi*tt;
    if !(denominator>0.&&denominator<1e34){
        if tt==0.&&mu>0.{return vec3<f32>(trial/mu,0.,trial);}
        return vec3<f32>(0.);
    }
    let pressure=trial*(eta_beta*zz/denominator);
    let gamma=(beta*trial)*(phi*tt/denominator);
    let stress=trial*(eta_beta*friction/denominator);
    return vec3<f32>(pressure,gamma,stress);
}
fn joint_return(solid:f32,wet:f32,theta:f32,alpha:f32,beta:f32,old_bulk:f32,trial_bulk:f32,mu:f32)->vec2<f32> {
    let phi=solid/wet;let trial=min(trial_bulk/solid,1e30);
    // No volume response: theta is exactly zero when all relative modes
    // are closed. The finite-flow root is phi_eq=phi below phi_m; above it
    // the minimum admissible static yield stress arrests the shear rate.
    if alpha<=1e-30&&theta==0.{
        if trial<=0.{return vec2<f32>(0.);}
        let t=max(0.,1.-phi/CRITICAL_PACKING);
        return solid*joint_state(phi,trial,beta,t,mu).xz;
    }
    let free_theta=theta-alpha*(old_bulk/solid);let free_gamma=beta*trial;
    if free_theta>=free_gamma*(1.-equilibrium_floor(free_gamma)/phi)/.48{return vec2<f32>(0.);}
    // K4=0 compaction yield uses the existing packing curve: t>=1-phi/phi_m.
    // At its endpoint Reynolds dilation vanishes; negative volume strain is
    // admissible plastic compaction, independent of the shear rate.
    var lo=max(0.,1.-phi/CRITICAL_PACKING);var hi=1.;
    if phi<CRITICAL_PACKING {
        let compacting=joint_state(phi,trial,beta,lo,mu);
        if free_theta+alpha*compacting.x<=0.{return solid*compacting.xz;}
    }else if alpha>1e-30&&free_theta+alpha*(trial/mu)<=0.{
        return vec2<f32>(solid*min(-free_theta/alpha,1e30),trial_bulk);
    }
    for(var iteration=0u;iteration<12u;iteration++){
        let t=.5*(lo+hi);let state=joint_state(phi,trial,beta,t,mu);
        let equilibrium=max(CRITICAL_PACKING*(1.-t),equilibrium_floor(state.y));let strain=select(.28,.48,phi>=equilibrium);
        let dilation=state.y*(1.-equilibrium/phi)/strain;
        if free_theta+alpha*state.x>dilation{lo=t;}else{hi=t;}
    }
    let state=joint_state(phi,trial,beta,.5*(lo+hi),mu);
    return solid*state.xz;
}
// The grains' mean velocity relative to the mixture on a face.
fn compression_speed(f:u32,predictor:bool)->f32 {
    let c=faces[f].composition.xyz;let solid=c.y+c.z;
    if phase_locked(f)||c.x<=0.||solid<TRACE_VOLUME||faces[f].composition.w*solid<TRACE_VOLUME{return 0.;}
    return -(c.x/solid)*relative_at_velocity(f,contact_velocity(f,predictor)).x;
}

fn deviatoric_stress(p:vec2<i32>)->vec2<f32> {
    let s=cells[at(inside(p))].stress;return vec2<f32>(s.normal,select(s.shear,-s.shear,any(inside(p)!=p)));
}
// The force per bulk volume of the grains' in-plane stress. The plates'
// friction is not in it: face_friction, part of the flow solve.
fn stress_force(p:vec2<i32>)->vec2<f32> {
    let l=deviatoric_stress(p-axis(0u));let r=deviatoric_stress(p+axis(0u));let b=deviatoric_stress(p-axis(1u));let t=deviatoric_stress(p+axis(1u));
    return vec2<f32>(r.x-l.x+t.y-b.y,r.y-l.y-t.x+b.x)/(2.*cfg.h);
}
fn grain_amount(p:vec2<i32>)->f32 {
    let c=cells[at(inside(p))];return grains(c).x+grains(c).y;
}
// Only a cell's own faces (d = 0) enter its grain velocity.
fn transverse_weight(d:i32)->f32 {return select(0.,1.,d==0);}
