// Trial interspecies contact drag. Duan et al. (2020), eqs14,22:
// beta_i=6 epsilon eta phi_i (u_i-u_s)/d^2, eta=tau/gamma.
// epsilon=1.7 is their measured dry, equal-size dense-mixture value. Extending
// this form to saturated unequal grains with the volume-mean diameter is an
// explicit approximation; the existing fluid-mediated drag remains separate.
fn contact_grain_history(p:vec2<i32>)->vec2<f32> {
    let c=cells[at(inside(p))];let mobile=grains(c);let amount=mobile.x+mobile.y;
    if amount<=0.{return vec2<f32>(0.);}return reflected(p,(c.momentum[1]+c.momentum[2])/amount);
}
fn contact_shear_rate(p:vec2<i32>)->f32 {
    let dx=(contact_grain_history(p+axis(0u))-contact_grain_history(p-axis(0u)))/(2.*cfg.h);
    let dy=(contact_grain_history(p+axis(1u))-contact_grain_history(p-axis(1u)))/(2.*cfg.h);
    return scaled_norm(vec2<f32>(dx.x-dy.y,dx.y+dy.x));
}
// Negative: no added contact resistance. Zero: its quasistatic no-slip limit.
// Keeping the reciprocal avoids a large stiffness and a numerical rate floor.
fn grain_contact_inverse(f:u32,c:vec3<f32>)->f32 {
    if c.y<=0.||c.z<=0.{return -1.;}
    let p=point(f/2u);let q=p+axis(f%2u);let l=cells[at(p)];let r=cells[at(q)];
    let wet=free_wet(l)+free_wet(r);if wet<=0.{return -1.;}
    let pressure=(max(l.stress.pressure,0.)+max(r.stress.pressure,0.))/wet;
    if pressure<=0.{return -1.;}
    let gamma=.5*(contact_shear_rate(p)+contact_shear_rate(q));let solid=c.y+c.z;
    // Positive pressure and zero shear give the exact no-slip contact limit.
    // Return it before forming mu(I): a trace pressure times .005 can
    // underflow, turning the otherwise well-defined static limit into 0/0.
    if gamma==0.{return 0.;}
    let diameter=dot(c.yz,GRAIN_DIAMETER)/solid;
    let viscous=.00094*gamma;let denom=.005*pressure+viscous;
    let friction=grain_friction(c.yz)+select(0.,.38*viscous/max(denom,1e-30),viscous>0.&&denom>0.);
    // Contact part of the current granular shear law; do not count its fluid
    // stress term a second time as interparticle contact resistance.
    let fp=10.2*friction*pressure;
    if !(fp>0.){return 0.;}
    return (solid*diameter*diameter)*(gamma/fp);
}
// The predictor's response with the packed side's drag coefficients
// (material.wgsl, hydraulic_response_lookup); the contact step is the same.
fn response_between(c:vec3<f32>,lookup:vec3<f32>,powder:PowderShare,inverse_contact:f32,plate:vec2<f32>,film:f32)->mat3x3<f32> {
    let m=hydraulic_response_lookup(c,lookup,powder,plate,film);if inverse_contact<0.{return m;}
    // Intrinsic rank-one force: beta*(0,c_coarse,-c_fine)*(u_fine-u_coarse).
    // Sherman--Morrison leaves common translation unpenalized and preserves
    // the volume-weighted symmetric dissipation, including beta->infinity.
    let q=m*vec3<f32>(0.,c.z,-c.y);
    let row=vec3<f32>(m[0].y-m[0].z,m[1].y-m[1].z,m[2].y-m[2].z);
    let denominator=inverse_contact+q.y-q.z;
    if denominator<=0.{return m;}
    return m-mat3x3<f32>(q*row.x,q*row.y,q*row.z)*(1./denominator);
}
fn grain_translation_response(f:u32)->f32 {
    let face=faces[f];let solid=face.composition.y+face.composition.z;
    if solid<=0.||face.response.w<=0.{return 0.;}
    return dot(face.composition.yz/solid,face.response.yz)/face.response.w;
}
