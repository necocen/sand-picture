// The gas as a volume of fluid with up to GAS_SLOTS identities a cell, whose
// films do not rupture (Karnakov et al. 2022, doi:10.1126/sciadv.abm0590)
// unless stacked (labels.wgsl): the planes' intercepts and areas, the gas
// rows and normals, and the entries of the stages that end each update.
//
// Not this repository's own: the MYC normal (label_normal) and the
// threshold-one parallelization, with the plane's intercept and area
// fraction (intercept, rectangle_fraction), are transcribed from Aphros,
// https://github.com/cselab/aphros at commit
// b60ce3da52c19935fa24c778f62f02141eaf7f80, Copyright (c) 2021 ETH Zurich,
// under the MIT licence. Its notice, ../APHROS_LICENSE.txt, goes wherever
// these routines go.

// The intercept at which n . x <= alpha covers `fraction` of the unit cell.
fn intercept(n:vec2<f32>,fraction:f32)->f32 {
    let a=min(abs(n.x),abs(n.y));let b=max(abs(n.x),abs(n.y));let c=min(fraction,1.-fraction);var t=0.;
    if a>0.&&c<a/(2.*b){t=sqrt(2.*a*b*c);}else{t=b*c+.5*a;}
    return select(t-.5*(a+b),.5*(a+b)-t,fraction>.5);
}
fn rectangle_fraction(n:vec2<f32>,alpha:f32,lo:vec2<f32>,hi:vec2<f32>)->f32 {
    let width=abs(n*(hi-lo));let a=min(width.x,width.y);let b=max(width.x,width.y);
    let t=alpha-dot(n,.5*(lo+hi))+.5*(a+b);
    if t<=0.{return 0.;}if t>=a+b{return 1.;}
    let q=min(t,a+b-t);var c=0.;
    if a>0.&&q<a{c=q*q/(2.*a*b);}else{c=(q-.5*a)/b;}
    return select(c,1.-c,t>.5*(a+b));
}
@compute @workgroup_size(128) fn reconstruct_interface(@builtin(global_invocation_id) id:vec3<u32>) { paired_reconstruct(id.x,true); }

fn label_liquid_aperture(f:u32)->f32 {
    return curved_passage(f,0.).total;
}
@compute @workgroup_size(128) fn wet_apertures(@builtin(global_invocation_id) id:vec3<u32>) {
    let f=id.x;if f>=2u*cfg.n{return;}
    faces[f].coarse_momentum.z=free_wet_aperture(f);
}

// The Euclidean norm, scaled before squaring against overflow and underflow.
fn scaled_norm(v:vec2<f32>)->f32 {
    let a=abs(v);let scale=max(a.x,a.y);
    if !(scale>0.&&scale<1e35){return 0.;}
    let q=v/scale;return scale*sqrt(dot(q,q));
}
// The ratio of two positive volumes with their binary exponents split off, so
// that a compiler reassociating it with a small mobility cannot underflow it.
fn phase_ratio(numerator:f32,denominator:f32)->f32 {
    if numerator<=0.||denominator<=0.{return 0.;}
    let n=frexp(numerator);let d=frexp(denominator);
    if d.fract==0.{return 0.;}
    return ldexp(n.fract/d.fract,clamp(n.exp-d.exp,-126,126));
}

// An identity's layer in a cell is (amount, identity, 0, an observation of
// the transport's); its shape is its plane (n, alpha) of contacts.wgsl.
fn layer_index(i:u32,l:u32)->u32{return LAYER_ROW*cfg.n+GAS_SLOTS*i+l;}
fn shape_index(i:u32,l:u32)->u32{return SHAPE_ROW*cfg.n+GAS_SLOTS*i+l;}

fn flux_index(i:u32,l:u32)->u32{return FLUX_ROW*cfg.n+GAS_SLOTS*i+l;}
fn label_slot(i:u32,label:f32)->i32 {
    if label>=1.&&label<=f32(GAS_SLOTS){return i32(label)-1;}return -1;
}
fn label_fraction(p:vec2<i32>,label:f32)->f32 {
    let i=at(clamp(p,vec2<i32>(0),vec2<i32>(i32(cfg.nx)-1,i32(cfg.ny)-1)));
    let l=label_slot(i,label);if l<0{return 0.;}return geometry[layer_index(i,u32(l))].x;
}
fn l1_normal(v:vec2<f32>)->vec2<f32> {
    let s=abs(v.x)+abs(v.y);if s==0.{return vec2<f32>(1.,0.);}return -v/s;
}
// The MYC normal of an identity in cell p: out of its gas, L1-normalised.
fn label_normal(p:vec2<i32>,label:f32)->vec2<f32> {
    var q:array<f32,9>;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){q[u32((y+1)*3+x+1)]=label_fraction(p+vec2<i32>(x,y),label);}}
    let young=l1_normal(vec2<f32>(-q[0]-2.*q[3]-q[6]+q[2]+2.*q[5]+q[8],-q[0]+q[6]-2.*q[1]+2.*q[7]-q[2]+q[8]));
    let a=l1_normal(vec2<f32>(select(-2.,2.,q[5]>q[3]),q[6]+q[7]+q[8]-q[0]-q[1]-q[2]));
    let b=l1_normal(vec2<f32>(q[2]+q[5]+q[8]-q[0]-q[3]-q[6],select(-2.,2.,q[7]>q[1])));
    let d=select(1u,0u,abs(a.x)>=abs(b.y));let candidate=select(b,a,d==0u);
    return select(young,candidate,abs(candidate[d])<abs(young[d]));
}
