// Per-cell gas plane intercepts and area fractions, gas layer indexing, and
// the wet-aperture pass (`wet_apertures`) that ends each update after
// `foam_raster`.
//
// Not this repository's own: the plane's intercept and area fraction
// (`intercept`, `rectangle_fraction`) are transcribed from Aphros,
// https://github.com/cselab/aphros at commit
// b60ce3da52c19935fa24c778f62f02141eaf7f80, Copyright (c) 2021 ETH Zurich,
// under the MIT licence. Its notice, ../APHROS_LICENSE.txt, goes wherever
// these routines go.

// The intercept at which n . x <= alpha covers `fraction` of the unit cell.
fn intercept(n:vec2<f32>,fraction:f32)->f32 {
    let a=min(abs(n.x),abs(n.y));let b=max(abs(n.x),abs(n.y));
    if b<=0.{return 0.;}
    let c=max(min(fraction,1.-fraction),0.);var t=0.;
    if a>0.&&c<a/(2.*b){t=sqrt(max(2.*a*b*c,0.));}else{t=b*c+.5*a;}
    return select(t-.5*(a+b),.5*(a+b)-t,fraction>.5);
}
fn rectangle_fraction(n:vec2<f32>,alpha:f32,lo:vec2<f32>,hi:vec2<f32>)->f32 {
    let width=abs(n*(hi-lo));let a=min(width.x,width.y);let b=max(width.x,width.y);
    if b<=0.{return 0.;}
    let t=alpha-dot(n,.5*(lo+hi))+.5*(a+b);
    if t<=0.{return 0.;}if t>=a+b{return 1.;}
    let q=min(t,a+b-t);var c=0.;let ab2=2.*a*b;
    if a>0.&&q<a&&ab2>0.{c=q*q/ab2;}else{c=(q-.5*a)/b;}
    return select(c,1.-c,t>.5*(a+b));
}

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

// An identity's layer in a cell is (amount, identity, 0, 0); its shape is its
// plane (n, alpha) written by `foam_raster`.
fn layer_index(i:u32,l:u32)->u32{return LAYER_ROW*cfg.n+GAS_SLOTS*i+l;}
fn shape_index(i:u32,l:u32)->u32{return SHAPE_ROW*cfg.n+GAS_SLOTS*i+l;}

fn label_slot(i:u32,label:f32)->i32 {
    if label>=1.&&label<=f32(GAS_SLOTS){return i32(label)-1;}return -1;
}
fn label_fraction(p:vec2<i32>,label:f32)->f32 {
    let i=at(clamp(p,vec2<i32>(0),vec2<i32>(i32(cfg.nx)-1,i32(cfg.ny)-1)));
    let l=label_slot(i,label);if l<0{return 0.;}return geometry[layer_index(i,u32(l))].x;
}
