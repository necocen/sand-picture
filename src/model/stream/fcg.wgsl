// Flexible conjugate gradients on the finest flow equation, with one V-cycle
// (zero start, the residual as right-hand side) as the preconditioner. The
// cycle alone leaves most of its error inside the bubbles, where the gas's
// small resistance makes the equation stiff. Vectors live in `fcg`: x, b, r,
// p, q, r_old, then per-workgroup partial sums and the scalars.
@group(0) @binding(8) var<storage,read_write> fcg:array<f32>;
fn vertices()->u32{return cfg.nx*cfg.ny;}
fn fx_(i:u32)->u32{return i;}
fn fb_(i:u32)->u32{return vertices()+i;}
fn fr_(i:u32)->u32{return 2u*vertices()+i;}
fn fp_(i:u32)->u32{return 3u*vertices()+i;}
fn fq_(i:u32)->u32{return 4u*vertices()+i;}
fn fo_(i:u32)->u32{return 5u*vertices()+i;}
fn partials()->u32{return 6u*vertices();}
fn scalars()->u32{return 6u*vertices()+2u*((vertices()+127u)/128u);}
// Scalars: 0 rz (z.r of the current iteration), 1 beta.
fn apply_row(i:u32,base:u32)->f32{
    let p=vec2<i32>(i32(i%cfg.nx),i32(i/cfg.nx));let last=vec2<i32>(i32(cfg.nx)-1,i32(cfg.ny)-1);
    let x0=u32(clamp(p.x-2,0,last.x));let x1=u32(clamp(p.x-1,0,last.x));let x2=u32(p.x);
    let x3=u32(clamp(p.x+1,0,last.x));let x4=u32(clamp(p.x+2,0,last.x));
    let y0=base+u32(clamp(p.y-2,0,last.y))*cfg.nx;let y1=base+u32(clamp(p.y-1,0,last.y))*cfg.nx;let y2=base+u32(p.y)*cfg.nx;
    let y3=base+u32(clamp(p.y+1,0,last.y))*cfg.nx;let y4=base+u32(clamp(p.y+2,0,last.y))*cfg.nx;
    var value=0.;
    value+=rows[i].a[0]*fcg[y0+x0];value+=rows[i].a[1]*fcg[y0+x1];value+=rows[i].a[2]*fcg[y0+x2];value+=rows[i].a[3]*fcg[y0+x3];value+=rows[i].a[4]*fcg[y0+x4];
    value+=rows[i].a[5]*fcg[y1+x0];value+=rows[i].a[6]*fcg[y1+x1];value+=rows[i].a[7]*fcg[y1+x2];value+=rows[i].a[8]*fcg[y1+x3];value+=rows[i].a[9]*fcg[y1+x4];
    value+=rows[i].a[10]*fcg[y2+x0];value+=rows[i].a[11]*fcg[y2+x1];value+=rows[i].a[12]*fcg[y2+x2];value+=rows[i].a[13]*fcg[y2+x3];value+=rows[i].a[14]*fcg[y2+x4];
    value+=rows[i].a[15]*fcg[y3+x0];value+=rows[i].a[16]*fcg[y3+x1];value+=rows[i].a[17]*fcg[y3+x2];value+=rows[i].a[18]*fcg[y3+x3];value+=rows[i].a[19]*fcg[y3+x4];
    value+=rows[i].a[20]*fcg[y4+x0];value+=rows[i].a[21]*fcg[y4+x1];value+=rows[i].a[22]*fcg[y4+x2];value+=rows[i].a[23]*fcg[y4+x3];value+=rows[i].a[24]*fcg[y4+x4];
    return value;
}
@compute @workgroup_size(128) fn fcg_start(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=vertices(){return;}
    fcg[fx_(i)]=solution[i];fcg[fb_(i)]=rows[i].rhs;fcg[fp_(i)]=0.;fcg[fo_(i)]=0.;
}
// The preconditioning cycle solves A z = r from zero on the finest rows: the
// residual is their right-hand side, the solution starts at zero. The
// kernels that produce r set this up for the next cycle (fcg_finish restores
// b and x after the last).
fn precondition(i:u32,r:f32){rows[i].rhs=r;solution[i]=0.;}
@compute @workgroup_size(128) fn fcg_residual(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=vertices(){return;}
    let r=fcg[fb_(i)]-apply_row(i,fx_(0u));fcg[fr_(i)]=r;precondition(i,r);
    if i==0u{let s=scalars();fcg[s]=0.;fcg[s+1u]=0.;}
}
var<workgroup> partial_a:array<f32,128>;
var<workgroup> partial_b:array<f32,128>;
fn reduce_pair(lane:u32,a:f32,b:f32)->vec2<f32>{
    partial_a[lane]=a;partial_b[lane]=b;workgroupBarrier();
    for(var k=64u;k>0u;k/=2u){if lane<k{partial_a[lane]+=partial_a[lane+k];partial_b[lane]+=partial_b[lane+k];}workgroupBarrier();}
    return vec2<f32>(partial_a[0],partial_b[0]);
}
// z (the preconditioned residual) is `solution` after the cycle.
@compute @workgroup_size(128) fn fcg_dots_z(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let i=id.x;var a=0.;var b=0.;
    if i<vertices(){let z=solution[i];a=z*fcg[fr_(i)];b=z*fcg[fo_(i)];}
    let sums=reduce_pair(lane,a,b);
    if lane==0u{fcg[partials()+2u*group.x]=sums.x;fcg[partials()+2u*group.x+1u]=sums.y;}
}
fn total_partials(lane:u32)->vec2<f32>{
    let groups=(vertices()+127u)/128u;var a=0.;var b=0.;
    for(var k=lane;k<groups;k+=128u){a+=fcg[partials()+2u*k];b+=fcg[partials()+2u*k+1u];}
    return reduce_pair(lane,a,b);
}
// Polak-Ribiere: beta = z.(r - r_old) / (z_old . r_old); zero on the first pass.
@compute @workgroup_size(128) fn fcg_beta(@builtin(local_invocation_index) lane:u32){
    let sums=total_partials(lane);
    if lane==0u{
        let s=scalars();let previous=fcg[s];var beta=0.;
        if previous>1e-30{beta=clamp((sums.x-sums.y)/previous,0.,1e6);}
        fcg[s]=sums.x;fcg[s+1u]=beta;
    }
}
@compute @workgroup_size(128) fn fcg_direction(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=vertices(){return;}
    fcg[fp_(i)]=solution[i]+fcg[scalars()+1u]*fcg[fp_(i)];
}
// q = A p, and the workgroup's part of p.q.
@compute @workgroup_size(128) fn fcg_matvec(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let i=id.x;var a=0.;
    if i<vertices(){let q=apply_row(i,fp_(0u));fcg[fq_(i)]=q;a=fcg[fp_(i)]*q;}
    let sums=reduce_pair(lane,a,0.);
    if lane==0u{fcg[partials()+2u*group.x]=sums.x;fcg[partials()+2u*group.x+1u]=0.;}
}
// alpha = z.r / p.q, summed by every workgroup alike from the same parts.
@compute @workgroup_size(128) fn fcg_update(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let sums=total_partials(lane);var alpha=0.;if sums.x>1e-30{alpha=clamp(fcg[scalars()]/sums.x,-1e6,1e6);}
    let i=id.x;if i>=vertices(){return;}
    fcg[fx_(i)]+=alpha*fcg[fp_(i)];fcg[fo_(i)]=fcg[fr_(i)];
    let r=fcg[fr_(i)]-alpha*fcg[fq_(i)];fcg[fr_(i)]=r;precondition(i,r);
}
@compute @workgroup_size(128) fn fcg_finish(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=vertices(){return;}solution[i]=fcg[fx_(i)];rows[i].rhs=fcg[fb_(i)];
}
