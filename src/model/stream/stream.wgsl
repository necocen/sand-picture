// The flow solve, twice an update. The common volume flow is the curl of a
// streamfunction psi on the grid's vertices, the minimum of one quadratic:
// the faces' resistance to their flux plus the interfaces' implicit
// tension. This file holds its rows and the multigrid cycle that
// preconditions fcg.wgsl; with free_stream.wgsl they are one shader module.

// The dispatched level, the next coarser one and the grid's cells (fx, fy).
// `color`: a relax's colour; 1 for a cold assembly; coarse_cycle's levels.
struct Config {nx:u32,ny:u32,offset:u32,cx:u32,cy:u32,coarse:u32,fx:u32,fy:u32,h:f32,color:u32,capillary:f32,gas_slots:u32}
// A row's 5x5 stencil, signs included (slot 12 is the diagonal), and its
// right-hand side.
struct Row {a:array<f32,25>,rhs:f32}
struct Level {nx:u32,ny:u32,offset:u32,cx:u32,cy:u32,coarse:u32}
fn dispatched_level()->Level{return Level(cfg.nx,cfg.ny,cfg.offset,cfg.cx,cfg.cy,cfg.coarse);}
// Viscosity (Pa s) of the implicit damping inside bubbles (interface_tensor):
// a numerical choice, far above air's 1.8e-5 Pa s.
const GAS_VISCOSITY:f32=.02;
// The model's Face (shared.wgsl): the two layouts must match.
struct Face {composition:vec4<f32>,velocity:vec4<f32>,response:vec4<f32>,grain_response:vec4<f32>,volume:vec4<f32>,water_momentum:vec4<f32>,coarse_momentum:vec4<f32>,support:vec4<f32>,displacement:vec4<f32>,previous_displacement:vec4<f32>}
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> rows:array<Row>;
@group(0) @binding(2) var<storage,read_write> solution:array<f32>;
@group(0) @binding(3) var<storage,read_write> residual:array<f32>;
@group(0) @binding(4) var<storage,read_write> faces:array<Face>;
@group(0) @binding(5) var<storage,read_write> cells:array<f32>;
@group(0) @binding(6) var<storage,read_write> geometry:array<vec4<f32>>;
// Per finest vertex i: [i] = the tension's tensor (xx, xy, yy) times
// cfg.capillary, with w = 1 where the bed fixes psi, else minus the vertex's
// interpolation weight; then nine entries a vertex, its gradient_basis.
@group(0) @binding(7) var<storage,read_write> tensor:array<vec4<f32>>;
fn valid(p:vec2<i32>,nx:u32,ny:u32)->bool{return all(p>=vec2<i32>(0))&&p.x<i32(nx)&&p.y<i32(ny);}
fn at(p:vec2<i32>,nx:u32)->u32{return u32(p.y)*nx+u32(p.x);}
// psi is zero on the tank's rim and, on the finest level, at the vertices
// of a face that the bed closes.
fn boundary(p:vec2<i32>)->bool{return outside_boundary(p)||(cfg.offset==0u&&tensor[at(p,cfg.nx)].w>0.);}
fn slot(d:vec2<i32>)->u32{return u32((d.y+2)*5+d.x+2);}
// The face a psi edge crosses, as (weight, preferred flux in the sense of
// the edge's difference of psi): minus the upward flux for the edge from
// vertex (x, y) to (x+1, y) (a = 0), the flux toward +x for a = 1.
fn edge(x:u32,y:u32,a:u32)->vec2<f32> {
    if a==0u{if y==0u||y>=cfg.fy||x>=cfg.fx{return vec2<f32>(0.);}let f=faces[2u*((y-1u)*cfg.fx+x)+1u];return vec2<f32>(f.coarse_momentum.x,-f.coarse_momentum.y);}
    if x==0u||x>=cfg.fx||y>=cfg.fy{return vec2<f32>(0.);}return faces[2u*(y*cfg.fx+x-1u)].coarse_momentum.xy;
}
fn rhs(p:vec2<i32>)->f32 {
    let x=u32(p.x);let y=u32(p.y);let e=edge(x,y,0u);let w=edge(x-1u,y,0u);let n=edge(x,y,1u);let s=edge(x,y-1u,1u);
    return cfg.h*(w.x*w.y-e.x*e.y+s.x*s.y-n.x*n.y)+capillary_rhs(p);
}
// |grad(alpha)| (I - n n^T) as (xx, xy, yy). The gradient is scaled before
// it is squared, so that tiny amplitudes keep their linear response.
fn gradient_tensor(g:vec2<f32>)->vec3<f32> {
    let scale=max(abs(g.x),abs(g.y));if !(scale>0.&&scale<1e35){return vec3<f32>(0.);}
    let u=g/scale;
    return (scale/length(u))*vec3<f32>(u.y*u.y,-u.x*u.y,u.x*u.x);
}
// The gas of identity slot l in cell c.
fn corner_gas(c:u32,l:u32)->f32 {
    let amount=geometry[LAYER_ROW*cfg.fx*cfg.fy+cfg.gas_slots*c+l].x;
    return select(0.,amount,amount>0.);
}
// The relabelling's word of each cell (labels.wgsl, label_cell_word), from
// LABEL_CELL_WORDS on: if its bit 31 is set, the bits from gas_slots up name
// the identity slots that hold gas.
@group(0) @binding(11) var<storage,read_write> labels:array<u32>;
// The identity slots of cell c that hold gas: by its word (most cells hold
// none, and asking each slot of each would be most of this kernel's time),
// or asked one by one where the word does not say.
fn corner_slots(c:u32)->u32 {
    let every=(1u<<cfg.gas_slots)-1u;let word=labels[LABEL_CELL_WORDS+c];
    if (word>>31u)!=0u{return (word>>cfg.gas_slots)&every;}
    var slots=0u;
    for(var l=0u;l<cfg.gas_slots;l++){if corner_gas(c,l)>0.{slots|=1u<<l;}}
    return slots;
}
// The tension's tensor at the vertex of the four cells c, which hold gas in
// the slots `present`; `inside` is 0 for a cell beyond the tank's wall, which
// counts as water. The identities add in the order of their first
// appearance over the four cells (raster order, slots ascending): the order
// is part of the sum's bits.
fn vertex_tensor(c:vec4<u32>,present:vec4<u32>,inside:vec4<f32>)->vec3<f32> {
    var value=vec3<f32>(0.);var seen=0u;
    for(var corner=0u;corner<4u;corner++){
        var fresh=present[corner]&~seen;seen|=present[corner];
        loop{
            if fresh==0u{break;}
            let l=countTrailingZeros(fresh);fresh&=fresh-1u;
            let q=inside*vec4<f32>(corner_gas(c.x,l),corner_gas(c.y,l),corner_gas(c.z,l),corner_gas(c.w,l));
            // Opposite cells' differences are paired before the mean: an
            // axis-aligned plane keeps an exactly zero tangential gradient.
            let g=.5*vec2<f32>((q.y-q.x)+(q.w-q.z),(q.z-q.x)+(q.w-q.y));
            value+=gradient_tensor(g);
        }
    }
    return value;
}
@compute @workgroup_size(POINT_THREADS) fn interface_tensor(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.nx*cfg.ny{return;}let p=vec2<i32>(i32(i%cfg.nx),i32(i/cfg.nx));var value=vec3<f32>(0.);
    // The vertex's four cells. The soap solution wets the frame: a film of
    // water lies between a bubble and a wall, and no line of contact holds
    // the bubble's surface there. So on the tank's rim the cells beyond the
    // wall count as water, which closes the surface of a bubble at a wall
    // along the wall. With the tensor at interior vertices only, the surface
    // would end on the wall at a right angle, held at its two ends by the
    // tension 2 sigma: enough to keep a bubble of less than
    // 2 sigma / (rho g), 7 mm2, on the floor against its buoyancy.
    let last=vec2<i32>(i32(cfg.fx)-1,i32(cfg.fy)-1);
    let low=clamp(p-vec2<i32>(1),vec2<i32>(0),last);let high=clamp(p,vec2<i32>(0),last);
    let c=vec4<u32>(at(low,cfg.fx),at(vec2<i32>(high.x,low.y),cfg.fx),
        at(vec2<i32>(low.x,high.y),cfg.fx),at(high,cfg.fx));
    let left=p.x>0;let right=p.x<=last.x;let down=p.y>0;let up=p.y<=last.y;
    let inside=vec4<f32>(select(0.,1.,left&&down),select(0.,1.,right&&down),select(0.,1.,left&&up),select(0.,1.,right&&up));
    var present=vec4<u32>(0u);
    if inside.x>0.{present.x=corner_slots(c.x);}
    if inside.y>0.{present.y=corner_slots(c.y);}
    if inside.z>0.{present.z=corner_slots(c.z);}
    if inside.w>0.{present.w=corner_slots(c.w);}
    // Surface tension is solved semi-implicitly on the 2D closed
    // curves (`foam_step`), so the flow solver drops the stiff VOF capillary
    // tensor and keeps only interior gas viscosity damping and bed boundaries.
    value=vec3<f32>(0.);
    // Inside a bubble an isotropic term penalises the velocity gradient: the
    // viscosity GAS_VISCOSITY acting implicitly, weighted by the least gas
    // fraction of the vertex's four cells and not only where all four are
    // gas (a puddle two cells thick has no such vertex). The discrete
    // tension leaves a curl along interfaces that the gas, resisting almost
    // nothing, would close as a circulation of metres per second; at a
    // sharp interface the gas pressure is uniform and the tension is
    // balanced on the water side. The weight mu h/(dt sigma) is the same
    // viscosity on every grid; it does not slow a bubble's rise and does no
    // work on a divergence-free flow's right-hand side.
    if !outside_boundary(p){
        var least=1.;
        for(var corner=0u;corner<4u;corner++){
            var gas=0.;var slots=present[corner];
            loop{
                if slots==0u{break;}
                let l=countTrailingZeros(slots);slots&=slots-1u;
                gas+=corner_gas(c[corner],l);
            }
            least=min(least,gas);
        }
        if least>0.{let weight=GAS_VISCOSITY*cfg.h/(CAPILLARY_DT*SURFACE_TENSION)*select(least,1.,least>=.99);value+=vec3<f32>(weight,0.,weight);}
    }
    tensor[i]=vec4<f32>(cfg.capillary*value,select(0.,1.,body_boundary(p)));
    if any(value!=vec3<f32>(0.)){
        for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
            let d=vec2<i32>(x,y);tensor[gradient_index(p,d)]=gradient_basis(p,d);
        }}
    }
}
// The finest rows: the faces' weights on the differences of psi across them
// plus, from each of the nine vertices around p, the tension's Gram term in
// the gradient basis of the right-hand side. A fixed vertex is an identity
// row. A cold assembly (colour 1) also zeroes psi: once the bed has moved,
// the old circulation can have a very large energy in the new operator.
@compute @workgroup_size(POINT_THREADS) fn assemble(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.nx*cfg.ny{return;}let p=vec2<i32>(i32(i%cfg.nx),i32(i/cfg.nx));var row:Row;
    if boundary(p){row.a[12]=1.;rows[i]=row;solution[i]=0.;anchors[i]=FIVE_POINT_FLAG;return;}
    let x=u32(p.x);let y=u32(p.y);let e=edge(x,y,0u);let w=edge(x-1u,y,0u);let n=edge(x,y,1u);let s=edge(x,y-1u,1u);
    row.a[12]=e.x+w.x+n.x+s.x;row.rhs=rhs(p);
    if !boundary(p-vec2<i32>(1,0)){row.a[11]=-w.x;}if !boundary(p+vec2<i32>(1,0)){row.a[13]=-e.x;}
    if !boundary(p-vec2<i32>(0,1)){row.a[7]=-s.x;}if !boundary(p+vec2<i32>(0,1)){row.a[17]=-n.x;}
    var five_point=true;
    for(var cy=-1;cy<=1;cy++){for(var cx=-1;cx<=1;cx++){
        let offset=vec2<i32>(cx,cy);let center=p+offset;if !valid(center,cfg.nx,cfg.ny){continue;}
        let t=tensor[at(center,cfg.nx)].xyz;if all(t==vec3<f32>(0.)){continue;}
        five_point=false;let a=gradient(center,-offset);
        for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
            let d=vec2<i32>(dx,dy);let q=center+d;if !valid(q,cfg.nx,cfg.ny)||boundary(q){continue;}
            row.a[slot(offset+d)]+=gram(a,gradient(center,d),t);
        }}
    }}
    rows[i]=row;
    anchors[i]=select(0u,FIVE_POINT_FLAG,five_point);
    if cfg.color==1u{solution[i]=0.;}
}
// Rows hold exact zeros for neighbours outside the grid (assembly and
// coarsening write only existing ones): the index is clamped, not tested.
fn row_product(l:Level,i:u32,diagonal:bool)->f32 {
    let p=vec2<i32>(i32(i%l.nx),i32(i/l.nx));let last=vec2<i32>(i32(l.nx)-1,i32(l.ny)-1);let j=l.offset+i;
    let x1=u32(clamp(p.x-1,0,last.x));let x2=u32(p.x);let x3=u32(clamp(p.x+1,0,last.x));
    let y1=l.offset+u32(clamp(p.y-1,0,last.y))*l.nx;let y2=l.offset+u32(p.y)*l.nx;let y3=l.offset+u32(clamp(p.y+1,0,last.y))*l.nx;
    if l.offset==0u&&(anchors[i]&FIVE_POINT_FLAG)!=0u{
        var value=rows[j].a[7]*solution[y1+x2]+rows[j].a[11]*solution[y2+x1]+rows[j].a[13]*solution[y2+x3]+rows[j].a[17]*solution[y3+x2];
        if diagonal{value+=rows[j].a[12]*solution[y2+x2];}
        return value;
    }
    let x0=u32(clamp(p.x-2,0,last.x));let x4=u32(clamp(p.x+2,0,last.x));
    let y0=l.offset+u32(clamp(p.y-2,0,last.y))*l.nx;let y4=l.offset+u32(clamp(p.y+2,0,last.y))*l.nx;
    // The 25 terms in the stencil's order, written out: a loop over them
    // costs this GPU more than the terms themselves.
    var value=0.;
    value+=rows[j].a[0]*solution[y0+x0];value+=rows[j].a[1]*solution[y0+x1];value+=rows[j].a[2]*solution[y0+x2];value+=rows[j].a[3]*solution[y0+x3];value+=rows[j].a[4]*solution[y0+x4];
    value+=rows[j].a[5]*solution[y1+x0];value+=rows[j].a[6]*solution[y1+x1];value+=rows[j].a[7]*solution[y1+x2];value+=rows[j].a[8]*solution[y1+x3];value+=rows[j].a[9]*solution[y1+x4];
    value+=rows[j].a[10]*solution[y2+x0];value+=rows[j].a[11]*solution[y2+x1];if diagonal{value+=rows[j].a[12]*solution[y2+x2];}value+=rows[j].a[13]*solution[y2+x3];value+=rows[j].a[14]*solution[y2+x4];
    value+=rows[j].a[15]*solution[y3+x0];value+=rows[j].a[16]*solution[y3+x1];value+=rows[j].a[17]*solution[y3+x2];value+=rows[j].a[18]*solution[y3+x3];value+=rows[j].a[19]*solution[y3+x4];
    value+=rows[j].a[20]*solution[y4+x0];value+=rows[j].a[21]*solution[y4+x1];value+=rows[j].a[22]*solution[y4+x2];value+=rows[j].a[23]*solution[y4+x3];value+=rows[j].a[24]*solution[y4+x4];
    return value;
}
// Gauss-Seidel on point k of one colour of nine: the points with x % 3 and
// y % 3 fixed, three apart, share no row's 5x5 stencil. Only the colour's
// points are dispatched, numbered along its rows.
fn relax_point(l:Level,color:u32,k:u32){
    let cx=color%3u;let cy=color/3u;let across=(l.nx-cx+2u)/3u;
    let y=cy+3u*(k/across);if y>=l.ny{return;}let i=y*l.nx+cx+3u*(k%across);
    let j=l.offset+i;let diagonal=rows[j].a[12];if diagonal>0.{solution[j]=(rows[j].rhs-row_product(l,i,false))/diagonal;}
}
@compute @workgroup_size(POINT_THREADS) fn relax(@builtin(global_invocation_id) id:vec3<u32>) {
    relax_point(dispatched_level(),cfg.color,id.x);
}
fn residual_point(l:Level,i:u32){residual[l.offset+i]=rows[l.offset+i].rhs-row_product(l,i,true);}
@compute @workgroup_size(POINT_THREADS) fn evaluate_residual(@builtin(global_invocation_id) id:vec3<u32>) {
    let l=dispatched_level();if id.x<l.nx*l.ny{residual_point(l,id.x);}
}
// The bilinear weight of coarse vertex `parent` at fine vertex p: zero on
// the finest level's fixed vertices and scaled there by the vertex's own
// weight (scale_interpolation). Its 3x3 support keeps P^T A P in a 5x5 row.
fn interpolation(l:Level,p:vec2<i32>,parent:vec2<i32>)->f32 {
    if !valid(p,l.nx,l.ny)||!valid(parent,l.cx,l.cy){return 0.;}
    if l.offset==0u&&boundary(p){return 0.;}
    let distance=abs(p-2*parent);
    if any(distance>vec2<i32>(1)){return 0.;}
    var response=1.;if l.offset==0u{response=max(0.,-tensor[at(p,l.nx)].w);}
    return response*(1.-.5*f32(distance.x))*(1.-.5*f32(distance.y));
}
// Five coarse horizontal stencil slots (ox = -2, -1, 0, +1 in `a`, ox = +2 in `b`).
struct X5 { a:vec4<f32>, b:f32 }
fn x5_zero()->X5{return X5(vec4<f32>(0.),0.);}
fn x5_add_scaled(u:X5,s:f32,v:X5)->X5{return X5(u.a+s*v.a,u.b+s*v.b);}
fn x5_comb3(u:X5,a:f32,v:X5,b:f32,w:X5,c:f32)->X5{return X5(a*u.a+b*v.a+c*w.a,a*u.b+b*v.b+c*w.b);}
fn restrict_x(x:i32,v0:f32,v1:f32,v2:f32,v3:f32,v4:f32)->X5{
    if x<0{return X5(vec4<f32>(.5*v0,.5*v0+v1+.5*v2,.5*v2+v3+.5*v4,.5*v4),0.);}
    if x==0{return X5(vec4<f32>(0.,v0+.5*v1,.5*v1+v2+.5*v3,.5*v3+v4),0.);}
    return X5(vec4<f32>(0.,.5*v0,.5*v0+v1+.5*v2,.5*v2+v3+.5*v4),.5*v4);
}
var<workgroup> coarsen_resp:array<f32,441>;
// The coarse operator P^T A P, one thread per coarse vertex in 8x8 tiles.
// Each fine 5x5 row is read once with static indices and restricted separably
// along x then y with the bilinear [0.5, 1, 0.5] weights; on level 0, fine
// interpolation responses are staged once per tile in workgroup memory and
// pure 5-point rows skip the 20 zero entries.
@compute @workgroup_size(64) fn coarsen(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) t:u32) {
    let l=dispatched_level();
    let tiles_x=(l.cx+7u)/8u;
    let bx=i32((group.x%tiles_x)*8u);let by=i32((group.x/tiles_x)*8u);
    let lx=i32(t&7u);let ly=i32(t>>3u);
    if l.offset==0u{
        let fx0=2*bx-3;let fy0=2*by-3;
        for(var k=t;k<441u;k+=64u){
            let q=vec2<i32>(fx0+i32(k%21u),fy0+i32(k/21u));
            var rv=0.;
            if valid(q,l.nx,l.ny)&&!outside_boundary(q){rv=max(0.,-tensor[at(q,l.nx)].w);}
            coarsen_resp[k]=rv;
        }
        workgroupBarrier();
    }
    let parent=vec2<i32>(bx+lx,by+ly);
    if parent.x>=i32(l.cx)||parent.y>=i32(l.cy){return;}
    var c_m2=x5_zero();var c_m1=x5_zero();var c_0=x5_zero();var c_p1=x5_zero();var c_p2=x5_zero();
    for(var y=-1;y<=1;y++){
        let py=2*parent.y+y;
        if py<0||py>=i32(l.ny){continue;}
        let wy=select(.5,1.,y==0);
        for(var x=-1;x<=1;x++){
            let px=2*parent.x+x;
            if px<0||px>=i32(l.nx){continue;}
            var left=wy*select(.5,1.,x==0);
            let fine_i=at(vec2<i32>(px,py),l.nx);
            let fine_row=l.offset+fine_i;
            var r0=x5_zero();var r1=x5_zero();var r2=x5_zero();var r3=x5_zero();var r4=x5_zero();
            if l.offset==0u{
                let cr=(2*ly+y+3)*21+(2*lx+x+3);
                let rp=coarsen_resp[u32(cr)];
                if rp<=0.{continue;}
                left*=rp;
                if (anchors[fine_i]&FIVE_POINT_FLAG)!=0u{
                    r1=restrict_x(x,0.,0.,rows[fine_row].a[7]*coarsen_resp[u32(cr-21)],0.,0.);
                    r2=restrict_x(x,0.,rows[fine_row].a[11]*coarsen_resp[u32(cr-1)],rows[fine_row].a[12]*rp,rows[fine_row].a[13]*coarsen_resp[u32(cr+1)],0.);
                    r3=restrict_x(x,0.,0.,rows[fine_row].a[17]*coarsen_resp[u32(cr+21)],0.,0.);
                }else{
                    let a=rows[fine_row].a;
                    r0=restrict_x(x,a[0]*coarsen_resp[u32(cr-44)],a[1]*coarsen_resp[u32(cr-43)],a[2]*coarsen_resp[u32(cr-42)],a[3]*coarsen_resp[u32(cr-41)],a[4]*coarsen_resp[u32(cr-40)]);
                    r1=restrict_x(x,a[5]*coarsen_resp[u32(cr-23)],a[6]*coarsen_resp[u32(cr-22)],a[7]*coarsen_resp[u32(cr-21)],a[8]*coarsen_resp[u32(cr-20)],a[9]*coarsen_resp[u32(cr-19)]);
                    r2=restrict_x(x,a[10]*coarsen_resp[u32(cr-2)],a[11]*coarsen_resp[u32(cr-1)],a[12]*rp,a[13]*coarsen_resp[u32(cr+1)],a[14]*coarsen_resp[u32(cr+2)]);
                    r3=restrict_x(x,a[15]*coarsen_resp[u32(cr+19)],a[16]*coarsen_resp[u32(cr+20)],a[17]*coarsen_resp[u32(cr+21)],a[18]*coarsen_resp[u32(cr+22)],a[19]*coarsen_resp[u32(cr+23)]);
                    r4=restrict_x(x,a[20]*coarsen_resp[u32(cr+40)],a[21]*coarsen_resp[u32(cr+41)],a[22]*coarsen_resp[u32(cr+42)],a[23]*coarsen_resp[u32(cr+43)],a[24]*coarsen_resp[u32(cr+44)]);
                }
            }else{
                let a=rows[fine_row].a;
                r0=restrict_x(x,a[0],a[1],a[2],a[3],a[4]);
                r1=restrict_x(x,a[5],a[6],a[7],a[8],a[9]);
                r2=restrict_x(x,a[10],a[11],a[12],a[13],a[14]);
                r3=restrict_x(x,a[15],a[16],a[17],a[18],a[19]);
                r4=restrict_x(x,a[20],a[21],a[22],a[23],a[24]);
            }
            if y<0{
                c_m2=x5_add_scaled(c_m2,.5*left,r0);
                c_m1=x5_add_scaled(c_m1,left,x5_comb3(r0,.5,r1,1.,r2,.5));
                c_0=x5_add_scaled(c_0,left,x5_comb3(r2,.5,r3,1.,r4,.5));
                c_p1=x5_add_scaled(c_p1,.5*left,r4);
            }else if y==0{
                c_m1=x5_add_scaled(c_m1,left,x5_add_scaled(r0,.5,r1));
                c_0=x5_add_scaled(c_0,left,x5_comb3(r1,.5,r2,1.,r3,.5));
                c_p1=x5_add_scaled(c_p1,left,x5_add_scaled(r4,.5,r3));
            }else{
                c_m1=x5_add_scaled(c_m1,.5*left,r0);
                c_0=x5_add_scaled(c_0,left,x5_comb3(r0,.5,r1,1.,r2,.5));
                c_p1=x5_add_scaled(c_p1,left,x5_comb3(r2,.5,r3,1.,r4,.5));
                c_p2=x5_add_scaled(c_p2,.5*left,r4);
            }
        }
    }
    let xm=vec4<f32>(select(0.,1.,parent.x>=2),select(0.,1.,parent.x>=1),1.,select(0.,1.,parent.x+1<i32(l.cx)));
    let xm2=select(0.,1.,parent.x+2<i32(l.cx));
    let ym2=select(0.,1.,parent.y>=2);let ym1=select(0.,1.,parent.y>=1);
    let yp1=select(0.,1.,parent.y+1<i32(l.cy));let yp2=select(0.,1.,parent.y+2<i32(l.cy));
    var out:Row;
    let o0=c_m2.a*(xm*ym2);out.a[0]=o0.x;out.a[1]=o0.y;out.a[2]=o0.z;out.a[3]=o0.w;out.a[4]=c_m2.b*xm2*ym2;
    let o1=c_m1.a*(xm*ym1);out.a[5]=o1.x;out.a[6]=o1.y;out.a[7]=o1.z;out.a[8]=o1.w;out.a[9]=c_m1.b*xm2*ym1;
    let o2=c_0.a*xm;out.a[10]=o2.x;out.a[11]=o2.y;out.a[12]=o2.z;out.a[13]=o2.w;out.a[14]=c_0.b*xm2;
    let o3=c_p1.a*(xm*yp1);out.a[15]=o3.x;out.a[16]=o3.y;out.a[17]=o3.z;out.a[18]=o3.w;out.a[19]=c_p1.b*xm2*yp1;
    let o4=c_p2.a*(xm*yp2);out.a[20]=o4.x;out.a[21]=o4.y;out.a[22]=o4.z;out.a[23]=o4.w;out.a[24]=c_p2.b*xm2*yp2;
    out.rhs=0.;
    let ci=at(parent,l.cx);
    rows[l.coarse+ci]=out;
    solution[l.coarse+ci]=0.;
}
fn restrict_point(l:Level,i:u32){
    let base=vec2<i32>(i32(i%l.cx),i32(i/l.cx))*2;var b=0.;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let p=base+vec2<i32>(x,y);let weight=interpolation(l,p,base/2);
        if weight>0.{b+=weight*residual[l.offset+at(p,l.nx)];}
    }}rows[l.coarse+i].rhs=b;solution[l.coarse+i]=0.;
}
@compute @workgroup_size(POINT_THREADS) fn restrict_rhs(@builtin(global_invocation_id) id:vec3<u32>) {
    let l=dispatched_level();if id.x<l.cx*l.cy{restrict_point(l,id.x);}
}
fn prolong_point(l:Level,i:u32){
    let p=vec2<i32>(i32(i%l.nx),i32(i/l.nx));
    if l.offset==0u&&boundary(p){solution[i]=0.;return;}
    var value=0.;let base=p/2;
    let p00=base;let w00=interpolation(l,p,p00);if w00>0.{value+=w00*solution[l.coarse+at(p00,l.cx)];}
    let p10=base+vec2<i32>(1,0);let w10=interpolation(l,p,p10);if w10>0.{value+=w10*solution[l.coarse+at(p10,l.cx)];}
    let p01=base+vec2<i32>(0,1);let w01=interpolation(l,p,p01);if w01>0.{value+=w01*solution[l.coarse+at(p01,l.cx)];}
    let p11=base+vec2<i32>(1,1);let w11=interpolation(l,p,p11);if w11>0.{value+=w11*solution[l.coarse+at(p11,l.cx)];}
    solution[l.offset+i]+=value;
}
@compute @workgroup_size(POINT_THREADS) fn prolong(@builtin(global_invocation_id) id:vec3<u32>) {
    let l=dispatched_level();if id.x<l.nx*l.ny{prolong_point(l,id.x);}
}
fn entry(l:Level,i:u32,j:u32)->f32 {
    if i==j{return select(rows[l.offset+i].a[12],1.,rows[l.offset+i].a[12]==0.);}
    let d=vec2<i32>(i32(j%l.nx)-i32(i%l.nx),i32(j/l.nx)-i32(i/l.nx));
    if any(abs(d)>vec2<i32>(2)){return 0.;}return rows[l.offset+i].a[slot(d)];
}
var<workgroup> sums:array<vec2<f32>,128>;
var<workgroup> x:array<f32,POINT_THREADS>;
var<workgroup> r:array<f32,POINT_THREADS>;
var<workgroup> direction:array<f32,POINT_THREADS>;
// Workgroup-staged coarse V-cycle state: holds `solution` for all levels in
// `coarse_cycle` (<= 1152 points), the active level's `residual` (<= 1024),
// and `rhs` for levels below the entry level (<= 320). Staging in workgroup
// memory avoids `storageBarrier()` inside single-workgroup loops (which hits
// Apple Metal load-reordering bugs) and keeps every coarse sweep on-chip.
const COARSE_SOL_CAP:u32=1152u;
const COARSE_RES_CAP:u32=1024u;
const COARSE_RHS_CAP:u32=320u;
var<workgroup> coarse_sol:array<f32,COARSE_SOL_CAP>;
var<workgroup> coarse_res:array<f32,COARSE_RES_CAP>;
var<workgroup> coarse_rhs:array<f32,COARSE_RHS_CAP>;
// Two workgroup sums in one tree: each component adds in the order of a
// single sum. The tree spans the first 128 threads, which hold every point
// of the coarsest level; the others wait at its barriers.
fn totals(v:vec2<f32>,i:u32)->vec2<f32> {if i<128u{sums[i]=v;}workgroupBarrier();for(var k=64u;k>0u;k/=2u){if i<k{sums[i]+=sums[i+k];}workgroupBarrier();}return workgroupUniformLoad(&sums[0]);}
fn sparse_product(l:Level,i:u32)->f32 {
    let p=vec2<i32>(i32(i%l.nx),i32(i/l.nx));let last=vec2<i32>(i32(l.nx)-1,i32(l.ny)-1);let j=l.offset+i;
    let x0=u32(clamp(p.x-2,0,last.x));let x1=u32(clamp(p.x-1,0,last.x));let x2=u32(p.x);
    let x3=u32(clamp(p.x+1,0,last.x));let x4=u32(clamp(p.x+2,0,last.x));
    let y0=u32(clamp(p.y-2,0,last.y))*l.nx;let y1=u32(clamp(p.y-1,0,last.y))*l.nx;let y2=u32(p.y)*l.nx;
    let y3=u32(clamp(p.y+1,0,last.y))*l.nx;let y4=u32(clamp(p.y+2,0,last.y))*l.nx;
    let diag=select(rows[j].a[12],1.,rows[j].a[12]==0.);
    var value=0.;
    value+=rows[j].a[0]*direction[y0+x0];value+=rows[j].a[1]*direction[y0+x1];value+=rows[j].a[2]*direction[y0+x2];value+=rows[j].a[3]*direction[y0+x3];value+=rows[j].a[4]*direction[y0+x4];
    value+=rows[j].a[5]*direction[y1+x0];value+=rows[j].a[6]*direction[y1+x1];value+=rows[j].a[7]*direction[y1+x2];value+=rows[j].a[8]*direction[y1+x3];value+=rows[j].a[9]*direction[y1+x4];
    value+=rows[j].a[10]*direction[y2+x0];value+=rows[j].a[11]*direction[y2+x1];value+=diag*direction[y2+x2];value+=rows[j].a[13]*direction[y2+x3];value+=rows[j].a[14]*direction[y2+x4];
    value+=rows[j].a[15]*direction[y3+x0];value+=rows[j].a[16]*direction[y3+x1];value+=rows[j].a[17]*direction[y3+x2];value+=rows[j].a[18]*direction[y3+x3];value+=rows[j].a[19]*direction[y3+x4];
    value+=rows[j].a[20]*direction[y4+x0];value+=rows[j].a[21]*direction[y4+x1];value+=rows[j].a[22]*direction[y4+x2];value+=rows[j].a[23]*direction[y4+x3];value+=rows[j].a[24]*direction[y4+x4];
    return value;
}
// The Jacobi preconditioner, from the operator's present diagonal: an
// inverse cached earlier is wrong once a passage has closed.
fn coarse_response(l:Level,i:u32,n:u32)->f32{if i<n{return r[i]/entry(l,i,i);}return 0.;}
fn coarse_rhs_value(l:Level,base:u32,rhs_base:u32,i:u32)->f32{
    if l.offset==base{return rows[l.offset+i].rhs;}
    return coarse_rhs[l.offset-rhs_base+i];
}
fn coarse_row_product(l:Level,base:u32,i:u32,diagonal:bool)->f32 {
    let p=vec2<i32>(i32(i%l.nx),i32(i/l.nx));let last=vec2<i32>(i32(l.nx)-1,i32(l.ny)-1);
    let j=l.offset+i;let sbase=l.offset-base;
    let x0=u32(clamp(p.x-2,0,last.x));let x1=u32(clamp(p.x-1,0,last.x));let x2=u32(p.x);
    let x3=u32(clamp(p.x+1,0,last.x));let x4=u32(clamp(p.x+2,0,last.x));
    let y0=sbase+u32(clamp(p.y-2,0,last.y))*l.nx;let y1=sbase+u32(clamp(p.y-1,0,last.y))*l.nx;let y2=sbase+u32(p.y)*l.nx;
    let y3=sbase+u32(clamp(p.y+1,0,last.y))*l.nx;let y4=sbase+u32(clamp(p.y+2,0,last.y))*l.nx;
    var value=0.;
    value+=rows[j].a[0]*coarse_sol[y0+x0];value+=rows[j].a[1]*coarse_sol[y0+x1];value+=rows[j].a[2]*coarse_sol[y0+x2];value+=rows[j].a[3]*coarse_sol[y0+x3];value+=rows[j].a[4]*coarse_sol[y0+x4];
    value+=rows[j].a[5]*coarse_sol[y1+x0];value+=rows[j].a[6]*coarse_sol[y1+x1];value+=rows[j].a[7]*coarse_sol[y1+x2];value+=rows[j].a[8]*coarse_sol[y1+x3];value+=rows[j].a[9]*coarse_sol[y1+x4];
    value+=rows[j].a[10]*coarse_sol[y2+x0];value+=rows[j].a[11]*coarse_sol[y2+x1];if diagonal{value+=rows[j].a[12]*coarse_sol[y2+x2];}value+=rows[j].a[13]*coarse_sol[y2+x3];value+=rows[j].a[14]*coarse_sol[y2+x4];
    value+=rows[j].a[15]*coarse_sol[y3+x0];value+=rows[j].a[16]*coarse_sol[y3+x1];value+=rows[j].a[17]*coarse_sol[y3+x2];value+=rows[j].a[18]*coarse_sol[y3+x3];value+=rows[j].a[19]*coarse_sol[y3+x4];
    value+=rows[j].a[20]*coarse_sol[y4+x0];value+=rows[j].a[21]*coarse_sol[y4+x1];value+=rows[j].a[22]*coarse_sol[y4+x2];value+=rows[j].a[23]*coarse_sol[y4+x3];value+=rows[j].a[24]*coarse_sol[y4+x4];
    return value;
}
fn coarse_smooth_colour(l:Level,base:u32,rhs_base:u32,color:u32,t:u32){
    let cx=color%3u;let cy=color/3u;let across=(l.nx-cx+2u)/3u;let points=across*((l.ny-cy+2u)/3u);
    let sbase=l.offset-base;
    for(var k=t;k<points;k+=POINT_THREADS){
        let y=cy+3u*(k/across);if y>=l.ny{continue;}let i=y*l.nx+cx+3u*(k%across);
        let diagonal=rows[l.offset+i].a[12];
        if diagonal>0.{coarse_sol[sbase+i]=(coarse_rhs_value(l,base,rhs_base,i)-coarse_row_product(l,base,i,false))/diagonal;}
    }
}
fn coarse_interpolation(l:Level,p:vec2<i32>,parent:vec2<i32>)->f32{
    if !valid(p,l.nx,l.ny)||!valid(parent,l.cx,l.cy){return 0.;}
    let distance=abs(p-2*parent);
    if any(distance>vec2<i32>(1)){return 0.;}
    return (1.-.5*f32(distance.x))*(1.-.5*f32(distance.y));
}
// Conjugate gradients on the coarsest level, one point per invocation: at
// most eight steps, ended by a residual 1e-6 of the first or by a direction
// without positive energy.
fn solve_coarsest(l:Level,base:u32,rhs_base:u32,i:u32){
    let n=l.nx*l.ny;
    x[i]=0.;r[i]=0.;direction[i]=0.;if i<n{r[i]=coarse_rhs_value(l,base,rhs_base,i);}workgroupBarrier();
    // r.r decides whether to go on, r.z is the step's numerator; both are
    // summed together after each update of r.
    var previous=1.;var z=coarse_response(l,i,n);var sums=totals(vec2<f32>(r[i]*r[i],r[i]*z),i);
    let initial=sums.x;
    for(var step=0u;step<8u;step++){
        if sums.x<=initial*1e-12||!(initial>0.&&initial<1e34){break;}
        let rz=sums.y;if !(abs(rz)<1e34){break;}
        var beta=0.;if step>0u&&previous>0.&&previous<1e34{beta=clamp(rz/previous,0.,1e6);}
        direction[i]=z+beta*direction[i];workgroupBarrier();var ad=0.;
        if i<n{ad=sparse_product(l,i);}
        let denominator=totals(vec2<f32>(direction[i]*ad,0.),i).x;if !(denominator>0.&&denominator<1e34){break;}let alpha=clamp(rz/denominator,-1e6,1e6);
        x[i]+=alpha*direction[i];r[i]-=alpha*ad;workgroupBarrier();previous=rz;
        z=coarse_response(l,i,n);sums=totals(vec2<f32>(r[i]*r[i],r[i]*z),i);
    }
    if i<n{coarse_sol[l.offset-base+i]=x[i];}
}
// Entry k: level k's size and offset, zero past the coarsest.
@group(0) @binding(10) var<storage,read_write> hierarchy:array<vec4<u32>>;
fn hierarchy_level(k:u32)->Level{
    let level=hierarchy[k];let coarse=hierarchy[k+1u];
    return Level(level.x,level.y,level.z,coarse.x,coarse.y,coarse.z);
}
// The cycle's levels from hierarchy[0] down in one workgroup and one
// dispatch (a small level's dispatches cost more than its work), in the
// dispatched cycle's order: each level's nine colours, its residual and the
// restriction, the coarsest level's conjugate gradients, then each level's
// prolongation and its colours in reverse. All intermediate fields stay in
// workgroup memory with pure workgroupBarrier() synchronisation; `solution`
// is written once at the end.
@compute @workgroup_size(POINT_THREADS) fn coarse_cycle(@builtin(local_invocation_index) t:u32){
    let smoothed=cfg.color;
    let l0=hierarchy_level(0u);
    let base=l0.offset;
    let rhs_base=select(base,hierarchy_level(1u).offset,smoothed>0u);
    let last=hierarchy_level(smoothed);
    let total_sol=last.offset-base+last.nx*last.ny;
    for(var i=t;i<total_sol;i+=POINT_THREADS){coarse_sol[i]=0.;}
    workgroupBarrier();
    for(var k=0u;k<smoothed;k++){
        let l=hierarchy_level(k);
        for(var color=0u;color<9u;color++){coarse_smooth_colour(l,base,rhs_base,color,t);workgroupBarrier();}
        for(var i=t;i<l.nx*l.ny;i+=POINT_THREADS){coarse_res[i]=coarse_rhs_value(l,base,rhs_base,i)-coarse_row_product(l,base,i,true);}
        workgroupBarrier();
        for(var i=t;i<l.cx*l.cy;i+=POINT_THREADS){
            let cbase=vec2<i32>(i32(i%l.cx),i32(i/l.cx))*2;var b=0.;
            for(var ry=-1;ry<=1;ry++){for(var rx=-1;rx<=1;rx++){
                let p=cbase+vec2<i32>(rx,ry);let weight=coarse_interpolation(l,p,cbase/2);
                if weight>0.{b+=weight*coarse_res[at(p,l.nx)];}
            }}
            coarse_rhs[l.coarse-rhs_base+i]=b;coarse_sol[l.coarse-base+i]=0.;
        }
        workgroupBarrier();
    }
    solve_coarsest(last,base,rhs_base,t);
    workgroupBarrier();
    for(var k=smoothed;k>0u;k--){
        let l=hierarchy_level(k-1u);
        let sbase=l.offset-base;let cbase=l.coarse-base;
        for(var i=t;i<l.nx*l.ny;i+=POINT_THREADS){
            let p=vec2<i32>(i32(i%l.nx),i32(i/l.nx));var value=0.;let pbase=p/2;
            for(var py=0;py<=1;py++){for(var px=0;px<=1;px++){
                let parent=pbase+vec2<i32>(px,py);let weight=coarse_interpolation(l,p,parent);
                if weight>0.{value+=weight*coarse_sol[cbase+at(parent,l.cx)];}
            }}
            coarse_sol[sbase+i]+=value;
        }
        workgroupBarrier();
        for(var color=0u;color<9u;color++){coarse_smooth_colour(l,base,rhs_base,8u-color,t);workgroupBarrier();}
    }
    for(var i=t;i<total_sol;i+=POINT_THREADS){solution[base+i]=coarse_sol[i];}
}

// The tension's force on psi. tau = sigma |grad(alpha)| (I - n n^T), and for
// u = (psi_y, -psi_x), -tau:grad(u) = tau_xy psi_xx + (tau_yy - tau_xx)
// psi_xy - tau_xy psi_yy. It uses the adjoint of the centred Hessian of the
// implicit term: a constant or affine psi has zero Hessian, so the tension
// imparts no net force or torque. tensor.xyz holds dt^2 sigma/(rho h^3)
// times the tangent measure; times h^2/dt the right-hand side scales as
// dt sigma/(rho h). CAPILLARY_DT is the update interval DT (stream.rs).
fn capillary_rhs(p:vec2<i32>)->f32 {
    var value=0.;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let d=vec2<i32>(x,y);let center=p+d;
        if !valid(center,cfg.nx,cfg.ny){continue;}
        let s=tensor[at(center,cfg.nx)].xyz;if all(s==vec3<f32>(0.)){continue;}
        let b=gradient(center,-d);
        value-=s.x*b.x+s.y*(b.y+b.z)+s.z*b.w;
    }}
    return (cfg.h*cfg.h/CAPILLARY_DT)*value;
}

// The neighbour with the largest coefficient in row i (the lower index on a
// tie), or i itself.
fn strongest_neighbor(i:u32)->u32{
    let p=vec2<i32>(i32(i%cfg.nx),i32(i/cfg.nx));var peer=i;var largest=0.;
    for(var y=-2;y<=2;y++){for(var x=-2;x<=2;x++){
        if x==0&&y==0{continue;}let q=p+vec2<i32>(x,y);if !valid(q,cfg.nx,cfg.ny){continue;}
        let j=at(q,cfg.nx);let amount=abs(rows[i].a[slot(vec2<i32>(x,y))]);
        if amount>largest||(amount==largest&&amount>0.&&j<peer){largest=amount;peer=j;}
    }}
    return peer;
}
fn external_rhs(i:u32,peer:u32)->f32{
    let p=vec2<i32>(i32(i%cfg.nx),i32(i/cfg.nx));var value=rows[i].rhs;
    if (anchors[i]&FIVE_POINT_FLAG)!=0u{
        if p.y>0{let j=i-cfg.nx;if j!=peer{value-=rows[i].a[7]*solution[j];}}
        if p.x>0{let j=i-1u;if j!=peer{value-=rows[i].a[11]*solution[j];}}
        if p.x+1<i32(cfg.nx){let j=i+1u;if j!=peer{value-=rows[i].a[13]*solution[j];}}
        if p.y+1<i32(cfg.ny){let j=i+cfg.nx;if j!=peer{value-=rows[i].a[17]*solution[j];}}
        return value;
    }
    for(var y=-2;y<=2;y++){for(var x=-2;x<=2;x++){
        if x==0&&y==0{continue;}let q=p+vec2<i32>(x,y);if !valid(q,cfg.nx,cfg.ny){continue;}
        let j=at(q,cfg.nx);if j!=peer{value-=rows[i].a[slot(vec2<i32>(x,y))]*solution[j];}
    }}
    return value;
}
// The pair (i, j) solved with every other unknown frozen: x = 1 if it has a
// solution, y = psi_i. The sum and difference basis separates the pair's
// stiff relative mode from its common mode and avoids subtracting products
// of two large diagonals.
fn block_prediction(i:u32,j:u32)->vec2<f32>{
    let left=min(i,j);let right=max(i,j);let a=rows[left].a[12];let d=rows[right].a[12];
    let p=vec2<i32>(i32(right%cfg.nx)-i32(left%cfg.nx),i32(right/cfg.nx)-i32(left/cfg.nx));
    let b=rows[left].a[slot(p)];let g=abs(b);let orientation=select(1.,-1.,b>0.);
    let stiff=(a+g)+(d+g);if stiff<=0.{return vec2<f32>(0.);}
    let mixed=a-d;let weak=(a-g)+(d-g);let denominator=weak-(mixed/stiff)*mixed;
    // A rounded block that is singular or not positive has no minimum of
    // its own: the row's scalar prediction stands.
    if !(denominator>1e-12*stiff){return vec2<f32>(0.);}
    let l=external_rhs(left,right);let r=orientation*external_rhs(right,left);
    let mean=(l+r-(mixed/stiff)*(l-r))/denominator;
    let difference=(l-r-mixed*mean)/stiff;
    return vec2<f32>(1.,select(orientation*(mean-difference),mean+difference,i==left));
}
// The finest rows' anchoring, fixed by the rows alone for the whole solve:
// the mutually strongest neighbour (low 30 bits, NO_PEER if none), the
// 5-point stencil flag (bit 30, written by assemble), and strict diagonal
// dominance (bit 31). The cycles then search no stencils.
@group(0) @binding(9) var<storage,read_write> anchors:array<u32>;
const PEER_MASK:u32=0x3fffffffu;
const FIVE_POINT_FLAG:u32=1u<<30u;
const NO_PEER:u32=PEER_MASK;
fn off_diagonal_sum(i:u32,five_point:bool)->f32{
    if five_point{
        return abs(rows[i].a[7])+abs(rows[i].a[11])+abs(rows[i].a[13])+abs(rows[i].a[17]);
    }
    let a=rows[i].a;
    return abs(a[0])+abs(a[1])+abs(a[2])+abs(a[3])+abs(a[4])
        +abs(a[5])+abs(a[6])+abs(a[7])+abs(a[8])+abs(a[9])
        +abs(a[10])+abs(a[11])+abs(a[13])+abs(a[14])
        +abs(a[15])+abs(a[16])+abs(a[17])+abs(a[18])+abs(a[19])
        +abs(a[20])+abs(a[21])+abs(a[22])+abs(a[23])+abs(a[24]);
}
@compute @workgroup_size(POINT_THREADS) fn anchor_plan(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.nx*cfg.ny{return;}
    let five=anchors[i]&FIVE_POINT_FLAG;
    let off=off_diagonal_sum(i,five!=0u);
    var plan=NO_PEER;let peer=strongest_neighbor(i);
    if peer!=i&&strongest_neighbor(peer)==i{plan=peer;}
    if rows[i].a[12]>off{plan|=1u<<31u;}
    anchors[i]=plan|five;
}
// Before a cycle's sweeps: a row mutually strongest with a neighbour takes
// the pair's solution, a strictly dominant row its Jacobi value, the others
// keep theirs. Rows anchored by a fixed boundary or by each other are thus
// resolved before a colour sweep can spread their stale value to the less
// anchored ones. Dominance is a property of the row, not a cutoff on an
// aperture. The values go to `residual` and anchor_apply stores them in a
// second dispatch, so that no prediction reads a new neighbour.
@compute @workgroup_size(POINT_THREADS) fn anchor_predict(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.nx*cfg.ny{return;}
    let plan=anchors[i];let peer=plan&PEER_MASK;
    var value=solution[i];var paired=false;
    if peer!=NO_PEER{
        let prediction=block_prediction(i,peer);
        if prediction.x>0.{value=prediction.y;paired=true;}
    }
    if !paired&&(plan>>31u)!=0u{let row=rows[i];value=(row.rhs-row_product(dispatched_level(),i,false))/row.a[12];}
    residual[i]=value;
}

@compute @workgroup_size(POINT_THREADS) fn anchor_apply(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i<cfg.nx*cfg.ny{solution[i]=residual[i];}
}

// The finest level's interpolation weight of a vertex, s = min(1, sum of
// |a_ij| / a_ii), kept as -s in tensor.w: the coarse correction reaches a
// row no further than the row answers a unit correction of its neighbours.
// The finest equation is unchanged.
@compute @workgroup_size(POINT_THREADS) fn scale_interpolation(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=cfg.nx*cfg.ny||tensor[i].w>0.{return;}
    let diag=rows[i].a[12];
    var response=0.;if diag>0.{response=min(1.,off_diagonal_sum(i,(anchors[i]&FIVE_POINT_FLAG)!=0u)/diag);}
    tensor[i].w=-response;
}
