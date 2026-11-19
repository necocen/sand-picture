// The gas of a cell as shapes, built at the end of each update (stages
// InterfacePlanes, InterfaceContacts): parallel planes, which carry the gas,
// and contact shapes, planes or disks, that close faces to grains and liquid.

// Where an identity's gas lies around cell p: the first moment of its 3x3
// amounts along the common axis. The layers of one side stack in this order,
// the furthest on the gas side innermost, and the identity only breaks ties:
// by identity alone the sweep would carry the wrong bubble across a face.
fn layer_key(p:vec2<i32>,label:f32,orientation:vec2<f32>)->f32 {
    var moment=vec2<f32>(0.);var mass=0.;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let q=label_fraction(p+vec2<i32>(x,y),label);moment+=q*vec2<f32>(f32(x),f32(y));mass+=q;
    }}
    if mass<=0.{return 0.;}
    return dot(moment/mass,orientation);
}
fn layer_before(positive:bool,key:f32,label:f32,other_key:f32,other_label:f32)->bool {
    if other_key!=key{return select(other_key>key,other_key<key,positive);}
    return other_label<label;
}
// The identities' layers of cell i as strips between parallel planes
// (SHAPE_ROW): one axis, the principal axis of their normals weighted by
// gas, and intercepts that hold each layer's gas and that of those inside it.
fn parallel_gas_layers(i:u32,base:u32) {
    let p=point(i);
    var tensor=vec3<f32>(0.);var largest=0.;var fallback=vec2<f32>(1.,0.);
    /*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){
        let gas=geometry[layer_index(i,l)].x;if gas<=0.{continue;}
        let n=shared_normals[base+l];let norm=dot(n,n);
        if norm>0.{tensor+=(gas/norm)*vec3<f32>(n.x*n.x,n.x*n.y,n.y*n.y);}
        if gas>largest{largest=gas;fallback=n;}
    }
    let difference=tensor.x-tensor.z;
    let eigen=.5*(tensor.x+tensor.z+sqrt(difference*difference+4.*tensor.y*tensor.y));
    var orientation=select(vec2<f32>(tensor.y,eigen-tensor.x),vec2<f32>(eigen-tensor.z,tensor.y),tensor.x>=tensor.z);
    let length=abs(orientation.x)+abs(orientation.y);
    if length>0.{orientation/=length;}else{orientation=fallback;}
    let wet=max(0.,liquid(cells[i]));
    /*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){
        let entry=geometry[layer_index(i,l)];var value=vec4<f32>(0.);
        if entry.x>0.&&entry.y>0.{
            let positive=dot(shared_normals[base+l],orientation)>=0.;
            let n=select(-orientation,orientation,positive);
            let key=layer_key(p,entry.y,orientation);
            var cumulative=entry.x;var remainder=wet;
            /*loop: slower written out*/for(var k=0u;k<GAS_SLOTS;k++){
                if k==l{continue;}
                let other=geometry[layer_index(i,k)];
                if other.x<=0.||other.y==0.{continue;}
                let same=(dot(shared_normals[base+k],orientation)>=0.)==positive;
                if same&&layer_before(positive,key,entry.y,layer_key(p,other.y,orientation),other.y){cumulative+=other.x;}
                else{remainder+=other.x;}
            }
            var alpha=0.;
            if cumulative<=remainder{alpha=intercept(n,cumulative/(cumulative+remainder));}
            else{alpha=-intercept(-n,remainder/(cumulative+remainder));}
            value=vec4<f32>(n,alpha,0.);
        }
        geometry[shape_index(i,l)]=value;
    }
}
// The share of the rectangle lo..hi of cell i that identity l's strip covers:
// its half-plane less the next smaller one on its side.
fn parallel_slab(i:u32,l:u32,lo:vec2<f32>,hi:vec2<f32>)->f32 {
    let entry=geometry[layer_index(i,l)];if entry.x<=0.{return 0.;}
    let shape=geometry[shape_index(i,l)];var previous=-1.e10;var found=false;
    /*loop: slower written out*/for(var k=0u;k<GAS_SLOTS;k++){
        if k==l{continue;}
        let other=geometry[layer_index(i,k)];let s=geometry[shape_index(i,k)];
        if other.x>0.&&other.y>0.&&dot(shape.xy,s.xy)>0.&&s.z<shape.z{
            previous=max(previous,s.z);found=true;
        }
    }
    let outer=rectangle_fraction(shape.xy,shape.z,lo,hi);var inner=0.;
    if found{inner=rectangle_fraction(shape.xy,previous,lo,hi);}
    return max(0.,outer-inner);
}

// On each side the half-plane of the largest intercept contains the others,
// so the cell's gas, as an obstacle, is at most two clipped polygons.
struct OuterGas { shapes:array<vec4<f32>,2>, present:vec2<u32> }
fn outer_gas_shapes(i:u32)->OuterGas {
    var result:OuterGas;
    /*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){
        if geometry[layer_index(i,l)].x<=0.{continue;}
        let shape=geometry[shape_index(i,l)];if !represented_gas_area(shape){continue;}var side=0u;
        if result.present.x>0u&&dot(result.shapes[0].xy,shape.xy)<0.{side=1u;}
        if result.present[side]==0u||shape.z>result.shapes[side].z{
            result.shapes[side]=shape;result.present[side]=1u;
        }
    }return result;
}

var<workgroup> shared_normals:array<vec2<f32>,576>;
fn lone_blob_moves(i:u32)->bool {
    let c=cells[i];if max(c.bed.x,0.)+max(c.bed.y,0.)>0.{return false;}
    if c.volume.gas*cfg.h<GRAIN_LAYER{let slip=bubble_slip(i);return slip.x!=0.||slip.y!=0.;}
    let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    return wet>0.&&solid<LOOSE_PACKING*wet;
}
fn lone_label(p:vec2<i32>,label:f32)->bool {
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let q=p+vec2<i32>(x,y);
        if (x!=0||y!=0)&&valid(q)&&label_fraction(q,label)*cfg.h>=GRAIN_LAYER{return false;}
    }}
    return true;
}
// The normal of an identity's plane in cell i. Around a bubble that no
// neighbour holds a grain layer of, the fractions do not say where in the
// cell it lies: the MYC plane may put its gas against a face it does not
// cross, and the sweep towards the face it does cross finds none. Smaller
// than the gap, such a bubble is a sphere moving with its gas velocity and
// lies against the face it moves towards, where it is free to move: in the
// pores of the deposit or of sand in contact it keeps the MYC normal.
fn label_plane_normal(i:u32,label:f32)->vec2<f32> {
    let p=point(i);let c=cells[i];
    if c.volume.gas>0.&&lone_blob_moves(i)&&lone_label(p,label){
        let v=c.momentum[3];
        if dot(v,v)>0.{return l1_normal(v);}
    }
    return label_normal(p,label);
}
// Two lanes per cell compute its identities' normals into shared_normals;
// the first then builds its layers and the plane of all its gas (row 0).
fn paired_reconstruct(thread:u32,enabled:bool) {
    let i=thread/2u;let lane=thread%2u;let base=(thread%128u)/2u*GAS_SLOTS;
    if i<cfg.n&&enabled {
        for(var l=lane;l<GAS_SLOTS;l+=2u) {
            var n=vec2<f32>(0.);
            let entry=geometry[layer_index(i,l)];
            if entry.x>0.&&entry.y>0.{n=label_plane_normal(i,entry.y);}
            shared_normals[base+l]=n;
        }
    }
    workgroupBarrier();
    if i>=cfg.n||lane!=0u||!enabled{return;}
    parallel_gas_layers(i,base);
    let c=cells[i].volume.gas;let g=-young_gradient(point(i));let scale=abs(g.x)+abs(g.y);
    var n=vec2<f32>(1.,0.);if scale>0.{n=g/scale;}
    geometry[i]=vec4<f32>(n,intercept(n,clamp(c,0.,1.)),0.);
}

// Whether the half-plane n.x x + n.y y <= alpha covers some area of the unit
// cell: a line or a point is no obstacle (its gas stays in the inventory).
fn represented_gas_area(shape:vec4<f32>)->bool {
    let width=abs(shape.x)+abs(shape.y);
    return width>0.&&shape.z>-.5*width;
}

const CIRCLE_TAU:f32=6.283185307179586;
// Missing side intersections sort after every angle in [0, 2 pi].
const NO_CUT:f32=1.e30;
struct SideCut { angles:vec2<f32>, edge:f32 }
fn arc_cut(a:u32,side:f32,tangent:f32)->f32 {
    var angle=select(atan2(tangent,side),atan2(side,tangent),a==1u);
    if angle<0.{angle+=CIRCLE_TAU;}
    return angle;
}
fn side_cut(a:u32,side:f32,lo:vec2<f32>,hi:vec2<f32>,r:f32)->SideCut {
    if abs(side)>=r{return SideCut(vec2<f32>(NO_CUT),0.);}
    let extent=sqrt(max(0.,r*r-side*side));
    let edge=max(0.,min(hi[1u-a],extent)-max(lo[1u-a],-extent));
    return SideCut(vec2<f32>(arc_cut(a,side,-extent),arc_cut(a,side,extent)),edge);
}
fn arc_inside(first:f32,second:f32,r:f32,lo:vec2<f32>,hi:vec2<f32>)->bool {
    let theta=.5*(first+second);let p=r*vec2<f32>(cos(theta),sin(theta));
    return all(p>=lo)&&all(p<=hi);
}
// The area of disk d (centre, radius) within the unit cell at `cell`, and its
// derivatives by (cx, cy, r): the arcs and the cell's edges inside the disk.
fn disk_area(d:vec3<f32>,cell:vec2<f32>)->vec4<f32> {
    let lo=cell-vec2<f32>(.5)-d.xy;let hi=lo+vec2<f32>(1.);let r=d.z;
    let nearest=max(lo,min(hi,vec2<f32>(0.)));
    if dot(nearest,nearest)>=r*r{return vec4<f32>(0.);}
    let farthest=max(abs(lo),abs(hi));
    if dot(farthest,farthest)<=r*r{return vec4<f32>(1.,0.,0.,0.);}
    if all(lo<=vec2<f32>(-r))&&all(hi>=vec2<f32>(r)){return vec4<f32>(.5*CIRCLE_TAU*r*r,0.,0.,CIRCLE_TAU*r);}
    let left=side_cut(0u,lo.x,lo,hi,r);let right=side_cut(0u,hi.x,lo,hi,r);
    let bottom=side_cut(1u,lo.y,lo,hi,r);let top=side_cut(1u,hi.y,lo,hi,r);
    let low_edge=vec2<f32>(left.edge,bottom.edge);let high_edge=vec2<f32>(right.edge,top.edge);
    // Batcher's 19-comparator network sorts the eight intersection angles.
    var c=array<f32,8>(left.angles.x,left.angles.y,right.angles.x,right.angles.y,
        bottom.angles.x,bottom.angles.y,top.angles.x,top.angles.y);
    var t=vec2<f32>(0.);
    t=vec2<f32>(min(c[0],c[2]),max(c[0],c[2]));c[0]=t.x;c[2]=t.y;
    t=vec2<f32>(min(c[1],c[3]),max(c[1],c[3]));c[1]=t.x;c[3]=t.y;
    t=vec2<f32>(min(c[4],c[6]),max(c[4],c[6]));c[4]=t.x;c[6]=t.y;
    t=vec2<f32>(min(c[5],c[7]),max(c[5],c[7]));c[5]=t.x;c[7]=t.y;
    t=vec2<f32>(min(c[0],c[4]),max(c[0],c[4]));c[0]=t.x;c[4]=t.y;
    t=vec2<f32>(min(c[1],c[5]),max(c[1],c[5]));c[1]=t.x;c[5]=t.y;
    t=vec2<f32>(min(c[2],c[6]),max(c[2],c[6]));c[2]=t.x;c[6]=t.y;
    t=vec2<f32>(min(c[3],c[7]),max(c[3],c[7]));c[3]=t.x;c[7]=t.y;
    t=vec2<f32>(min(c[0],c[1]),max(c[0],c[1]));c[0]=t.x;c[1]=t.y;
    t=vec2<f32>(min(c[2],c[3]),max(c[2],c[3]));c[2]=t.x;c[3]=t.y;
    t=vec2<f32>(min(c[4],c[5]),max(c[4],c[5]));c[4]=t.x;c[5]=t.y;
    t=vec2<f32>(min(c[6],c[7]),max(c[6],c[7]));c[6]=t.x;c[7]=t.y;
    t=vec2<f32>(min(c[2],c[4]),max(c[2],c[4]));c[2]=t.x;c[4]=t.y;
    t=vec2<f32>(min(c[3],c[5]),max(c[3],c[5]));c[3]=t.x;c[5]=t.y;
    t=vec2<f32>(min(c[1],c[4]),max(c[1],c[4]));c[1]=t.x;c[4]=t.y;
    t=vec2<f32>(min(c[3],c[6]),max(c[3],c[6]));c[3]=t.x;c[6]=t.y;
    t=vec2<f32>(min(c[1],c[2]),max(c[1],c[2]));c[1]=t.x;c[2]=t.y;
    t=vec2<f32>(min(c[3],c[4]),max(c[3],c[4]));c[3]=t.x;c[4]=t.y;
    t=vec2<f32>(min(c[5],c[6]),max(c[5],c[6]));c[5]=t.x;c[6]=t.y;
    var angle=0.;var previous=0.;
    if c[0]<NO_CUT{if arc_inside(previous,c[0],r,lo,hi){angle+=c[0]-previous;}previous=c[0];}
    if c[1]<NO_CUT{if arc_inside(previous,c[1],r,lo,hi){angle+=c[1]-previous;}previous=c[1];}
    if c[2]<NO_CUT{if arc_inside(previous,c[2],r,lo,hi){angle+=c[2]-previous;}previous=c[2];}
    if c[3]<NO_CUT{if arc_inside(previous,c[3],r,lo,hi){angle+=c[3]-previous;}previous=c[3];}
    if c[4]<NO_CUT{if arc_inside(previous,c[4],r,lo,hi){angle+=c[4]-previous;}previous=c[4];}
    if c[5]<NO_CUT{if arc_inside(previous,c[5],r,lo,hi){angle+=c[5]-previous;}previous=c[5];}
    if c[6]<NO_CUT{if arc_inside(previous,c[6],r,lo,hi){angle+=c[6]-previous;}previous=c[6];}
    if c[7]<NO_CUT{if arc_inside(previous,c[7],r,lo,hi){angle+=c[7]-previous;}previous=c[7];}
    if arc_inside(previous,CIRCLE_TAU,r,lo,hi){angle+=CIRCLE_TAU-previous;}
    let area=.5*(r*r*angle+dot(hi,high_edge)-dot(lo,low_edge));
    return vec4<f32>(clamp(area,0.,1.),low_edge-high_edge,r*angle);
}
fn symmetric_solve(m:mat3x3<f32>,b:vec3<f32>)->vec4<f32> {
    let co0=cross(m[1],m[2]);let co1=cross(m[2],m[0]);let co2=cross(m[0],m[1]);
    let determinant=dot(m[0],co0);
    let scale=max(max(length(m[0]),length(m[1])),length(m[2]));
    if abs(determinant)<=1.e-8*scale*scale*scale{return vec4<f32>(0.);}
    return vec4<f32>(dot(co0,b),dot(co1,b),dot(co2,b),determinant)/vec4<f32>(vec3<f32>(determinant),1.);
}
// The segment a plane cuts from the unit cell: midpoint, length, 1 if any.
fn shape_segment(s:vec4<f32>)->vec4<f32> {
    var pts:array<vec2<f32>,4>;var count=0u;
    for(var a=0u;a<2u;a++){
        if s[1u-a]==0.{continue;}
        for(var sign=-1;sign<=1;sign+=2){
            let side=.5*f32(sign);let t=(s.z-s[a]*side)/s[1u-a];
            if t>=-.5&&t<=.5{var p=vec2<f32>(0.);p[a]=side;p[1u-a]=t;pts[count]=p;count++;}
        }
    }
    var result=vec4<f32>(0.);
    for(var j=0u;j<count;j++){for(var k=j+1u;k<count;k++){
        let size=length(pts[j]-pts[k]);if size>result.z{result=vec4<f32>(.5*(pts[j]+pts[k]),size,1.);}
    }}return result;
}

// A cell's two contact shapes, a plane (n, alpha, 0) or a disk (centre,
// radius, 1), kept until the next transport reuses the rows for its fluxes.
fn contact_index(i:u32,l:u32)->u32{return CONTACT_ROW*cfg.n+2u*i+l;}
fn contact_diagnostic(i:u32,l:u32)->u32{return CONTACT_AUDIT_ROW*cfg.n+2u*i+l;}
struct DiskFit { shape:vec4<f32>, audit:vec4<f32> }
// One fitted cell per 16-lane workgroup: lanes 0 to 8 evaluate a disk's nine
// sample cells at once, every lane forms the same sums, and decisions pass
// through workgroup memory (agree) so barriers stay in uniform control flow.
var<workgroup> fit_area:array<vec4<f32>,9>;
var<workgroup> fit_below:array<u32,15>;
var<workgroup> fit_vote:u32;
fn agree(value:bool,lane:u32)->bool {
    if lane==0u{fit_vote=select(0u,1u,value);}
    let vote=workgroupUniformLoad(&fit_vote);workgroupBarrier();return vote==1u;
}
fn sample_cell(k:u32)->vec2<f32>{return vec2<f32>(f32(k%3u)-1.,f32(k/3u)-1.);}
// A disk for identity `label` around cell i, or `plane` where none fits the
// 3x3 amounts better (Gauss-Newton on the areas, from an algebraic circle).
fn fit_contact(i:u32,label:f32,plane:vec4<f32>,lane:u32)->DiskFit {
    var out=DiskFit(plane,vec4<f32>(0.));
    var amounts:array<f32,9>;var matrix=mat3x3<f32>();var rhs=vec3<f32>(0.);var samples=0u;var inside=true;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let shift=vec2<i32>(x,y);let p=point(i)+shift;let k=u32((y+1)*3+x+1);
        if !valid(p){inside=false;continue;}
        let j=at(p);
        let l=u32(label)-1u;let e=geometry[layer_index(j,l)];amounts[k]+=e.x;
        if e.x<=0.||e.x>=1.{continue;}
        let seg=shape_segment(geometry[shape_index(j,l)]);if seg.w==0.{continue;}
        let p3=vec3<f32>(seg.xy+vec2<f32>(shift),1.);let w=seg.z;
        matrix+=mat3x3<f32>(w*p3.x*p3,w*p3.y*p3,w*p3.z*p3);
        rhs-=w*dot(p3.xy,p3.xy)*p3;samples++;
    }}
    var plane_error=0.;
    for(var k=0u;k<9u;k++){
        let p=sample_cell(k);
        let residual=rectangle_fraction(plane.xy,plane.z,p-vec2<f32>(.5),p+vec2<f32>(.5))-amounts[k];
        plane_error+=residual*residual;
    }
    let seed=symmetric_solve(matrix,rhs);
    let center=-.5*seed.xy;let r2=dot(center,center)-seed.z;
    if !agree(inside&&!(samples<3u||plane_error<1.e-10)&&seed.w!=0.&&!(r2<=0.||r2>1024.),lane){return out;}
    let own=sample_cell(min(lane,8u));
    var disk=vec3<f32>(center,sqrt(r2));var iterations=0u;
    for(var iteration=0u;iteration<10u;iteration++){
        if lane<9u{fit_area[lane]=disk_area(disk,own);}
        workgroupBarrier();
        matrix=mat3x3<f32>();rhs=vec3<f32>(0.);var error=0.;
        for(var k=0u;k<9u;k++){
            let v=fit_area[k];let residual=v.x-amounts[k];let g=v.yzw;error+=residual*residual;
            matrix+=mat3x3<f32>(g.x*g,g.y*g,g.z*g);rhs+=g*residual;
        }
        iterations=iteration+1u;
        let solved=symmetric_solve(matrix,rhs);
        if agree(error<1.e-10||solved.w==0.,lane){break;}
        let delta=solved.xyz;var scale=min(1.,1./max(length(delta),1.e-20));var accepted=false;
        for(var trial=0u;trial<6u;trial++){
            let candidate=disk-scale*delta;
            if agree(candidate.z>0.05&&candidate.z<32.,lane){
                if lane<9u{fit_area[lane].x=disk_area(candidate,own).x;}
                workgroupBarrier();
                var next_error=0.;
                for(var k=0u;k<9u;k++){let residual=fit_area[k].x-amounts[k];next_error+=residual*residual;}
                if agree(next_error<error,lane){disk=candidate;accepted=true;break;}
            }
            scale*=.5;
        }
        if !accepted{break;}
    }
    // Then the radius alone matches the centre cell's amount, by bisection:
    // four levels of its tree a round (lane t is node t, breadth first).
    var low=length(max(abs(disk.xy)-vec2<f32>(.5),vec2<f32>(0.)));
    var high=length(abs(disk.xy)+vec2<f32>(.5));
    for(var level=0u;level<18u;level+=4u){
        let depth=min(4u,18u-level);
        if lane<(1u<<depth)-1u{
            var lo=low;var hi=high;let node=lane+1u;
            for(var d=firstLeadingBit(node);d>0u;d--){
                let middle=.5*(lo+hi);if ((node>>(d-1u))&1u)==1u{lo=middle;}else{hi=middle;}
            }
            fit_below[lane]=select(0u,1u,disk_area(vec3<f32>(disk.xy,.5*(lo+hi)),vec2<f32>(0.)).x<amounts[4]);
        }
        workgroupBarrier();
        var node=0u;
        for(var d=0u;d<depth;d++){
            let middle=.5*(low+high);
            if fit_below[node]==1u{low=middle;node=2u*node+2u;}else{high=middle;node=2u*node+1u;}
        }
        workgroupBarrier();
    }
    disk.z=.5*(low+high);
    if lane<9u{fit_area[lane]=disk_area(disk,own);}
    workgroupBarrier();
    var final_error=0.;
    for(var k=0u;k<9u;k++){let residual=fit_area[k].x-amounts[k];final_error+=residual*residual;}
    let center_area=fit_area[4].x;
    workgroupBarrier();
    if !(final_error>=plane_error){
        out.shape=vec4<f32>(disk,1.);
        out.audit=vec4<f32>(sqrt(final_error),center_area-amounts[4],f32(iterations),1.);
    }
    return out;
}
fn contact_disjoint(a:vec4<f32>,b:vec4<f32>)->bool {
    if a.w>0.&&b.w>0.{return length(a.xy-b.xy)>=a.z+b.z;}
    if a.w==0.&&b.w==0.{return true;}
    var disk=a;var plane=b;if b.w>0.{disk=b;plane=a;}
    return dot(plane.xy,disk.xy)-length(plane.xy)*disk.z>=plane.z;
}
// The identities on one outer shape's side. Only a single one takes a disk:
// a stack is one gas region, which a disk would shrink to one bubble's.
struct ContactSide { label:f32, disk:bool }
fn contact_side(i:u32,outer:vec4<f32>)->ContactSide {
    var labels=0u;var label=0.;var amount=0.;
    for(var l=0u;l<GAS_SLOTS;l++){
        let entry=geometry[layer_index(i,l)];let shape=geometry[shape_index(i,l)];
        if entry.x>0.&&entry.y>0.&&dot(shape.xy,outer.xy)>0.{labels++;label=entry.y;amount+=entry.x;}
    }
    return ContactSide(label,labels==1u&&amount>0.&&amount<1.);
}
struct ContactWork { count:atomic<u32>, cells:array<u32> }
@group(0) @binding(13) var<storage,read_write> contact_work:ContactWork;
@group(0) @binding(14) var<storage,read_write> contact_fit_groups:array<u32,3>;
var<workgroup> listed:atomic<u32>;
var<workgroup> listed_base:u32;
@compute @workgroup_size(1) fn reset_contact_work(){atomicStore(&contact_work.count,0u);}
// Writes the outer shapes of the cells whose gas stays planar (two planes are
// always disjoint) and lists those with a disk candidate for fit_contacts.
@compute @workgroup_size(128)
fn plan_contacts(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let i=id.x;
    if lane==0u{atomicStore(&listed,0u);}
    workgroupBarrier();
    var slot=0xffffffffu;
    if i<cfg.n{
        let outer=outer_gas_shapes(i);var disk=false;
        for(var s=0u;s<2u;s++){
            if outer.present[s]>0u&&contact_side(i,outer.shapes[s]).disk{disk=true;}
        }
        if disk{slot=atomicAdd(&listed,1u);}
        else{
            for(var s=0u;s<2u;s++){
                geometry[contact_index(i,s)]=outer.shapes[s];geometry[contact_diagnostic(i,s)]=vec4<f32>(0.);
            }
        }
    }
    workgroupBarrier();
    if lane==0u{listed_base=atomicAdd(&contact_work.count,atomicLoad(&listed));}
    let base=workgroupUniformLoad(&listed_base);
    if slot!=0xffffffffu{contact_work.cells[base+slot]=i;}
}
// Indirect arguments for the fits: one workgroup per listed cell.
@compute @workgroup_size(1) fn seal_contact_work(){
    contact_fit_groups[0]=atomicLoad(&contact_work.count);contact_fit_groups[1]=1u;contact_fit_groups[2]=1u;
}
@compute @workgroup_size(16)
fn fit_contacts(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let i=contact_work.cells[group.x];
    let outer=outer_gas_shapes(i);var fitted:array<vec4<f32>,2>;var diag:array<vec4<f32>,2>;
    for(var s=0u;s<2u;s++){
        fitted[s]=outer.shapes[s];let side=contact_side(i,outer.shapes[s]);
        if agree(outer.present[s]>0u&&side.disk,lane){
            let value=fit_contact(i,side.label,outer.shapes[s],lane);fitted[s]=value.shape;diag[s]=value.audit;
        }
    }
    if outer.present.x>0u&&outer.present.y>0u&&!contact_disjoint(fitted[0],fitted[1]){
        fitted=outer.shapes;diag[0].w=-1.;diag[1].w=-1.;
    }
    if lane==0u{
        for(var s=0u;s<2u;s++){
            geometry[contact_index(i,s)]=fitted[s];geometry[contact_diagnostic(i,s)]=diag[s];
        }
    }
}

fn vertex_span(v:vec2<f32>,a:u32,side:f32,row:f32,r:f32)->vec2<f32> {
    let distance=side-v[a];if abs(distance)>r{return vec2<f32>(1.e10,-1.e10);}
    let extent=sqrt(max(0.,r*r-distance*distance));return vec2<f32>(v[1u-a]+row-extent,v[1u-a]+row+extent);
}
fn span_union(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return vec2<f32>(min(a.x,b.x),max(a.y,b.y));}
// The span of a face closed to a grain's centre by (disk d within its cell)
// expanded by the grain's radius r: offset arc points and expanded vertices.
fn disk_exclusion(d:vec4<f32>,a:u32,side:f32,row:f32,r:f32)->vec2<f32> {
    var span=vec2<f32>(1.e10,-1.e10);let radius=d.z;
    let corners=array<vec2<f32>,4>(vec2<f32>(-.5,-.5),vec2<f32>(.5,-.5),vec2<f32>(.5,.5),vec2<f32>(-.5,.5));
    for(var k=0u;k<4u;k++){
        let v=corners[k];if dot(v-d.xy,v-d.xy)<=radius*radius{span=span_union(span,vertex_span(v,a,side,row,r));}
    }
    for(var b=0u;b<2u;b++){for(var sign=-1;sign<=1;sign+=2){
        let edge=.5*f32(sign);let distance=edge-d[b];if abs(distance)>radius{continue;}
        let extent=sqrt(max(0.,radius*radius-distance*distance));
        for(var t=-1;t<=1;t+=2){
            let tangent=d[1u-b]+f32(t)*extent;if tangent<-.5||tangent>.5{continue;}
            var v=vec2<f32>(0.);v[b]=edge;v[1u-b]=tangent;
            span=span_union(span,vertex_span(v,a,side,row,r));
        }
    }}
    let expanded=radius+r;let distance=side-d[a];
    if abs(distance)<=expanded{
        let extent=sqrt(max(0.,expanded*expanded-distance*distance));
        for(var t=-1;t<=1;t+=2){
            var p=vec2<f32>(0.);p[a]=side;p[1u-a]=d[1u-a]+f32(t)*extent;
            let nearest=p-(r/expanded)*(p-d.xy);
            if all(nearest>=vec2<f32>(-.5))&&all(nearest<=vec2<f32>(.5)){
                span=span_union(span,vec2<f32>(p[1u-a]+row));
            }
        }
    }
    return span;
}
struct Passage { largest:vec2<f32>, total:f32 }
// The part of face f free for the centre of a grain of radius r (cells; 0
// for the liquid), beside the contact shapes of the cells around it and the
// tank's wall. Curved gas can cut a face's middle: every free piece counts.
fn curved_passage(f:u32,r:f32)->Passage {
    let i=f/2u;let a=f%2u;let p=point(i);let q=p+axis(a);
    if !valid(q){return Passage(vec2<f32>(.5,-.5),0.);}
    var spans:array<vec2<f32>,12>;var count=0u;
    for(var row=-1;row<=1;row++){
        if r==0.&&row!=0{continue;}
        for(var column=0;column<2;column++){
            let cell=p+column*axis(a)+row*axis(1u-a);let side=.5-f32(column);
            if !valid(cell){
                if row<0{spans[count]=vec2<f32>(-.5,-.5+r);}else{spans[count]=vec2<f32>(.5-r,.5);}count++;continue;
            }
            let j=at(cell);if cells[j].volume.gas<=0.{continue;}
            for(var s=0u;s<2u;s++){
                let shape=geometry[contact_index(j,s)];var span=vec2<f32>(1.e10,-1.e10);
                if shape.w>0.{span=disk_exclusion(shape,a,side,f32(row),r);}
                else if represented_gas_area(shape){span=polygon_exclusion(shape,a,side,f32(row),r);}
                span=vec2<f32>(max(-.5,span.x),min(.5,span.y));
                if span.x<span.y{spans[count]=span;count++;}
            }
        }
    }
    for(var j=1u;j<count;j++){
        let v=spans[j];var k=j;
        while k>0u{if spans[k-1u].x<=v.x{break;}spans[k]=spans[k-1u];k--;}
        spans[k]=v;
    }
    var out=Passage(vec2<f32>(-.5,-.5),0.);var cursor=-.5;
    for(var j=0u;j<count;j++){
        let span=spans[j];let width=max(0.,span.x-cursor);out.total+=width;
        if width>out.largest.y-out.largest.x{out.largest=vec2<f32>(cursor,span.x);}
        cursor=max(cursor,span.y);
    }
    let width=max(0.,.5-cursor);out.total+=width;
    if width>out.largest.y-out.largest.x{out.largest=vec2<f32>(cursor,.5);}
    return out;
}
