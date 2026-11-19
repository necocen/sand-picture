// Whether a grain's centre can cross a face, beside the gas and beside the
// bed. The mechanics caches the outcome as the face's grain mode, which its
// solve and the transport's sieve read.

fn exclusion_vertex(interval:ptr<function,vec2<f32>>,v:vec2<f32>,side:f32,r:f32) {
    let distance=side-v.x;
    if abs(distance)<=r{
        let extent=sqrt(max(0.,r*r-distance*distance));
        (*interval).x=min((*interval).x,v.y-extent);(*interval).y=max((*interval).y,v.y+extent);
    }
}
fn exclusion_edge(interval:ptr<function,vec2<f32>>,v:vec2<f32>,next:vec2<f32>,side:f32,r:f32) {
    let d=next-v;let size=length(d);
    if size>0.&&d.x!=0.{
        let offset=r*vec2<f32>(-d.y,d.x)/size;
        for(var sign=-1;sign<=1;sign+=2){
            let start=v+f32(sign)*offset;let t=(side-start.x)/d.x;
            if t>=0.&&t<=1.{
                let y=start.y+t*d.y;
                (*interval).x=min((*interval).x,y);(*interval).y=max((*interval).y,y);
            }
        }
    }
}
struct ClipWalk { interval:vec2<f32>, first:vec2<f32>, previous:vec2<f32>, count:u32 }
fn clip_vertex(walk:ptr<function,ClipWalk>,v:vec2<f32>,side:f32,r:f32) {
    exclusion_vertex(&(*walk).interval,v,side,r);
    if (*walk).count>0u{exclusion_edge(&(*walk).interval,(*walk).previous,v,side,r);}else{(*walk).first=v;}
    (*walk).previous=v;(*walk).count++;
}
// The span of the face at `side` closed to a grain's centre by a cell's gas
// half-plane: its polygon in the cell expanded by the grain's radius r, with
// ends on offset edges or vertex circles. Clip before expanding: a half-plane
// extrapolated outside its cell both misses and invents obstructions.
fn polygon_exclusion(shape:vec4<f32>,a:u32,side:f32,row:f32,r:f32)->vec2<f32> {
    let corners=array<vec2<f32>,4>(vec2<f32>(-.5,-.5),vec2<f32>(.5,-.5),
        vec2<f32>(.5,.5),vec2<f32>(-.5,.5));
    var walk=ClipWalk(vec2<f32>(1.e10,-1.e10),vec2<f32>(0.),vec2<f32>(0.),0u);
    for(var k=0u;k<4u;k++){
        let p=corners[k];let q=corners[(k+1u)%4u];
        let sp=dot(shape.xy,p)-shape.z;let sq=dot(shape.xy,q)-shape.z;
        if sp<=0.{clip_vertex(&walk,vec2<f32>(p[a],p[1u-a]+row),side,r);}
        if (sp<0.&&sq>0.)||(sp>0.&&sq<0.){
            let v=p+(sp/(sp-sq))*(q-p);
            clip_vertex(&walk,vec2<f32>(v[a],v[1u-a]+row),side,r);
        }
    }
    if walk.count>0u{exclusion_edge(&walk.interval,walk.previous,walk.first,side,r);}
    return walk.interval;
}
// Whether a grain of this diameter (m) can cross face f beside the gas. Gas
// too little for a layer one grain thick across the face, both cells
// together, is a trace left among the grains by passing bubbles, not a wall.
fn grain_crosses(f:u32,diameter:f32)->bool {
    let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q){return false;}
    let gas=vec2<f32>(cells[i].volume.gas,cells[at(q)].volume.gas);
    if gas.x>=1.||gas.y>=1.{return false;}
    if (gas.x+gas.y)*cfg.h<diameter{return true;}
    return curved_passage(f,diameter/(2.*cfg.h)).total>0.;
}
// The room (m) beside the bed at face f: the thinnest free wet layer of its
// cells that hold bed (unbounded if neither does). The layer lies on the bed
// or is a film trapped under it, and a grain wider than it crosses none of
// the cell's faces, whatever the gas; the capture takes the sand above from
// the cell's top. Grains pressed into such a film would read as compressing
// its tiny volume, and the grain pressure would diverge.
fn bed_room(f:u32)->f32 {
    let i=f/2u;let q=point(i)+axis(f%2u);var room=1e10;
    if !valid(q){return room;}
    for(var k=0u;k<2u;k++){
        let cell=cells[select(i,at(q),k==1u)];
        if cell.bed.z>0.{room=min(room,free_wet(cell)*cfg.h);}
    }
    return room;
}
// The face's grain mode, cached in water_momentum.z: 0 the sands cross, 1 the
// coarse follows the common flow, 2 both do. Only the sands present count.
fn grain_face_mode(f:u32,c:vec3<f32>)->f32 {
    let room=bed_room(f);
    if room>=GRAIN_DIAMETER.y&&grain_crosses(f,GRAIN_DIAMETER.y){return 0.;}
    if room<GRAIN_DIAMETER.x||!grain_crosses(f,GRAIN_DIAMETER.x){return 2.;}
    if c.z<=0.{return 0.;}
    if c.x<=0.||c.y<=0.{return 2.;}
    return 1.;
}
fn phase_locked(f:u32)->bool {return faces[f].water_momentum.z==2.;}
fn coarse_locked(f:u32)->bool {return faces[f].water_momentum.z==1.;}
