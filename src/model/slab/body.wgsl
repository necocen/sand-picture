// Packed bodies: sand at CRITICAL_PACKING or denser, which moves as a
// skeleton with its pore water or stands, and a body's last layer, which
// fills part of its cell. body_faces records the layers (BODY_ROW) in the
// Slab stage, before the update's mechanics and transport read them.

// A face inside a packed skeleton: both cells at CRITICAL_PACKING or denser.
// Its grains have no flux relative to the water: a jammed skeleton does not
// settle through its pore water, and any residual drift piles up where the
// raft blocks the slab's base.
fn skeleton_face(f:u32)->bool{
    let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q){return false;}
    let a=cells[i];let b=cells[at(q)];let ga=grains(a);let gb=grains(b);
    let wa=max(a.volume.water,0.)+ga.x+ga.y;let wb=max(b.volume.water,0.)+gb.x+gb.y;
    return wa>0.&&wb>0.&&ga.x+ga.y>=CRITICAL_PACKING*wa&&gb.x+gb.y>=CRITICAL_PACKING*wb;
}
// A packed cell that the plates hold still (stuck and slower than CREEP by
// HOLD_ROW) stands: its jammed grains cross none of its faces, grains
// arriving from a neighbour stop at its face, and the water seeps through
// as through a sieve. A sliding packed cell carries its pore water with it.
// Every face is closed, toward a loose neighbour too: grains riding the
// pore water the raft pumps through the slab stay behind where it leaves.
fn standing(i:u32)->bool{
    let c=cells[i];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if wet<=0.||solid<CRITICAL_PACKING*wet{return false;}
    let state=geometry[HOLD_ROW*cfg.n+i];return state.z==0.&&state.w<CREEP;
}
fn standing_face(f:u32)->bool{
    let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q){return false;}
    return standing(i)||standing(at(q));
}

fn body_speed(c:Cell)->vec2<f32>{
    let g=grains(c);let s=g.x+g.y;if s<=0.{return vec2<f32>(0.);}
    return (c.momentum[1]+c.momentum[2])/s;
}
// Per cell, geometry[BODY_ROW n + i] = (f, side, left, right). A cell below
// CRITICAL_PACKING with a grain layer or more that sits against a packed
// neighbour moving with it (within TEETH_STILL), or rests on a packed cell
// below it, is that body's last layer: its grains fill the fraction f of
// its wet part at the neighbour's packing, on the neighbour's side
// (2a + 1 toward +axis(a), 2a toward -axis(a)), and the rest is free water;
// (1, -1) for any other cell. A layer resting on its body touches it
// whatever its speed: told by its speed, a funnel's surface would run down
// its wall as a suspension once it started to slide. left (.z) and right
// (.w) are 1 in open water where grains shed by the sand on that side of
// the cell fall away (falls_away).
@compute @workgroup_size(128) fn body_faces(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let c=cells[i];let g=grains(c);let solid=g.x+g.y;
    let wet=max(c.volume.water,0.)+solid;var out=vec4<f32>(1.,-1.,0.,0.);
    // Trace grains are no layer: their diameter-weighted sum flushes to zero
    // and a layer that thin reads as a skeleton of unbounded suction.
    if solid>=TRACE_VOLUME&&wet>0.&&solid<CRITICAL_PACKING*wet{
        let layer=dot(g,GRAIN_DIAMETER)/(solid*cfg.h)*CRITICAL_PACKING;
        if solid>=layer{
            let p=point(i);let v=body_speed(c);var packing=0.;var side=-1.;
            for(var a=0u;a<2u;a++){for(var d=-1;d<=1;d+=2){
                let q=p+d*axis(a);if !valid(q){continue;}
                let cj=cells[at(q)];let gj=grains(cj);let sj=gj.x+gj.y;let wj=max(cj.volume.water,0.)+sj;
                if wj<=0.||sj<CRITICAL_PACKING*wj{continue;}
                let rests=a==1u&&d<0;
                if !rests&&length(body_speed(cj)-v)>TEETH_STILL{continue;}
                let pj=sj/wj;if pj>packing{packing=pj;side=f32(2u*a+select(0u,1u,d>0));}
            }}
            if side>=0.{out=vec4<f32>(min(1.,solid/(packing*wet)),side,0.,0.);}
        }
    }
    if open_water(c){
        let p=point(i);let left=p-axis(0u);let right=p+axis(0u);
        if valid(left)&&breach_donor(cells[at(left)]){out.z=select(0.,1.,falls_away(p,1));}
        if valid(right)&&breach_donor(cells[at(right)]){out.w=select(0.,1.,falls_away(p,-1));}
    }
    scratch[BODY_ROW*cfg.n+i]=out;
}
// The part (cells) of cell i's wet volume that its grains fill at their
// body's packing: all of it, or the fraction f for a body's last layer. The
// mechanics reads the layer's dilatancy, yield and plate friction at the
// body's packing, not at the mean of the dense part and the free water
// beside it; that dilatancy is suction, which holds and does not push.
fn dense_volume(i:u32)->f32{
    let c=cells[i];let wet=max(c.volume.water,0.)+grains(c).x+grains(c).y;
    let entry=scratch[BODY_ROW*cfg.n+i];
    if entry.y<0.||entry.x>=1.||entry.x<=0.{return wet;}
    return entry.x*wet;
}
// Whether `donor` is a body's last layer that does not fill its cell and
// `f` is one of its faces other than the one toward its body.
fn layer_far_face(f:u32,donor:u32)->bool{
    let entry=scratch[BODY_ROW*cfg.n+donor];if entry.y<0.||entry.x>=1.{return false;}
    let a=f%2u;let body=u32(entry.y);if body/2u!=a{return true;}
    let face_plus=donor==f/2u;let body_plus=body%2u==1u;
    return face_plus!=body_plus;
}
// The most grains (fine, coarse; cells per substep) the last layer `donor`
// passes through its face `f`: a packed surface under water advances only
// as its body moves or as it breaches. Unbounded (1e9) for a cell that is
// no layer and through the face toward the body, into which it may compact.
fn layer_supply(f:u32,donor:u32,interval:f32)->vec2<f32>{
    let entry=scratch[BODY_ROW*cfg.n+donor];if entry.y<0.||entry.x>=1.{return vec2<f32>(1e9);}
    let a=f%2u;let body=u32(entry.y);let body_axis=body/2u;let body_plus=body%2u==1u;
    let face_plus=donor==f/2u;
    if body_axis==a&&face_plus==body_plus{return vec2<f32>(1e9);}
    // The layer moves with its body: it passes what the body's own grains
    // move toward the face, nothing for a still body.
    let q=point(donor)+select(-axis(body_axis),axis(body_axis),body_plus);
    let cb=cells[at(q)];let gb=grains(cb);var supply=vec2<f32>(0.);
    for(var phase=0u;phase<2u;phase++){
        if gb[phase]<=0.{continue;}
        let v=cb.momentum[phase+1u][a]/gb[phase];
        let toward=select(max(-v,0.),max(v,0.),face_plus);
        supply[phase]=toward*interval*gb[phase];
    }
    // A layer resting on its body also slides sideways at its own speed
    // where the mechanics, reading it at the body's packing (dense_volume),
    // finds the slope too steep: an avalanche down a funnel's wall.
    if body_axis==1u&&!body_plus&&a==0u{
        let c=cells[donor];let g=grains(c);
        for(var phase=0u;phase<2u;phase++){
            if g[phase]<=0.{continue;}
            let v=c.momentum[phase+1u].x/g[phase];
            supply[phase]+=select(max(-v,0.),max(v,0.),face_plus)*interval*g[phase];
        }
    }
    // Breaching: the layer's grains leave through its water face as fast as
    // the face recedes for its inclination (a face looking down as a roof,
    // a side face by the packing's slope), whether or not the plates hold
    // the body, and only where they fall away: through a face looking down,
    // or sideways by the receiver's BODY_ROW. Into water standing on packed
    // sand (a step of a funnel's wall) they rest on the step and buttress
    // the face: between the plates such a wall stands far steeper than the
    // angle of repose (Taberlet et al. 2003).
    let c=cells[donor];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    let dense=max(scratch[BODY_ROW*cfg.n+donor].x,1e-3)*wet;let phi=min(solid/dense,CLOSE_PACKING);
    var falls=a==1u&&!face_plus;
    if a==0u{
        let receiver=point(donor)+select(-axis(0u),axis(0u),face_plus);
        if valid(receiver){let record=scratch[BODY_ROW*cfg.n+at(receiver)];falls=select(record.w,record.z,face_plus)>0.;}
    }
    if solid>0.&&falls{
        var up=-1.;
        if a==0u{up=side_face_up(point(donor),point(donor)+select(-axis(0u),axis(0u),face_plus));}
        let face_speed=breaching_rate(g,white_powder(cells[donor]),phi)*inclination_factor(up,grain_friction(g));
        supply+=face_speed*interval*g/dense;
    }
    return supply;
}
