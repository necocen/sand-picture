// The slab's teeth: firm sand hanging below the slab's underside, and the
// throat under a tooth, which a bubble's gas does not enter. teeth_underside
// runs in the Slab stage, at the start of an update, for its transport.

// Sand a bubble's top rests under: this share of a cell's volume in grains,
// free or deposited.
const TEETH_SAND:f32=.2;
// Firm sand is half or more of its wet part and slower than this (m/s): a
// bubble pushes a falling curtain or a loosened column aside. body_faces
// tells a layer that moves with its body by the same speed.
const TEETH_STILL:f32=2e-3;
// A tooth hangs at least this far (m) below the underside the gas rests
// under: half a pressed bubble's height, the radius of its edge's own
// curvature, which absorbs shallower roughness. Two rows on either grid.
const TEETH_DEPTH:f32=1.8e-3;
// The depth (m) of the throat under a tooth: the raft's thickness, a
// pressed bubble's height and the loose layer it presses against.
const TEETH_SHADOW:f32=5.1e-3;

fn teeth_sandy(c:Cell)->bool{
    return max(c.volume.fine,0.)+max(c.volume.coarse,0.)+max(c.bed.x,0.)+max(c.bed.y,0.)>=TEETH_SAND;
}
fn teeth_firm(c:Cell)->bool{
    let g=grains(c);let sand=g.x+g.y+max(c.bed.x,0.)+max(c.bed.y,0.);
    let wet=max(c.volume.water,0.)+sand;
    if wet<=0.||sand<.5*wet{return false;}
    var speed=0.;
    if g.x>0.{speed=max(speed,length(c.momentum[1])/g.x);}
    if g.y>0.{speed=max(speed,length(c.momentum[2])/g.y);}
    return speed<TEETH_STILL;
}
fn teeth_column(col:i32)->vec2<f32>{
    if col<0||col>=i32(cfg.nx){return vec2<f32>(-1.);}
    return scratch[TEETH_ROW*cfg.n+u32(col)].xy;
}
// Per column, geometry[TEETH_ROW n + col] = (the lowest row of the sandy run
// hanging from the ceiling, the lowest row of the firm sand unbroken from
// its top), or (-1, -1) with no sand in the tank's upper half. Water above
// the run (a crater) is skipped; one loose cell over sand stays in the run.
@compute @workgroup_size(128) fn teeth_underside(@builtin(global_invocation_id) id:vec3<u32>){
    let col=id.x;if col>=cfg.nx{return;}
    var y=i32(cfg.ny)-1;
    while y>=0&&!teeth_sandy(cells[at(vec2<i32>(i32(col),y))]){y-=1;}
    if y<0||y<i32(cfg.ny)/2{scratch[TEETH_ROW*cfg.n+col]=vec4<f32>(-1.,-1.,0.,0.);return;}
    var firm=-1.;var run=true;
    for(var r=y;r>=0;r--){
        let c=cells[at(vec2<i32>(i32(col),r))];
        if !teeth_sandy(c){
            if r==0||!teeth_sandy(cells[at(vec2<i32>(i32(col),r-1))]){break;}
            run=false;y=r;continue;
        }
        if run&&teeth_firm(c){firm=f32(r);}else{run=false;}
        y=r;
    }
    scratch[TEETH_ROW*cfg.n+col]=vec4<f32>(f32(y),firm,0.,0.);
}
// Whether a bubble's gas is kept from crossing from `source` into `dest`:
// `dest` lies within TEETH_SHADOW below a tooth, the firm underside of its
// column, that hangs TEETH_DEPTH or more below the underside the gas rests
// under. That is the one over `source` for gas entering from the side; for
// gas rising from the cell below, which would have to fit a channel
// narrower than its own height, the higher of the two beside the tooth.
// A bubble spreads along the ceiling it rests against but cannot creep
// under a tooth: bending its edge round the tooth's corner into the throat
// takes a curvature pressure (sigma over the throat's half width, 42 Pa for
// one standard cell) that neither its buoyancy head (35 Pa for its height)
// nor the row's compression (about 25 Pa) provides, and the wet sand's
// pore-scale roughness holds its contact line. The joint under a tooth
// stays water while the tooth lasts.
// The held gas has no candidate across the face; the water of its cell
// takes the gas's share within its budget, like every other increase (an
// unbudgeted share creates volume in the throat), and where the water
// cannot give it the bubble crosses at its upwind share.
fn teeth_shadow(source:vec2<i32>,dest:vec2<i32>)->bool{
    let tooth=teeth_column(dest.x).y;if tooth<0.{return false;}
    var over=-1.;
    if source.y==dest.y{over=teeth_column(source.x).x;}
    else if dest.y==source.y+1&&dest.x==source.x{over=max(teeth_column(dest.x-1).x,teeth_column(dest.x+1).x);}
    if over<0.{return false;}
    return (over-tooth)*cfg.h>=TEETH_DEPTH&&f32(dest.y)<tooth&&(tooth-f32(dest.y))*cfg.h<=TEETH_SHADOW;
}
