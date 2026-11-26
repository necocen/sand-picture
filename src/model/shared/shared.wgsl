// Declarations shared by every stage: the configuration, the records of a
// cell, a face and a transport with their buffers, and the cell accessors.

// cfg.gravity is the downward gravity in the plane of the glass (-y),
// cfg.lateral is the apparent gravity along +x (shaking or sideways tilt), and
// cfg.plate is the out-of-plane gravity pressing grains against the glass
// plates (z). The material table's speeds are at the standard gravity.
struct Config { nx:u32, ny:u32, n:u32, table_side:u32, dt:f32, h:f32, gravity:f32,substeps:u32, lateral:f32, air_delta:f32, air_seed:u32, plate:f32 }
const STANDARD_GRAVITY:f32=9.80665;
// The white is its grit and powder (material.rs). Of its volume `total` is
// powder and `free` is free powder, single grains dispersed in the water.
// Bound powder moves and settles with the grit, free powder at its own speed,
// the white at their volume mean (white_speed). Its packing drags as a grain
// of the Sauter mean of the grit and all its powder (Kozeny-Carman), not as
// the mean of the speeds (packing_speeds, at the standard gravity).
struct PowderShare {free:f32,total:f32}
fn white_speed(powder:PowderShare)->f32{return mix(TERMINAL_SPEED.x,POWDER_SPEED,powder.free);}
fn settling_speeds(powder:PowderShare)->vec2<f32>{return vec2<f32>(white_speed(powder),TERMINAL_SPEED.y)*(cfg.gravity/STANDARD_GRAVITY);}
fn packing_speeds(powder:PowderShare)->vec2<f32>{
    let r=mix(1.,GRAIN_DIAMETER.x/POWDER_DIAMETER,powder.total);
    return vec2<f32>(TERMINAL_SPEED.x/(r*r),TERMINAL_SPEED.y);
}
fn plate_gravity()->f32{return cfg.plate;}
struct PhaseVolumes {water:f32,fine:f32,coarse:f32,gas:f32}
// `load` is the part of `pressure` that pushes the neighbours: the reaction
// to the compression the flow imposes. The rest, and `suction` (the pore
// water's in a dilating skeleton, dilatancy_suction), only hold: they count
// in the friction and the yield, and `suction` never in the dilatancy law.
struct GrainStress {pressure:f32,normal:f32,shear:f32,load:f32,suction:f32}
// The white's powder in a cell (cell volume fractions): `mobile` within the
// free white grains, `bed` within the deposit's white, `free` the part of
// `mobile` dispersed as single grains. Each is at most what it is part of.
struct CellPowder {mobile:f32,bed:f32,free:f32}
// `fine` is the white sand, `coarse` the black; momentum is volume times
// velocity. `bed` is the deposit's white, black, region and pore water.
struct Cell {@align(16) volume:PhaseVolumes,momentum:array<vec2<f32>,4>,stress:GrainStress,powder:CellPowder,bed:vec4<f32>}
// What the transport carries for a cell or a group: phase volumes and
// momenta, the grains' advected stress, the white's powder (all and free).
struct GroupPayload {volume:vec4<f32>,momentum:array<vec2<f32>,4>,stress:vec2<f32>,powder:vec2<f32>}
// A face's flux. `powder` is the white's, all and free; until the limit pass
// writes the free powder's flux, .y is its drift relative to the white.
struct Transport {volume:vec4<f32>,momentum:array<vec2<f32>,4>,forward:vec2<f32>,reverse:vec2<f32>,gas_reverse:vec2<f32>,powder:vec2<f32>}
struct Face {
    composition:vec4<f32>, // intrinsic phase fractions and wet dual volume
    velocity:vec4<f32>, // unconstrained velocities/mean; after projection xyz=slip, w=gas velocity
    response:vec4<f32>, // velocity response to pressure difference; flux response
    grain_response:vec4<f32>, // xyz=grain-pressure response; w=intrinsic relative grain mobility
    volume:vec4<f32>, // corrected phase point velocities and common point velocity
    water_momentum:vec4<f32>, // relative mechanics: drive, coupling L/D, closure mode, inverse resistance 1/D
    coarse_momentum:vec4<f32>, // common energy weight, preferred velocity, wet aperture, shear response bound
    support:vec4<f32>, // supported fine/coarse fractions, pressure multiplier, closed grain passage
    displacement:vec4<f32>, // finite remap bed.xy and ambient-grain.zw currents
    previous_displacement:vec4<f32>, // preceding corrected hydraulic state
}
// `geometry` holds what an update leaves for the next beside the cells, and
// `scratch` what one update's kernels write before they read it: rows of the
// cell count cfg.n, one vec4 an entry, each use's first row a constant
// NAME_ROW (rows.rs).
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(2) var<storage,read_write> scratch:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> faces:array<Face>;
@group(0) @binding(4) var<storage,read_write> geometry:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> table:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> potential:array<f32>;
// Densities (kg/m3): the water, the white grains, the black grains.
const RHO=vec3<f32>(997.,2450.,4700.);
// The gap between the plates (m) and the water's viscosity (Pa s).
const PLATE_GAP:f32=.001;
const WATER_VISCOSITY:f32=.00094;
// Integrated gradient across the two thin sheared layers of a grain plug
// that slides between the plates (2/b).
const GAP_SHEAR=2./.001;
// The grid's dimensions live in cfg.nx and cfg.ny so the tank can rotate by
// exact 90-degree quarter turns (216x144 <-> 144x216) in place.
fn point(i:u32)->vec2<i32> {return vec2<i32>(vec2<u32>(i%cfg.nx,i/cfg.nx));}
fn valid(p:vec2<i32>)->bool {return all(p>=vec2<i32>(0)) && p.x<i32(cfg.nx) && p.y<i32(cfg.ny);}
fn at(p:vec2<i32>)->u32 {return u32(p.y)*cfg.nx+u32(p.x);}
fn cell(p:vec2<i32>)->Cell {return cells[at(clamp(p,vec2<i32>(0),vec2<i32>(i32(cfg.nx)-1,i32(cfg.ny)-1)))];}
fn axis(a:u32)->vec2<i32> {return select(vec2<i32>(1,0),vec2<i32>(0,1),a==1u);}
// The tank's walls are rigid mirrors for the grain fields: a point outside
// reads its mirror cell, with the velocity normal to the wall and the shear
// stress reversed (no flow through the wall; its traction is face_friction).
fn inside(p:vec2<i32>)->vec2<i32> {return clamp(p,vec2<i32>(0),vec2<i32>(i32(cfg.nx)-1,i32(cfg.ny)-1));}
fn reflected(p:vec2<i32>,v:vec2<f32>)->vec2<f32> {return select(v,-v,inside(p)!=p);}
fn grains(c:Cell)->vec2<f32> {return max(vec2<f32>(0.),vec2<f32>(c.volume.fine,c.volume.coarse));}
// The powder shares of a cell's free white grains.
fn white_powder(c:Cell)->PowderShare {
    if c.volume.fine<=0. {return PowderShare(0.,0.);}
    let total=clamp(c.powder.mobile/c.volume.fine,0.,1.);
    return PowderShare(clamp(c.powder.free/c.volume.fine,0.,total),total);
}
// The shares across a face: of its two cells' free white grains together.
fn face_powder(l:Cell,r:Cell)->PowderShare {
    let white=max(l.volume.fine,0.)+max(r.volume.fine,0.);if white<=0. {return PowderShare(0.,0.);}
    let total=clamp((max(l.powder.mobile,0.)+max(r.powder.mobile,0.))/white,0.,1.);
    return PowderShare(clamp((max(l.powder.free,0.)+max(r.powder.free,0.))/white,0.,total),total);
}
// Below this share of a cell an amount is a trace, under the float
// resolution of a full cell (2^-24): upwind transport leaks every phase into
// its neighbours in geometrically falling amounts. Trace gas returns to the
// liquid (clear_trace_gas); trace grains take no part in the contact problem.
const TRACE_VOLUME:f32=1e-8;
fn liquid(c:Cell)->f32 {return free_wet(c)+c.bed.z;}
fn fractions(c:Cell)->vec3<f32> {return vec3<f32>(max(0.,c.volume.water),grains(c));}
// The grains' concentrations in cell p's wet part, reconstructed towards the
// face a carrier of Courant number `courant` crosses: the wet mixture's
// composition, not the cell's volumes, which change as gas sweeps past.
fn wet_concentration(p:vec2<i32>,a:u32,courant:f32)->vec2<f32> {
    let middle=cell(p);let wet=free_wet(middle);if wet<=0.{return vec2<f32>(0.);}
    let center=grains(middle)/wet;
    let left=cell(p-axis(a));let right=cell(p+axis(a));
    if wet<.5||free_wet(left)<.5||free_wet(right)<.5{return center;}
    let l=grains(left)/free_wet(left);let r=grains(right)/free_wet(right);
    let c3=vec3<f32>(1.-center.x-center.y,center);
    let l3=vec3<f32>(1.-l.x-l.y,l);let r3=vec3<f32>(1.-r.x-r.y,r);
    let slope=.5*(r-l);let s3=vec3<f32>(-slope.x-slope.y,slope);
    let lo=max(vec3<f32>(0.),min(c3,min(l3,r3)));let hi=min(vec3<f32>(1.),max(c3,max(l3,r3)));
    var factor=1.;for(var k=0u;k<3u;k++){
        if s3[k]!=0.{factor=min(factor,max(0.,2.*min(c3[k]-lo[k],hi[k]-c3[k]))/abs(s3[k]));}
    }
    // A shared limiter preserves the sum of all three constituent slopes.
    return center+.5*select(-1.,1.,courant>=0.)*max(0.,1.-abs(courant))*factor*slope;
}
fn volume_current(f:u32)->f32 {
    let i=f/2u;let x=i%cfg.nx;let y=i/cfg.nx;let stride=cfg.nx+1u;
    if f%2u==0u{return (potential[(y+1u)*stride+x+1u]-potential[y*stride+x+1u])/cfg.h;}
    return (potential[(y+1u)*stride+x]-potential[(y+1u)*stride+x+1u])/cfg.h;
}
