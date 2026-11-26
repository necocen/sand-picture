// The gas's high flux by identity (layer_flux, the second kernel of a
// subdivision): a bubble's gas crosses a face with its own bubble, swept
// from its plane. Also the bubbles smaller than a cell or than the gap, and
// the return of trace gas to the liquid (clear_trace_gas, stage TraceGas).

// The width (cells) the gas sweeps across face f in a subdivision.
fn gas_carrier(f:u32)->f32{
    if internal_face(f){return 0.;}return faces[f].velocity.w*cfg.dt/(carry_divisions()*cfg.h);
}
// scratch[flux_index(f, l)] = (forward amount, identity, backward amount,
// identity) of identity l + 1: one amount from each side's cell, since the
// gas and the common flow may cross a face in opposite directions.
fn source_gas_flux(f:u32,l:u32,origin:u32)->f32{return scratch[flux_index(f,l)][2u*origin];}
fn put_source_gas_flux(f:u32,l:u32,origin:u32,value:f32){scratch[flux_index(f,l)][2u*origin]=value;}
// The gas crossing face f (forward, backward; both nonnegative): the net
// flux where the gas moves with the common flow, else each side's sum.
fn gas_sweeps(f:u32)->vec2<f32>{
    if faces[f].velocity.w==faces[f].volume.w{
        let gas=transport[f].volume.w;return vec2<f32>(max(gas,0.),max(-gas,0.));
    }
    var amount=vec2<f32>(0.);
    for(var l=0u;l<GAS_SLOTS;l++){let v=scratch[flux_index(f,l)];amount+=vec2<f32>(v.x,-v.z);}
    return amount;
}
// A bubble smaller than a cell: no neighbour holds a grain layer of gas.
fn lone_blob_cell(p:vec2<i32>)->bool{
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let q=p+vec2<i32>(x,y);
        if (x!=0||y!=0)&&valid(q)&&cells[at(q)].volume.gas*cfg.h>=GRAIN_LAYER{return false;}
    }}
    return true;
}
// The slip (m/s) of a bubble smaller than the gap between the plates: a
// sphere below the grid, which rises through the water that the flow moves
// around it at its own terminal speed (material.rs, bubble_rise_speeds),
// against the apparent gravity and in proportion to it. Only for a bubble
// alone in a suspension (lone_blob_cell, no deposit, grains below the loose
// packing): the speed of the sphere of its identity's gas over the 3x3,
// hindered as grains are by the voidage to the power HINDRANCE. Its gas
// crosses the faces ahead that much faster than the flow (gas_velocity)
// and lies against them (label_plane_normal).
fn bubble_slip(i:u32)->vec2<f32>{
    // Gas for a sphere as large as the gap (pi b^2 / 6 h^2 of the cell) is
    // a bubble against the plates, or bubbles pressed together.
    let gas=cells[i].volume.gas;
    if gas<=0.||gas*6.*cfg.h*cfg.h>=3.14159265*PLATE_GAP*PLATE_GAP{return vec2<f32>(0.);}
    let c=cells[i];let p=point(i);
    if max(c.bed.x,0.)+max(c.bed.y,0.)>0.||!lone_blob_cell(p){return vec2<f32>(0.);}
    let g=grains(c);let solid=g.x+g.y;let water=max(c.volume.water,0.);let wet=water+solid;
    if wet<=0.||solid>=LOOSE_PACKING*wet{return vec2<f32>(0.);}
    var label=0.;var largest=0.;
    for(var l=0u;l<GAS_SLOTS;l++){let entry=geometry[layer_index(i,l)];if entry.x>largest{largest=entry.x;label=entry.y;}}
    if largest<=0.{return vec2<f32>(0.);}
    var volume=0.;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let q=p+vec2<i32>(x,y);if valid(q){volume+=label_fraction(q,label);}}}
    // The sphere's diameter over the gap b, its volume the gas times h^2 b.
    if volume<=0.||water<=0.{return vec2<f32>(0.);}
    let ratio=pow(6.*volume*cfg.h*cfg.h/(3.14159265*PLATE_GAP*PLATE_GAP),1./3.);
    if ratio>=1.{return vec2<f32>(0.);}
    let level=ratio*f32(BUBBLE_LEVELS);let k=min(u32(level),BUBBLE_LEVELS-1u);
    var rise=BUBBLE_RISE;
    let speed=mix(rise[k],rise[k+1u],level-f32(k))*pow(water/wet,HINDRANCE);
    return speed*vec2<f32>(-cfg.lateral,cfg.gravity)/STANDARD_GRAVITY;
}
// A cell whose water is no more than the pore water of its grains at the
// close packing: a bubble cannot drain a packing further without entering
// its pores, against an entry pressure of kilopascals.
fn close_drained(c:Cell)->bool{
    let g=grains(c);let solid=g.x+g.y;
    return solid>0.&&max(c.volume.water,0.)<=solid*(1.-CLOSE_PACKING)/CLOSE_PACKING;
}
// Deposit, or grains that touch: their pores are narrower than a bubble.
fn pores_only(c:Cell)->bool{
    if max(c.bed.x,0.)+max(c.bed.y,0.)>0.{return true;}
    let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    return wet>0.&&solid>=LOOSE_PACKING*wet;
}
// The velocity of the bubble of identity `label` at p: its gas's over the
// 3x3, weighted by its amounts. A bubble's gas moves with its own bubble,
// not with the flow beside it, and crosses a face no faster than this
// (layer_flux): the streams' 1-3 cm/s give a capillary number mu U / sigma
// of about 1e-3, three decades below break-up, so the flow cannot tear a
// bubble. With the cell's one gas velocity a puddle's edge would follow
// the stream beside it, or the bubble it meets in a cell, and leave pieces
// behind; the rims that the grid smears into a small bubble's channel are
// no exception.
fn identity_velocity(p:vec2<i32>,label:f32)->vec2<f32>{
    var momentum=vec2<f32>(0.);var mass=0.;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let q=p+vec2<i32>(x,y);if !valid(q){continue;}
        let c=cells[at(q)];let g=c.volume.gas;if g<=1e-30{continue;}
        let share=label_fraction(q,label);momentum+=share*c.momentum[3]/g;mass+=share;
    }}
    if mass<=0.{return vec2<f32>(0.);}
    // A bubble whose buoyancy the frame's top or the slab's sand carries
    // (labels.wgsl, anchored) moves only by the creep its soap films allow
    // along that contact: the streams' inflow cannot slide it.
    if label_anchored_at(at(p),u32(label)-1u){return label_anchored_velocity(at(p),u32(label)-1u,momentum/mass);}
    return momentum/mass;
}
// The high flux of each gas identity through face f, and their sum in
// transport[f].volume.w: the gas of the upwind cell's planes within the
// width that its own bubble sweeps (parallel_slab).
@compute @workgroup_size(128) fn layer_flux(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let f=id.x;if f>=2u*cfg.n{return;}let i=f/2u;let a=f%2u;let q=point(i)+axis(a);
    var out:Transport;
    if !valid(q)||internal_face(f){transport[f]=out;/*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){scratch[flux_index(f,l)]=vec4<f32>(0.);}return;}
    let j=at(q);
    let amount=gas_carrier(f);
    let origin=select(1u,0u,amount>=0.);let donor=select(j,i,origin==0u);
    // Slot l holds identity l + 1 on both sides.
    /*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){scratch[flux_index(f,l)]=vec4<f32>(0.,f32(l+1u),0.,f32(l+1u));}
    if cells[donor].volume.gas>0.{
        var high:array<f32,GAS_SLOTS>;var low:array<f32,GAS_SLOTS>;var total_high=0.;var total_low=0.;
        /*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){
            let entry=geometry[layer_index(donor,l)];var lo=vec2<f32>(-.5);var hi=vec2<f32>(.5);
            var carried=amount;
            if entry.x>0.{
                let own=identity_velocity(point(donor),entry.y)[a]*cfg.dt/(carry_divisions()*cfg.h);
                carried=select(clamp(own,amount,0.),clamp(own,0.,amount),amount>=0.);
            }
            if carried>=0.{lo[a]=.5-carried;}else{hi[a]=-.5-carried;}
            high[l]=carried*parallel_slab(donor,l,lo,hi);low[l]=carried*entry.x;
            total_high+=high[l];total_low+=low[l];
        }
        // Together the sweeps fill no more than the width the gas sweeps.
        var blend=1.;
        if abs(total_high)>abs(amount)&&abs(total_high)>abs(total_low){
            blend=clamp((abs(amount)-abs(total_low))/(abs(total_high)-abs(total_low)),0.,1.);
        }
        // Gas that is held back has no candidate, and the water takes its
        // share of the carrier within its budget, like any other increase:
        // under a tooth (teeth_shadow), from a bubble smaller than a cell
        // into pores, into a column's plug (plug_wall), into a small
        // bubble's channel from outside it or of another identity, and into
        // a packing drained to its pore water.
        let receiver=select(i,j,origin==0u);
        let closed=teeth_shadow(point(donor),point(receiver))||(pores_only(cells[receiver])&&lone_blob_cell(point(donor)))||plug_wall(donor,receiver)||(border_cell(receiver)&&!border_cell(donor));
        let drained=close_drained(cells[receiver]);
        var gas=0.;/*loop: slower written out*/for(var l=0u;l<GAS_SLOTS;l++){
            let value=select(mix(low[l],high[l],blend),0.,closed||drained||spacer_channel_closed(l,receiver));gas+=value;put_source_gas_flux(f,l,origin,value);
        }out.volume.w=gas;
    }
    transport[f]=out;
}
// Trace gas identities (below TRACE_VOLUME of a cell, shared.wgsl) would
// send every cell through the gas geometry, the passages and the capillary
// paths: after the carry they return their volume to the liquid.
@compute @workgroup_size(128) fn clear_trace_gas(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    var kept=0.;var cleared=false;var held=0u;
    for(var l=0u;l<GAS_SLOTS;l++){
        let amount=geometry[layer_index(i,l)].x;
        if amount>0.&&amount<TRACE_VOLUME{geometry[layer_index(i,l)].x=0.;cleared=true;}
        else if amount>0.{kept+=amount;held|=1u<<l;}
    }
    // The slots that hold gas now, for the relabelling (label_cell_word).
    let before=label_cell_word(i);
    label_cell_keep(i,before,(before&~(CELL_SLOTS<<CELL_HELD))|(held<<CELL_HELD));
    let c=cells[i];let gas=c.volume.gas;
    if !cleared&&!(kept==0.&&gas>0.&&gas<TRACE_VOLUME){return;}
    // The identities make up the bulk gas; keep that relation exactly.
    var scale=0.;if gas>0.{scale=kept/gas;}
    cells[i].volume.gas=kept;cells[i].volume.water=c.volume.water+(gas-kept);
    cells[i].momentum[3]=scale*c.momentum[3];
}
