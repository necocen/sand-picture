// Slip velocity of isolated sub-gap spherical bubbles (smaller than the plate
// gap), read by momentum and face-velocity helpers. Macroscopic gas transport
// is owned by the Lagrangian foam network (`foam.wgsl`).

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
// hindered as grains are by the voidage to the power HINDRANCE.
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
