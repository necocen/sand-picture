// The white sand's powder (the patent's 200-mesh silica, a flour of about
// 20 um, material.rs): bound to the white's grit until the sand that holds
// it is torn apart (free_powder, before the carry, stage TransportPrepare),
// then lagging the white in a suspension (powder_drift, in the carry).

// Settled and consolidated, fine silica grains attract each other far more
// strongly than they weigh (van der Waals at contact, a Bond number of
// some hundreds for 20 um): a packing's powder leaves it bound to its grit,
// in the aggregates the grains come away in, and moves, settles and packs
// with it (white_speed). Sand in contact (the share in contact, the
// material table's blend from the freezing packing to the loose one)
// through which the water runs faster than its white grains settle is not
// settling grain by grain: the water's drag on a grain exceeds the grain's
// weight, tears the mass and its aggregates apart and frees the powder of
// the share in contact. The weight is the whole gravity's on the grain
// (with the hands' shake): while the picture is tipped over, the part
// across the gap presses the grains on the lower plate, and when it lies
// flat they weigh as much as when it stands. (Measured by the gravity along
// the glass alone, the speed falls to nothing at the flat moment and any
// motion frees the powder of all sand in contact.) The water's speed
// through the sand is the white's slip against the cell's water, so sand
// that falls with its own water keeps its powder. The free share only
// grows, until the powder reaches the deposit at its own speed and is bound
// again (arrival.wgsl).
@compute @workgroup_size(128) fn free_powder(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    let c=cells[i];let g=grains(c);let water=max(c.volume.water,0.);let wet=water+g.x+g.y;
    if g.x<=0.||water<=0.||c.powder.free>=c.powder.mobile{return;}
    let slip=c.momentum[1]/g.x-c.momentum[0]/water;
    let settling=TERMINAL_SPEED.x*length(vec3<f32>(cfg.lateral,cfg.gravity,plate_gravity()))/STANDARD_GRAVITY;
    if length(slip)<=settling{return;}
    let contact=smoothstep(FREEZING_PACKING,LOOSE_PACKING,(g.x+g.y)/wet);
    cells[i].powder.free=max(c.powder.free,contact*c.powder.mobile);
}
// Whether the white grains of cell i hold their free powder among them:
// grains that touch, caged (the freezing packing) or packed in a part of
// the cell, as a body's last layer (body_faces) or a layer one grain thick
// on the deposit (the cell's own or the one below). The cell's mean packing
// does not show a layer packed on one side of it.
fn powder_held(i:u32)->bool{
    let c=cells[i];let wet=free_wet(c);let g=grains(c);let solid=g.x+g.y;
    if wet<=0.||solid>=FREEZING_PACKING*wet||scratch[BODY_ROW*cfg.n+i].y>=0.{return true;}
    if solid<=0.||solid<dot(g,GRAIN_DIAMETER)/(solid*cfg.h)*CRITICAL_PACKING{return false;}
    let below=point(i)-axis(1u);
    return c.bed.z>0.||(valid(below)&&cells[at(below)].bed.z>0.);
}
// The shares of a transport group's white that are free powder and powder.
fn group_powder(i:u32)->PowderShare{
    let g=group_payload(i);if g.volume.y<=0.{return PowderShare(0.,0.);}
    let total=clamp(g.powder.x/g.volume.y,0.,1.);
    return PowderShare(clamp(g.powder.y/g.volume.y,0.,total),total);
}
// The free powder's flux across face f from cell i to cell j (cells per
// subdivision) beyond its share of the white flux `white`, given the face's
// free wet current `current`. Every grain slips through the liquid under
// the same body force per mass against a drag that goes as its size
// squared, so the free powder's slip against the water is the white's (the
// face's relative velocities) times POWDER_SPEED over white_speed: in a
// suspension it lags the white's grit by the difference of their settling
// speeds, and where the white rests or moves with the water it stays with
// it. It crosses from its own upwind side, and grains that hold it on
// either side (powder_held) keep it.
fn powder_drift(f:u32,i:u32,j:u32,interval:f32,current:f32,white:f32)->f32{
    if powder_held(i)||powder_held(j){return 0.;}
    let face=faces[f];
    let slip=face.velocity.x+(POWDER_SPEED/white_speed(face_powder(cells[i],cells[j])))*(face.velocity.y-face.velocity.x);
    let carrier=current+interval*face.coarse_momentum.z*slip;
    let donor=select(j,i,carrier>=0.);
    let crossing=carrier*group_powder(donor).free*wet_concentration(point(donor),f%2u,carrier).x;
    let powder=sign(crossing)*min(abs(crossing),cells[donor].powder.free);
    return powder-white*group_powder(select(j,i,white>=0.)).free;
}
