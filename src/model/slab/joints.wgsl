// The passages through the raft: the films where bubbles touch, the packed
// column sliding through a slot between bubbles and its plug, small bubbles
// and the channels beside them. gap_marks and column_cells record them in
// the Slab stage, after body_faces, for the update's mechanics (plug_wall,
// raft_film) and transport (joint_mouth, the gas closures).

// The jamming width in coarse-grain diameters (0.61 mm): sand arches across
// a narrower opening and clogs it.
const JAMMING_WIDTH:f32=3.3;
// A cell where two bubbles of the raft meet: between bubbles, no column
// cell, its wet part narrower than the jamming width. The bubbles touch
// there across a soap film whose liquid is the Plateau borders at its ends,
// a contact below the grid: no grain crosses the cell's top or bottom face,
// and the water does, as through a sieve.
fn raft_film(i:u32)->bool{
    let c=cells[i];let g=grains(c);
    if (max(c.volume.water,0.)+g.x+g.y)*cfg.h>=JAMMING_WIDTH*GRAIN_DIAMETER.y{return false;}
    return !column_cell(i)&&between_bubbles(point(i));
}
// Whether the cell at `p` lies in a slot between bubbles: along its row,
// within 5 mm (the raft's reach) on each side, a cell holding gas or the
// tank's wall.
fn between_bubbles(p:vec2<i32>)->bool{
    let reach=i32(ceil(.005/cfg.h));
    for(var side=0;side<2;side++){
        let step=select(-1,1,side==1);var closed=false;
        for(var k=1;k<=reach;k++){
            let q=p+vec2<i32>(step*k,0);
            if !valid(q){closed=true;break;}
            if cells[at(q)].volume.gas>.05{closed=true;break;}
        }
        if !closed{return false;}
    }
    return true;
}
// A cell of the packed column in a slot between bubbles (COLUMN_ROW): below
// the slab's body, the slot holds the sand the mouth above sends it, as
// wide as its grains make it, with the bubbles' liquid borders beside it.
// The bubbles cannot enter it (its pores' entry pressure is kPa against
// tens of Pa of pressing), and its sand goes down the chain (joint_mouth).
fn column_cell(i:u32)->bool{return scratch[COLUMN_ROW*cfg.n+i].x>0.;}
// A column cell whose sand holds its slot at the chain's packing is a plug:
// sand that arches, jammed in its slot, which the transport keeps to the
// chain's faces. The mouth feeding a slot opens at the jamming width, so a
// fed slot soon holds a plug. A column cell with less sand is a wet gap
// holding a suspension: its grains move with the water and the bubbles
// press the gap shut as it drains.
fn column_plug(i:u32)->bool{
    let chain=scratch[COLUMN_ROW*cfg.n+i];if chain.x<=0.||chain.z<=0.{return false;}
    return plug_holds_slot(i,chain.z);
}
// Sand that holds a slot open against the bubbles: caged grains
// (FREEZING_PACKING of the wet part or more) that span the jamming width
// when packed to `packing`. A looser suspension has no skeleton to push
// back and the bubbles close on it, whatever its amount.
fn plug_holds_slot(i:u32,packing:f32)->bool{
    let c=cells[i];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    return packing>0.&&solid>=FREEZING_PACKING*wet&&solid/packing*cfg.h>=JAMMING_WIDTH*GRAIN_DIAMETER.y;
}
// The width (cells) of a column plug's sand at the chain's packing; 0 for
// any other cell.
fn plug_width(i:u32)->f32{
    if !column_plug(i){return 0.;}
    let g=grains(cells[i]);return (g.x+g.y)/scratch[COLUMN_ROW*cfg.n+i].z;
}
// Whether the face between cells i and j lies between a column's plug and a
// bubble, a cell holding a grain layer of gas or more (the raft's bubble
// cells can be column cells themselves). The pores' entry pressure (kPa
// against tens of Pa of pressing) closes it both ways: the bubble neither
// pushes into the plug nor draws its pore water out, so a fed slot keeps
// the plug's width, whether the plug slides or the plates hold it. The
// flow solve and the mechanics close the slot's sides to the mixture, the
// transport any such face to gas alone: sealed along the chain too, a plug
// whose exit passes sand ratchets the pressure of the body above it.
fn plug_wall(i:u32,j:u32)->bool{
    let bubble_i=cells[i].volume.gas*cfg.h>=GRAIN_LAYER;
    let bubble_j=cells[j].volume.gas*cfg.h>=GRAIN_LAYER;
    return (column_plug(i)&&bubble_j)||(column_plug(j)&&bubble_i);
}
// A plug the plates hold still is a jammed tooth: it does not slide and its
// exit passes nothing. A sliding plug settles through its own counterflow,
// its grains carried by the rising water and not by each other.
fn static_plug(i:u32)->bool{return column_plug(i)&&held_still(i);}
// Whether the wet cell i between bubbles continues a chain of packing
// `packing`: beside a small bubble, or as a plug. Once a dense column
// stands in a slot the bubbles press it from both sides alike and cannot
// close it. Water alone between two bubbles, or between a bubble and the
// wall, is the wedge over their film, whatever its width on the grid: a
// dead end the sand fills to the contact packing and no further, since a
// grain cannot cross a soap film (pi sigma d, 2e-5 N, against the 8e-7 N
// the slab's base pressure of 23 Pa puts on one grain) and the raft's
// buoyancy head (33-37 Pa) exceeds that pressure. A slot that water alone
// keeps open is a small bubble's Plateau border reaching through the raft.
fn chain_slot(i:u32,packing:f32)->bool{
    if spacer_beside(i){return true;}
    return plug_holds_slot(i,packing);
}
// Whether a small bubble of the raft holds the slot of cell i open. A
// bubble pressed under the slab flattens into a puddle two capillary
// lengths thick once it is wider than that; a smaller one, of an area
// below pi l_c^2 (11 mm2), stays round, and its Laplace pressure (sigma/r,
// 28 Pa at 2.5 mm against the large puddles' 4-7 Pa) keeps the large
// bubbles from squeezing it out. Wedged between them it holds the junction
// open, and the water beside it is the Plateau border between three
// bodies, of radius sigma/(rho g 2 l_c) = l_c/2 under the raft's buoyancy
// head: a channel from the slab's underside to the open water below.
fn spacer_beside(i:u32)->bool{return spacer_slot(i)>=0;}
// The slot (identity - 1) of the small bubble whose channel cell i is, or
// -1. The channel is water and sand between the small bubble and its
// neighbour, so the cell holds at least the jamming width of them: a cell
// the gas fills to within less is a film over which the bubbles touch, and
// no grain passes it.
fn spacer_slot(i:u32)->i32{
    let c=cells[i];let g=grains(c);
    if (max(c.volume.water,0.)+g.x+g.y)*cfg.h<JAMMING_WIDTH*GRAIN_DIAMETER.y{return -1;}
    let own=gap_spacer(i);if own>=0{return own;}
    // The small bubble holds the junction open through the raft's height,
    // wherever in the gap it floats: the open passage between the large
    // bubbles above it, up to the slab, is its channel too.
    let p=point(i);let reach=i32(ceil(.005/cfg.h));
    for(var k=1;k<=reach;k++){
        let q=p-k*axis(1u);if !valid(q){return -1;}let j=at(q);
        if !gap_between(j)||!gap_open(j){return -1;}
        let below=gap_spacer(j);if below>=0{return below;}
    }
    return -1;
}
// The slot of the small bubble beside cell i in its row, or -1: a bubble a
// row neighbour belongs to and the cell itself does not, by the areas of
// the last relabelling (label_area), the larger of two. A bubble holds a
// junction open only if it is wider than the junction's Plateau border, a
// round diameter of l_c: a smaller one floats in the border's water and
// the large bubbles close round it. Small bubbles are the round diameters
// from l_c to 2 l_c, areas from pi l_c^2/4 (2.8 mm2) to pi l_c^2.
fn spacer_in_row(i:u32)->i32{
    let p=point(i);let spacer=3.14159265*CAPILLARY_LENGTH*CAPILLARY_LENGTH/(cfg.h*cfg.h);
    let sphere=.25*spacer;
    var best=-1;var largest=0.;
    for(var side=-1;side<=1;side+=2){
        let q=p+side*axis(0u);if !valid(q){continue;}
        for(var l=0u;l<GAS_SLOTS;l++){
            if label_area(i,l)>0.{continue;}
            let area=label_area(at(q),l);
            if area>=sphere&&area<spacer&&area>largest{best=i32(l);largest=area;}
        }
    }
    return best;
}
// A cell of a small bubble's channel (COLUMN_ROW.y > 0), whether or not a
// chain runs through it: the Plateau border beside the small bubble, whose
// curvature (radius l_c/2) a large bubble's surface cannot follow at its
// lower pressure. The transport gives no gas a candidate into it from
// outside; the gas the grid smears into it is the large bubbles' rounded
// edges, which move with their own bubbles. The flow solve leaves it open:
// closed there too, the room above the mouth is sealed and ratchets.
fn border_cell(i:u32)->bool{return scratch[COLUMN_ROW*cfg.n+i].y>0.;}
// Whether gas of slot l is kept out of channel cell i: the channel is
// another small bubble's.
fn spacer_channel_closed(l:u32,i:u32)->bool{
    let row=scratch[COLUMN_ROW*cfg.n+i];
    return row.y>0.&&f32(l+1u)!=row.w;
}
// The width (cells) of the Plateau border beside a small bubble: l_c/2.
fn spacer_channel()->f32{return .5*CAPILLARY_LENGTH/cfg.h;}
// The width (cells) of the slab's underside that breaches into the channel
// cell i beside the small bubble of slot l: the channel's own border and
// half the bubble's round diameter. The small bubble bobs in the gap below
// the slab, so the sand over it breaches too and flows round it into the
// channels on its two sides, half to each.
fn spacer_mouth(i:u32,l:u32)->f32{
    let p=point(i);var area=0.;
    for(var side=-1;side<=1;side+=2){
        let q=p+side*axis(0u);if !valid(q){continue;}
        area=max(area,label_area(at(q),l));
    }
    return spacer_channel()+sqrt(area/3.14159265);
}
// Per cell, written by gap_marks before the column scan: x = 1 where the
// cell lies between bubbles, and for such a cell y = 1 where its slot
// passes the jamming width (open_passage) and z = 1 + the slot of the small
// bubble beside it in its row (0 without one). The scan walks a grid column
// on one thread; the marks keep the searches along the rows out of it.
@group(0) @binding(15) var<storage,read_write> gaps:array<vec4<f32>>;
@compute @workgroup_size(128) fn gap_marks(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=point(i);var mark=vec4<f32>(0.);
    if between_bubbles(p){
        mark.x=1.;mark.z=f32(spacer_in_row(i)+1);
        if open_passage(p){mark.y=1.;}
    }
    gaps[i]=mark;
}
fn gap_between(i:u32)->bool{return gaps[i].x>0.;}
fn gap_open(i:u32)->bool{return gaps[i].y>0.;}
fn gap_spacer(i:u32)->i32{return i32(gaps[i].z)-1;}
// Per grid column, after body_faces: scratch[COLUMN_ROW n + i] =
// (1, w, the body's packing at the chain's top, s) for a column cell,
// (0, w, 0, s) for a cell of a small bubble's border that no chain runs
// through, and 0 for any other cell. s is the small bubble's identity and
// w the width (cells) of the underside that breaches into its channel
// (spacer_mouth), both 0 without a small bubble: the transport cannot read
// the relabelling itself.
@compute @workgroup_size(128) fn column_cells(@builtin(global_invocation_id) id:vec3<u32>){
    let col=id.x;if col>=cfg.nx{return;}
    for(var r=0;r<i32(cfg.ny);r++){scratch[COLUMN_ROW*cfg.n+at(vec2<i32>(i32(col),r))]=vec4<f32>(0.);}
    // The slab's sand below the ceiling (and below a crater's water).
    var y=i32(cfg.ny)-1;
    while y>=i32(cfg.ny)/2&&!teeth_sandy(cells[at(vec2<i32>(i32(col),y))]){y-=1;}
    if y<i32(cfg.ny)/2{return;}
    // Down through the slab: a body cell (packed, or a body's last layer)
    // sets the chain's packing; the slab's loose cells (a crater's floor, a
    // column loosened by the plug's deficits) are passed, however many;
    // below a body cell or a column cell, a wet cell between bubbles that a
    // plug or a small bubble holds open (chain_slot) is a column cell. Open
    // water, a bubble, a film's wedge or deposit ends the scan.
    var packing=0.;var chain=false;var border=vec2<f32>(0.);
    while y>=0{
        let i=at(vec2<i32>(i32(col),y));let c=cells[i];
        if plug_body(i){
            let g=grains(c);let solid=g.x+g.y;border=vec2<f32>(0.);
            packing=min(solid/max(dense_volume(i),1e-6),CLOSE_PACKING);chain=true;y-=1;continue;
        }
        if plug_deposit(c){break;}
        let g=grains(c);let wet=max(c.volume.water,0.)+g.x+g.y;
        // Below a small bubble's own rows its channel goes on to the open
        // water under the raft, the large bubbles' rounded edges staying
        // apart, through cells the grains can pass (gap_open); over a film
        // or a foam, narrower than the jamming width, the chain ends.
        if border.y>0.&&gap_between(i){
            if !gap_open(i){break;}
            scratch[COLUMN_ROW*cfg.n+i]=vec4<f32>(1.,border.x,packing,border.y);y-=1;continue;
        }
        if chain&&wet>0.&&gap_between(i)&&chain_slot(i,packing){
            let slot=spacer_slot(i);
            var mouth=0.;if slot>=0{mouth=spacer_mouth(i,u32(slot));border=vec2<f32>(mouth,f32(slot+1));}
            scratch[COLUMN_ROW*cfg.n+i]=vec4<f32>(1.,mouth,packing,f32(slot+1));y-=1;continue;
        }
        if teeth_sandy(c){chain=false;y-=1;continue;}
        break;
    }
    // A small bubble's border is the three bubbles' capillary geometry, not
    // the sand's: it is recorded, and closed to the large bubbles' gas,
    // whether or not the chain reaches it this update (the chain breaks
    // whenever the mouth's body cell drains below the body's packing).
    var mark=vec2<f32>(0.);
    for(var r=i32(cfg.ny)-1;r>=0;r--){
        let i=at(vec2<i32>(i32(col),r));let entry=scratch[COLUMN_ROW*cfg.n+i];
        if entry.x>0.{mark=select(vec2<f32>(0.),entry.yw,entry.y>0.);continue;}
        if !gap_between(i)||plug_deposit(cells[i])||!gap_open(i){mark=vec2<f32>(0.);continue;}
        let slot=spacer_slot(i);
        if slot>=0{mark=vec2<f32>(spacer_mouth(i,u32(slot)),f32(slot+1));}
        if mark.y>0.{scratch[COLUMN_ROW*cfg.n+i]=vec4<f32>(0.,mark.x,0.,mark.y);}
    }
}
// The span (cells) of the slot the cell at `p` lies in: its own wet part
// plus the larger wet part of its two row neighbours, counted where the
// neighbour lies between bubbles too and the face between them is open to
// water (a wet aperture of a grain layer or more, as the mechanics recorded
// it): a slot wider than the jamming width but narrower than two cells can
// straddle a cell boundary. A face the gas closes separates two films,
// whose water is not summed, or a row of bubbles' films would pass sand.
fn slot_span(p:vec2<i32>)->f32{
    let i=at(p);let c=cells[i];let g=grains(c);let own=max(c.volume.water,0.)+g.x+g.y;
    var beside=0.;
    for(var step=-1;step<=1;step+=2){
        let q=p+vec2<i32>(step,0);if !valid(q){continue;}
        let f=select(2u*at(q),2u*i,step>0);
        if faces[f].coarse_momentum.z*cfg.h<GRAIN_LAYER{continue;}
        if !between_bubbles(q){continue;}
        let cq=cells[at(q)];let gq=grains(cq);beside=max(beside,max(cq.volume.water,0.)+gq.x+gq.y);
    }
    return own+beside;
}
// A passage the grains can take through the raft's row: water and sand at
// least the jamming width across, in the cell or straddling it and an open
// neighbour (slot_span). A narrower cell is the bubbles' film or foam.
fn open_passage(p:vec2<i32>)->bool{return slot_span(p)*cfg.h>=JAMMING_WIDTH*GRAIN_DIAMETER.y;}
// The grains (cells per substep, downward) that cross the horizontal face
// `f` of a joint's chain, or -1 where `f` is no such face. The sand crosses
// the chain's faces at this allowance and at no other rate, nothing where
// the chain is closed: left to the transport's own flux, a plug whose exit
// is blocked keeps taking the body's mixture while its water leaves. The
// throat, the cell under the face, takes no more than brings it to
// LOOSE_PACKING: released sand is a suspension that must fall away, and a
// throat that cannot drain it is a clogged orifice.
fn joint_mouth(f:u32,interval:f32)->f32{
    if f%2u!=1u{return -1.;}
    let i=f/2u;let q=point(i)+axis(1u);if !valid(q){return -1.;}
    let j=at(q);let chain=scratch[COLUMN_ROW*cfg.n+j];let column=chain.x>0.;
    if !column&&!plug_body(j){return -1.;}
    let throat=cells[i];let gt=grains(throat);let st=gt.x+gt.y;let wt=max(throat.volume.water,0.)+st;
    if plug_deposit(throat)||wt<=0.{return -1.;}
    // A body's mouth opens on a column cell; a column's exit on the next
    // column cell or on the open water below the raft. Any other face is
    // the transport's.
    if !column&&!column_cell(i){return -1.;}
    if st>=LOOSE_PACKING*wt{return 0.;}
    // A slot a small bubble holds is its Plateau border, on either side of
    // the face: the capillary balance of the three bodies sets its width
    // below the grid, however the gas planes cut the face.
    let border=scratch[COLUMN_ROW*cfg.n+i].y;
    var held=0.;if border>0.||chain.y>0.{held=spacer_channel();}
    // A slot a plug holds is as wide as the plug: its sand at the chain's
    // packing arches over no narrower orifice than its own.
    if held<=0.{held=max(plug_width(i),select(0.,plug_width(j),column));}
    var aperture=faces[f].coarse_momentum.z;
    if held>0.{aperture=held;}else{
        // Otherwise the throat is water or a suspension, not a film beside
        // a bubble: its slot spans the jamming width or more.
        if slot_span(point(i))*cfg.h<JAMMING_WIDTH*GRAIN_DIAMETER.y{return 0.;}
        // The face is then the orifice, and one narrower than the jamming
        // width arches. Its aperture is the mechanics' record on the face:
        // the transport has since moved the gas away from the contact fits
        // (CONTACT_ROW) that free_wet_aperture reads.
        if aperture*cfg.h<JAMMING_WIDTH*GRAIN_DIAMETER.y{return 0.;}
    }
    var allowed=0.;
    let below=select(0.,st/wt,wt>0.);
    if column{
        // A column's sand passes at the batch flux of its own wet packing
        // and no more than the cell holds: dilute or dense it passes little,
        // and a fed column settles where its flux matches its feed. A plug
        // stays dense, for diluted the bubbles would close its slot, and
        // passes through no wider a face than its cell's wet part. A slot a
        // small bubble holds open (chain.y) is free to dilate: it passes the
        // Godunov flux between the packings above and below through its
        // Plateau border, however much of the cell the grid gives the gas.
        let c=cells[j];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
        let phi=min(solid/max(wet,1e-6),CLOSE_PACKING);
        var flux=batch_flux_at(g,white_powder(c),phi);
        if chain.y>0.{flux=godunov_flux(g,white_powder(c),phi,below);}
        let width=select(min(wet,aperture),aperture,chain.y>0.);
        allowed=min(flux*interval*width,solid);
        // A jammed tooth does not slide.
        if chain.y<=0.&&static_plug(j){allowed=0.;}
    }else{
        // The mouth is the body's face looking down into the slot. It
        // breaches as a roof at the body's packing (a last layer's is its
        // body's), whether or not the plates hold the body: a sandglass
        // under water runs through any orifice wider than a few grains.
        // The slot's counterflow caps what it releases. Into a small
        // bubble's channel the underside breaches over the width COLUMN_ROW
        // records, and the channel carries the Godunov flux to the throat.
        let body=cells[j];let g=grains(body);let solid=g.x+g.y;
        let phi=min(solid/max(dense_volume(j),1e-6),CLOSE_PACKING);
        let powder=white_powder(body);
        let rate=breaching_rate(g,powder,phi)*inclination_factor(-1.,grain_friction(g));
        var supply=rate*phi*aperture;var carry=batch_flux_rate(g,powder)*aperture;
        if border>0.{supply=rate*phi*border;carry=godunov_flux(g,powder,phi,below)*aperture;}
        allowed=min(supply,carry)*interval;
    }
    let room=max(LOOSE_PACKING*wt-st,0.);
    return min(allowed,room);
}
