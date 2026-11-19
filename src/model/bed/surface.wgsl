// The deposit's surface flow. Between two neighbouring columns the top of
// the higher bed slides down as one circulation mode: a packet of packed
// sand with its pore water changes places with the same volume of the
// suspension beside it. Once an update, after the capture: the even column
// pairs, then the odd ones, each pass writing disjoint pairs.

// The bed's flux along the slope (m2/s, toward +x) and the yield depth (m).
struct SurfaceRate {domain_flux:f32,depth:f32}
// The rate of a pair's surface flow. `slope` is the surface's rise per cell
// toward +x, `height` the donor's bed (m), `load` the support its top needs
// per solid volume (required_support), `impact` the reaction that stopped
// the captured sand (Pa), `previous` the pair's last depth and flux. The
// powders are the white's shares in the packed layer and in the suspension.
fn surface_rate(slope:f32,height:f32,packing:f32,share:vec2<f32>,bed_powder:PowderShare,ambient_powder:PowderShare,load:vec2<f32>,impact:vec2<f32>,ambient:vec3<f32>,previous:vec4<f32>,mu:f32)->SurfaceRate{
    var result:SurfaceRate;if packing<=0.||packing>=1.||height<=0.{return result;}
    let t=normalize(vec2<f32>(1.,slope));let n=vec2<f32>(-t.y,t.x);
    let normal=dot(load,n);if normal<=0.{return result;}
    let tangential=-dot(load,t);let pressure0=max(0.,dot(impact,n));let traction0=-dot(impact,t);
    let direction=select(-1.,1.,tangential+traction0/max(packing*height,1e-30)>=0.);
    let drive=direction*tangential;let boundary=direction*traction0;
    // The layer yields down to the depth d where the drive on it no longer
    // beats the friction of its grains and of the two plates under the
    // pressure P(z) = P0 + phi N z (the larger root below), at most the
    // bed's thickness. Under submerged gravity alone d is
    // b (tan(theta) - mu)/mu_w (Jop et al. 2005).
    let qa=KINETIC_FRICTION*packing*normal/PLATE_GAP;
    let qb=packing*(mu*normal-drive)+2.*KINETIC_FRICTION*pressure0/PLATE_GAP;
    let qc=mu*pressure0-boundary;
    let discriminant=qb*qb-4.*qa*qc;if discriminant<=0.{return result;}
    let depth=min(height*n.y,max(0.,(-qb+sqrt(discriminant))/(2.*qa)));
    if depth<=0.{return result;}
    // The packet's volume returns as suspension over the bed, not through
    // its pores: two lanes of the depth d with no net volume flow, the bed's
    // u/v = 1 - z/d and the exterior's 1 - 7 z/d + 6 (z/d)^2, continuous at
    // the surface and zero at d. Their squares integrate to d/3 and 8 d/15
    // and the exterior's squared shear to 13/d: the factors 1.6 and 39/d^2.
    let packed=vec3<f32>(1.-packing,packing*share);
    let resistance=common_resistance(packed,bed_powder,vec2<f32>(0.),1.)+1.6*common_resistance(ambient,ambient_powder,vec2<f32>(0.),1.)
        +39.*WATER_VISCOSITY/(depth*depth);
    let mass=dot(packed,RHO)+1.6*dot(ambient,RHO);
    result.depth=depth;
    // The surface speed the pair's last flux left: depth that yields anew
    // starts at rest, and the history is not advected.
    var old_speed=0.;
    if previous.x>0.{
        old_speed=max(0.,direction*2.*previous.y/previous.x)
            *min(1.,previous.x/depth);
    }
    let inertia=mass/cfg.dt;
    // Backward Euler for the surface speed v by the virtual work of the
    // bed's profile: the drive phi Ft d/2 + T0 and the inertia against the
    // plates' friction mu_w (P0 d + phi N d^2/3)/b, the grains' shear (the
    // contact law's at P(z) and the rate v/d, the mean of four depths) and
    // the drag (R + I) d v/3. Below the static shear nothing moves; else v
    // is bisected. The bed's lane carries v d/2.
    let forcing=packing*drive*depth*.5+boundary+inertia*depth*old_speed/3.;
    let wall=KINETIC_FRICTION*(pressure0*depth+packing*normal*depth*depth/3.)/PLATE_GAP;
    let static_shear=mu*(pressure0+packing*normal*depth*.5);
    let available=forcing-wall-static_shear;if available<=0.{return result;}
    var lo=0.;var hi=available/((resistance+inertia)*depth/3.);
    for(var iteration=0u;iteration<16u;iteration++){
        let speed=.5*(lo+hi);let viscous=WATER_VISCOSITY*speed/depth;var shear=0.;
        for(var q=0u;q<4u;q++){
            let z=depth*(f32(q)+.5)/4.;let pressure=pressure0+packing*normal*z;
            let rate=.38*viscous/(.005*pressure+viscous);
            shear+=.25*(pressure*(mu+rate)+2.5*packing*sqrt(pressure*viscous));
        }
        let opposed=wall+shear+(resistance+inertia)*depth*speed/3.;
        if opposed>forcing{hi=speed;}else{lo=speed;}
    }
    result.domain_flux=direction*depth*.5*(lo+hi)*.5;
    return result;
}

// How far the gas leaves face f open to a grain of `diameter` (0 to 1). Gas
// too little to make a layer one grain thick across the face's two cells
// does not stop a grain (as grain_crosses): the face is open.
fn surface_face_passage(f:u32,diameter:f32)->f32{
    let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q){return 0.;}
    if (cells[i].volume.gas+cells[at(q)].volume.gas)*cfg.h<diameter{return 1.;}
    return curved_passage(f,diameter/(2.*cfg.h)).total;
}
// A packet's way: `source` the row of xa's bed it leaves, `gate` the row
// where it passes from xa to xb, `row` the row of xb where it comes to rest,
// `aperture` how far the gas leaves the way open to the grain (0 to 1).
struct SurfacePath {row:u32,gate:u32,source:u32,aperture:f32}
// Sand in contact in the free part of a cell over a column's bed: a body
// that something else holds there (the slab on its raft, a lump on a
// bubble). A packet does not enter it or fall through it.
fn held_body(x:u32,row:u32)->bool{
    let c=cells[row*cfg.nx+x];let g=grains(c);let free=free_wet(c);
    return free>0.&&g.x+g.y>=LOOSE_PACKING*free;
}
// The way of a packet from the top of column xa's bed, row ya, into column
// xb, whose bed is open at row `open`. It crosses at the row of xa's top
// and falls down xb to the first cell whose floor a grain cannot cross: the
// bed's open cell, the top of a bubble lying on the bed or the floor, or
// the top of a held body. On a bubble it lies as loose sand (bed above a
// column's top is released) until the bubble carries a packing and the
// column goes on over it (pocket_cover); sent past the bubble to the open
// cell, it could not move at all. Driven uphill by a stream's traction
// (`open` above ya), it climbs xa's column to that row and crosses there.
// A top row covered on xb's side leaves by covered_path.
fn surface_path(xa:u32,ya:u32,xb:u32,open:u32,diameter:f32)->SurfacePath{
    var path:SurfacePath;path.row=max(ya,open);path.source=ya;
    path.aperture=surface_face_passage(2u*(path.row*cfg.nx+min(xa,xb)),diameter);
    for(var row=ya;row<open;row++){
        path.aperture=min(path.aperture,surface_face_passage(2u*(row*cfg.nx+xa)+1u,diameter));
    }
    path.gate=path.row;
    if path.row>open&&(path.aperture<=0.||held_body(xb,path.row)){return covered_path(xa,ya,xb,open,diameter);}
    while path.row>open{
        let below=surface_face_passage(2u*((path.row-1u)*cfg.nx+xb)+1u,diameter);
        if below<=0.||(path.row-1u>open&&held_body(xb,path.row-1u)){break;}
        path.aperture=min(path.aperture,below);path.row--;
    }
    return path;
}
// The way out of a face of xa's bed whose top row is covered on xb's side
// by a bubble lying against it or by a held body over xb's bed (the slab's
// end on its raft). Under the cover the face stands over xb's open water
// and its grains leave as under a gate: across at the cover's lip, the
// highest row of xa's bed that open water reaches from xb's open cell
// (`gate`), and down to that cell. The bed above the lip comes down behind
// them, so the packet is taken from the top of the bed standing on that row
// without a break (`source`): xa's top or, under a pocket in xa's column,
// the bed's surface under it, for sand over a bubble does not pass it.
fn covered_path(xa:u32,ya:u32,xb:u32,open:u32,diameter:f32)->SurfacePath{
    var path:SurfacePath;path.row=open;path.gate=open;path.source=ya;
    var way=1.;
    for(var row=open;row<=ya;row++){
        if row>open{
            way=min(way,surface_face_passage(2u*((row-1u)*cfg.nx+xb)+1u,diameter));
            if way<=0.||held_body(xb,row){break;}
        }
        if cells[row*cfg.nx+xa].bed.z<=0.{continue;}
        let across=min(way,surface_face_passage(2u*(row*cfg.nx+min(xa,xb)),diameter));
        if across>0.{path.gate=row;path.aperture=across;}
    }
    var source=path.gate;
    while source<ya{
        let above=cells[(source+1u)*cfg.nx+xa];
        if above.bed.z<=0.||pocket_row(above){break;}
        source++;
    }
    path.source=source;
    return path;
}
// Moves up to `requested` grains of each species (fine, coarse; cell
// volumes) from xa's bed onto xb's along surface_path and returns the
// grains moved. The packet keeps the donor's packing and pore water; the
// same volume of xb's suspension comes back with its momentum, its grains'
// stress and its powder.
fn transfer_surface(xa:u32,xb:u32,requested:vec2<f32>)->vec2<f32>{
    var remaining=requested;var moved=vec2<f32>(0.);
    for(var part=0u;part<cfg.ny;part++){
        if remaining.x+remaining.y<=0.{break;}
        let top=bed_top(xa);
        let path=surface_path(xa,top,xb,bed_open(xb),select(GRAIN_DIAMETER.x,GRAIN_DIAMETER.y,cells[top*cfg.nx+xa].bed.y>0.));
        let ya=path.source;let ia=ya*cfg.nx+xa;let a=cells[ia];let domain=a.bed.z;
        let yb=path.row;let ib=yb*cfg.nx+xb;
        let b=cells[ib];let free=free_wet(b);
        if domain<=0.||free<=0.||path.aperture<=0.{break;}
        var taken=min(remaining,max(a.bed.xy,vec2<f32>(0.)));
        let packing=(a.bed.x+a.bed.y)/domain;if packing<=0.{break;}
        var amount=(taken.x+taken.y)/packing;
        if amount>free{taken*=free/amount;amount=free;}
        // A packet that takes the cell's last grains takes its bed whole: the
        // quotient above rounds, and a domain left without a grain on a
        // column's top would stop the column's flow.
        else if all(taken>=a.bed.xy){amount=min(domain,free);}
        if amount<=0.{break;}
        var bed_packet=vec4<f32>(taken,amount,amount*a.bed.w/domain);
        if amount>=domain{taken=a.bed.xy;bed_packet=a.bed;}
        let ambient=amount*vec3<f32>(max(0.,b.volume.water),grains(b))/free;
        put_mobile_volumes(ia,mobile_volumes(a)+vec4<f32>(ambient,0.));
        put_mobile_volumes(ib,mobile_volumes(b)-vec4<f32>(ambient,0.));
        cells[ia].bed=a.bed-bed_packet;cells[ib].bed=b.bed+bed_packet;
        let ma=mobile_volumes(a);let mb=mobile_volumes(b);
        // The powder goes with the white it is part of, both ways, the
        // ambient's free powder with it.
        var bed_powder=0.;if a.bed.x>0.{bed_powder=taken.x*a.powder.bed/a.bed.x;}
        var ambient_powder=vec2<f32>(0.);if mb.y>0.{ambient_powder=ambient.y*vec2<f32>(b.powder.mobile,b.powder.free)/mb.y;}
        cells[ia].powder=CellPowder(a.powder.mobile+ambient_powder.x,a.powder.bed-bed_powder,a.powder.free+ambient_powder.y);
        cells[ib].powder=CellPowder(b.powder.mobile-ambient_powder.x,b.powder.bed+bed_powder,b.powder.free-ambient_powder.y);
        let water_forward=vec2<f32>(0.);
        var water_back=vec2<f32>(0.);if mb.x>0.{water_back=ambient.x*b.momentum[0]/mb.x;}
        cells[ia].momentum[0]=a.momentum[0]-water_forward+water_back;
        cells[ib].momentum[0]=b.momentum[0]+water_forward-water_back;
        for(var phase=1u;phase<3u;phase++){
            var carried=vec2<f32>(0.);if mb[phase]>0.{carried=ambient[phase]*b.momentum[phase]/mb[phase];}
            cells[ia].momentum[phase]=a.momentum[phase]+carried;
            cells[ib].momentum[phase]=b.momentum[phase]-carried;
        }
        let mobile=mb.y+mb.z;var fraction=0.;if mobile>0.{fraction=(ambient.y+ambient.z)/mobile;}
        put_stress(ia,grain_stress(a)+fraction*grain_stress(b),a.stress.suction);
        put_stress(ib,(1.-fraction)*grain_stress(b),b.stress.suction);
        record_surface_path(xa,ya,xb,yb,path.gate,bed_packet.xy,-ambient.yz);
        moved+=taken;remaining=max(vec2<f32>(0.),remaining-taken);
    }return moved;
}
// The depth (m) of the layer that a flow of depth `depth` moves: at least a
// grain. While the hands shake the picture the lateral load pushes the
// surface to and fro without reworking it: the yield depth under that load
// (tens of millimetres at a steep face) is no moving layer.
fn moving_depth(depth:f32)->f32{
    if cfg.lateral!=0.{return GRAIN_LAYER;}return max(depth,GRAIN_LAYER);
}
// The surface flow of the columns `left` and left+1 in one update. Its
// depth and the flux it moved are kept in terrain[nx+left].
fn surface_pair(left:u32){
    if left+1u>=cfg.nx{return;}let right=left+1u;
    let previous=terrain[cfg.nx+left];
    // The history is cleared first: a pair that moves nothing leaves no
    // momentum for a later flow to inherit.
    terrain[cfg.nx+left]=vec4<f32>(0.);
    let hl=bed_height(left);let hr=bed_height(right);var slope=hr-hl;
    var donor=select(right,left,hl>=hr);var receiver=select(left,right,hl>=hr);
    var top=bed_top(donor);var i=top*cfg.nx+donor;var c=cells[i];var solid=c.bed.x+c.bed.y;
    if solid<=0.||c.bed.z<=0.{return;}
    var path=surface_path(donor,top,receiver,bed_open(receiver),select(GRAIN_DIAMETER.x,GRAIN_DIAMETER.y,c.bed.y>0.));
    // A covered face runs from the cover's lip, the donor's bed at the row
    // where its grains leave, down to the receiver's surface: the sand above
    // the lip is shut in and adds no slope. What arrives and what a stream
    // drags act on the column's top, not on this face.
    let covered=path.gate<top;var height=bed_height(donor);
    if covered{
        height=f32(path.gate)+occupied_wet_fraction(cells[path.gate*cfg.nx+donor]);
        slope=select(height-hl,hr-height,donor==left);
        i=path.source*cfg.nx+donor;c=cells[i];solid=c.bed.x+c.bed.y;
        if path.aperture<=0.||solid<=0.||c.bed.z<=0.{return;}
    }
    var share=c.bed.xy/solid;var packing=solid/c.bed.z;
    let receiving=cells[path.row*cfg.nx+receiver];
    let free=free_wet(receiving);if free<=0.{return;}
    let ambient=fractions(receiving)/free;
    // The packet changes places with the suspension, whose grains return
    // uphill: its weight counts against the suspension's density.
    let load=required_support(i,share,cfg.lateral)
        -(dot(ambient,RHO)-RHO.x)*vec2<f32>(-cfg.lateral,cfg.gravity)/packing;
    let impact=select(.5*(terrain[left].zw+terrain[right].zw),vec2<f32>(0.),covered);
    var bed_powder=0.;if c.bed.x>0.{bed_powder=clamp(c.powder.bed/c.bed.x,0.,1.);}
    // The moving layer is the column's top as it was laid, as deep as the
    // pair's last flow (peel). In a dense avalanche the grains of both
    // species are in contact and move together: the layer yields and runs
    // at the friction of its mixture, not each species at its own.
    var mixture=share;
    if path.source==top{
        let layer0=peel(donor,moving_depth(previous.x)/cfg.h*packing);
        if layer0.x+layer0.y>0.{mixture=layer0/(layer0.x+layer0.y);}
    }
    let rate=surface_rate(slope,height*cfg.h,packing,share,PowderShare(0.,bed_powder),white_powder(receiving),
        load,impact,ambient,previous,grain_friction(mixture));
    let flux=rate.domain_flux;let depth=rate.depth;
    // A layer that does not flow leaves no depth in the history: the depth
    // at which a still face would yield is no moving layer.
    if flux==0.{return;}
    // The flux's direction chooses the donor (a stream's traction can drive
    // the layer uphill); the rate is not solved again for it.
    let forward=flux>0.;
    donor=select(right,left,forward);receiver=select(left,right,forward);
    top=bed_top(donor);i=top*cfg.nx+donor;c=cells[i];solid=c.bed.x+c.bed.y;
    if solid<=0.||c.bed.z<=0.{return;}
    path=surface_path(donor,top,receiver,bed_open(receiver),select(GRAIN_DIAMETER.x,GRAIN_DIAMETER.y,c.bed.y>0.));
    if path.source!=top{
        c=cells[path.source*cfg.nx+donor];solid=c.bed.x+c.bed.y;
        if solid<=0.||c.bed.z<=0.{return;}
    }
    share=c.bed.xy/solid;packing=solid/c.bed.z;
    // The flow runs the column's top out last laid first (peel): what
    // arrived since the record was written, then the record's slices. An
    // update moves no more than the moving layer holds. Under a pocket the
    // bed's surface has no record of its own (the column's goes on over the
    // pocket): its sand leaves as the cell holds it.
    let amount=min(path.aperture*packing*abs(flux)*cfg.dt/cfg.h/cfg.h,moving_depth(depth)/cfg.h*packing);
    var requested=amount*share;if path.source==top{requested=peel(donor,amount);}
    let room=bed_height(donor)*packing/max(requested.x+requested.y,1e-30);
    let moved=transfer_surface(donor,receiver,requested*min(1.,room));
    let domain=(moved.x+moved.y)/packing;
    terrain[cfg.nx+left]=vec4<f32>(depth,select(-1.,1.,forward)*domain*cfg.h*cfg.h/cfg.dt,0.,0.);
}
@compute @workgroup_size(128) fn surface_even(@builtin(global_invocation_id) id:vec3<u32>){surface_pair(2u*id.x);}
@compute @workgroup_size(128) fn surface_odd(@builtin(global_invocation_id) id:vec3<u32>){surface_pair(2u*id.x+1u);}
