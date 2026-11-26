// The capture: the bed takes the sand that reaches its surface or rests on
// it. Once an update, after the first flow solve, each column reads a
// snapshot of the cells (`arrival`), turns sand into bed and lays its wet
// material back into its rows, the bed below. Gas is not touched.

@group(0) @binding(11) var<storage,read_write> arrival:array<Cell>;
// A piece of wet material on its way back into the column: `mobile` (water,
// fine, coarse) with its momenta and the grains' stress, `bed` as a cell's,
// and the white's powder: in the free grains (x), in the bed (y), and of the
// free grains' powder the free powder (z).
struct ArrivalParcel {
    mobile:vec3<f32>,bed:vec4<f32>,momentum:array<vec2<f32>,3>,
    stress:vec3<f32>,powder:vec3<f32>
}
fn parcel_volume(p:ArrivalParcel)->f32{return p.mobile.x+p.mobile.y+p.mobile.z+p.bed.z;}
fn parcel_scaled(p:ArrivalParcel,f:f32)->ArrivalParcel{
    var q:ArrivalParcel;q.mobile=f*p.mobile;q.bed=f*p.bed;
    for(var k=0u;k<3u;k++){q.momentum[k]=f*p.momentum[k];}
    q.stress=f*p.stress;q.powder=f*p.powder;return q;
}
fn parcel_sum(a:ArrivalParcel,b:ArrivalParcel)->ArrivalParcel{
    var q:ArrivalParcel;q.mobile=a.mobile+b.mobile;q.bed=a.bed+b.bed;
    for(var k=0u;k<3u;k++){q.momentum[k]=a.momentum[k]+b.momentum[k];}
    q.stress=a.stress+b.stress;q.powder=a.powder+b.powder;return q;
}
fn old_bed_parcel(c:Cell)->ArrivalParcel{
    var p:ArrivalParcel;p.bed=c.bed;p.powder=vec3<f32>(0.,c.powder.bed,0.);
    return p;
}
// The volume the captured grains occupy as bed: a front lays them down at
// the deposit's packing; a `whole` free wet part joins the bed as it lies,
// with all its water. `fraction` is the captured share of each species.
fn captured_domain(c:Cell,fraction:vec2<f32>,whole:bool)->f32{
    if whole{return free_wet(c);}
    let solid=fraction*grains(c);
    return (solid.x+solid.y)/BED_PACKING;
}
// The cell volumes of the white's bound powder and free powder.
fn white_powder_volumes(c:Cell)->vec2<f32>{
    let share=white_powder(c);return grains(c).x*vec2<f32>(share.total-share.free,share.free);
}
// `powder` is the captured share of the white's bound powder, which reaches
// the front with its grit, and of its free powder, which reaches it at its
// own speed. The deposit binds what it takes.
fn captured_parcel(c:Cell,fraction:vec2<f32>,powder:vec2<f32>,whole:bool)->ArrivalParcel{
    var p:ArrivalParcel;let solid=fraction*grains(c);let amount=solid.x+solid.y;
    let domain=captured_domain(c,fraction,whole);
    p.bed=vec4<f32>(solid,domain,domain-amount);
    p.powder=vec3<f32>(0.,dot(powder,white_powder_volumes(c)),0.);
    return p;
}
// What stays mobile of cell c, less `removed_water` taken as pore water; the
// grains' stress stays with the grains that stay.
fn free_parcel(c:Cell,fraction:vec2<f32>,powder:vec2<f32>,removed_water:f32)->ArrivalParcel{
    var p:ArrivalParcel;let kept=1.-fraction;let g=grains(c);
    p.mobile=vec3<f32>(max(0.,c.volume.water-removed_water),kept*g);
    let free_water=max(0.,c.volume.water);
    if free_water>0.{p.momentum[0]=(p.mobile.x/free_water)*c.momentum[0];}
    p.momentum[1]=kept.x*c.momentum[1];p.momentum[2]=kept.y*c.momentum[2];
    let left=(vec2<f32>(1.)-powder)*white_powder_volumes(c);
    p.powder=vec3<f32>(left.x+left.y,0.,left.y);
    var solid=1.;if g.x+g.y>0.{solid=dot(kept,g)/(g.x+g.y);}
    p.stress=solid*grain_stress(c).xyz;return p;
}
// The snapshot the capture reads: the cells after the first mechanics.
@compute @workgroup_size(128) fn capture_bed_source(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x<cfg.n{arrival[id.x]=cells[id.x];}
}

// The row of a body, sand in contact that the plates hold still, one or two
// rows over the bed's open cell with only water and loose grains between;
// else, and over a pocket, `open`. The gap is less than two cells: the
// body's face breaches into it (Van Rhee; roof_breach only reaches a face
// over a cell without bed) and its grains settle there, so that within a
// second the body rests on the bed.
fn held_over_gap(x:u32,open:u32)->u32{
    for(var row=open;row<min(open+3u,cfg.ny);row++){
        let c=arrival[row*cfg.nx+x];if pocket_row(c){return open;}
        let g=grains(c);if g.x+g.y<LOOSE_PACKING*free_wet(c){continue;}
        if row>open&&held_still(row*cfg.nx+x){return row;}
        return open;
    }
    return open;
}
// The capture in column x. Writes the reaction that stopped the captured
// material to terrain[x].zw and its relocation to the faces' displacement.
fn exchange_from_source(x:u32){
    terrain[x].z=0.;terrain[x].w=0.;
    terrain[2u*cfg.nx+x]=vec4<f32>(0.);
    let column=bed_column(x);if !column.connected{return;}
    let bi=column.top*cfg.nx+x;let b=arrival[bi].bed;let bs=b.x+b.y;
    let normal=surface_normal(i32(x));
    // The top cell is released when it would need tension to stay, after
    // accounting for out-of-plane glass-plate Coulomb friction when tilted
    // toward flat (`plate_gravity > 0`). Support is the vertical one, here and
    // below: against the hands' shake the plates hold the deposit under its top
    // few millimetres, which the surface flow moves. Sand on a pocket rests on
    // it (on_pocket).
    if bs>0.&&!on_pocket(x,column.top){
        let share=b.xy/bs;
        let plate_hold=.4*dot(share,RHO.yz-vec2<f32>(RHO.x))*plate_gravity();
        if dot(required_support(bi,share,0.),normal)+plate_hold<0.{release_bed(bi);return;}
    }
    var first=column.open;
    // Where the front starts. A free layer thinner than a grain on the bed's
    // top cell holds no grain, and a face that gas closes to the finest
    // grain (a bubble trapped on the bed) passes none: the sand above meets
    // the bed at the open cell's top, and the front starts in the cell
    // above; the layer is enclosed in the deposit once the front takes sand.
    // Over a film without bed, sand in contact that the plates hold still
    // rests as on the floor and joins the bed as it lies, the film trapped
    // under it. A body held over a gap rests on the bed too: the front
    // starts at the body and the gap's rows are enclosed.
    var enclosed=0u;var gap=false;
    if first+1u<cfg.ny{
        let film=arrival[first*cfg.nx+x];let gf=grains(film);
        let above=arrival[(first+1u)*cfg.nx+x];let ga=grains(above);let fa=free_wet(above);
        let resting=gf.x+gf.y<=0.&&fa>0.&&ga.x+ga.y>=LOOSE_PACKING*fa&&held_still((first+1u)*cfg.nx+x);
        let closed=free_wet(film)*cfg.h<GRAIN_LAYER||!grain_crosses(2u*(first*cfg.nx+x)+1u,GRAIN_DIAMETER.x);
        if closed&&(film.bed.z>0.||resting){enclosed=select(0u,1u,film.bed.z>0.);first=first+1u;}
        else{let body=held_over_gap(x,first);if body>first{enclosed=body-first;first=body;gap=true;}}
    }
    let i=first*cfg.nx+x;let c=arrival[i];let m=grains(c);
    let solid=m.x+m.y;let free=free_wet(c);if solid<=0.||free<=0.{return;}
    // Never lock a supercritical cliff or overhang into rigid static bed:
    // keep it mobile so it slumps naturally under full 2D granular mechanics.
    if bed_cliff_unstable(i32(x),i32(first)){return;}
    let density=solid/free;let rests=on_pocket(x,first);
    let momentum=arriving_momentum(i);
    if solid>1e-30{terrain[2u*cfg.nx+x].w=dot(momentum,normal)/solid;}
    // `reach`: the distance a species moves toward the sloped surface in
    // this update, in cell heights. The white moves at the volume mean of
    // its grit (with its bound powder) and its free powder; each of the two
    // slips through the water by the ratio of its settling speed to that
    // mean (powder_drift), the free powder some twenty times slower.
    let water_in=incident_velocity(i,0u);let white_in=incident_velocity(i,1u);
    let mean=white_speed(white_powder(c));
    let grit_in=water_in+(TERMINAL_SPEED.x/mean)*(white_in-water_in);
    let reach=vec2<f32>(max(0.,-dot(grit_in,normal)),max(0.,-dot(incident_velocity(i,2u),normal)))
        *cfg.dt/(cfg.h*normal.y);
    let powder_in=water_in+(POWDER_SPEED/mean)*(white_in-water_in);
    let powder_reach=max(0.,-dot(powder_in,normal))*cfg.dt/(cfg.h*normal.y);
    let powder=white_powder_volumes(c).y;let grit=m.x-powder;
    // Grains in contact that the plates hold still on the bed are deposit as
    // they lie: the whole free wet part joins the bed at its own packing.
    // Nothing reaches the front from them, so a front would never pass them.
    var whole=density>=LOOSE_PACKING&&density<BED_PACKING&&held_still(i);
    // On a pocket the lining is a packing even loosened below contact
    // (caged): held still, it is deposit; denser than the deposit, it is
    // deposit as it lies, for laid at the deposit's packing it would spill
    // into the row that the bubble closes and cancel the capture.
    if rests{whole=density>=FREEZING_PACKING&&(held_still(i)||density>=BED_PACKING);}
    // A looser suspension feeds a rising front. Sand as dense as the deposit
    // and sand at rest are taken whole if they press on the bed; on a pocket
    // or over a gap they rest on what is under them and are not tested.
    var fraction=vec2<f32>(1.);var powder_fraction=vec2<f32>(1.);
    if density<BED_PACKING&&!whole{
        // Rankine-Hugoniot for each species s: phi_b x_s Hdot = phi_s (Hdot +
        // w_s). The front rises by sum phi_s reach_s/(phi_b - phi_a) and
        // takes, of each species, the suspension it sweeps and what reaches
        // it: the deposit takes the composition of the arriving flux. The
        // white's grit and its free powder are two species here.
        let rise_num=grit*reach.x+powder*powder_reach+m.y*reach.y;
        let rise_den=free*(BED_PACKING-density);
        var rise=0.;
        if rise_num>0.{if rise_den>0.{rise=rise_num/rise_den;}else{rise=free;}}
        fraction=min(vec2<f32>(1.),(vec2<f32>(rise)+reach)/free);
        powder_fraction=vec2<f32>(fraction.x,min(1.,(rise+powder_reach)/free));
        if m.x>0.{fraction.x=(grit*fraction.x+powder*powder_fraction.y)/m.x;}
    }else if !rests&&!gap&&dot(required_support(i,m/solid,0.),normal)<=0.{return;}
    let amount=dot(fraction,m);
    if amount<=0.{return;}
    // The captured share of each species in each row (up to MAX_SIDE).
    var captured:array<vec2<f32>,MAX_SIDE>;captured[first]=fraction;
    var captured_powder:array<vec2<f32>,MAX_SIDE>;captured_powder[first]=powder_fraction;
    var last=first;var overflow=0.;if !whole{overflow=amount/BED_PACKING-free_wet(c);}
    // A front that fills its cell goes on into the rows above and takes the
    // suspension it sweeps. Where gas closes a face to the coarse grain, or
    // the tank's top leaves no room, nothing is captured in this update.
    for(var row=first+1u;row<cfg.ny&&overflow>0.;row++){
        if bed_cliff_unstable(i32(x),i32(row)){return;}
        let f=2u*((row-1u)*cfg.nx+x)+1u;
        if !grain_crosses(f,GRAIN_DIAMETER.y){return;}
        let a=arrival[row*cfg.nx+x];let wet=free_wet(a);let g=grains(a);let s=g.x+g.y;
        let spare=wet-s/BED_PACKING;
        if spare>0.&&overflow<spare{
            captured[row]=vec2<f32>(overflow/spare);overflow=0.;
        }else{captured[row]=vec2<f32>(1.);overflow-=spare;}
        captured_powder[row]=vec2<f32>(captured[row].x);
        last=row;
    }
    if overflow>0.{return;}
    // Enclosed rows are filled from above: the parcels are laid from their
    // first row, so the captured sand settles into them at the deposit's
    // packing and their water rises. Joined to the bed with its water, each
    // layer a front passed would thin the deposit below contact.
    let start=first-enclosed;
    for(var row=start;row<first;row++){captured[row]=vec2<f32>(1.);captured_powder[row]=vec2<f32>(1.);}
    var pore_total=0.;var impulse=vec2<f32>(0.);
    for(var row=start;row<=last;row++){
        let a=arrival[row*cfg.nx+x];let g=captured[row]*grains(a);let s=g.x+g.y;
        pore_total+=captured_domain(a,captured[row],whole&&row>=first)-s;
        terrain[2u*cfg.nx+x].y+=s;
        // The captured grains stop: their momentum at the arriving velocity
        // is the impulse on the surface.
        let j=row*cfg.nx+x;
        impulse+=RHO.y*g.x*incident_velocity(j,1u)+RHO.z*g.y*incident_velocity(j,2u);
    }
    // The pore water comes from the rows' water, bottom up, and stops too.
    var water_needed=pore_total;var pore_momentum=vec2<f32>(0.);
    for(var row=start;row<=last;row++){
        let a=arrival[row*cfg.nx+x];let take=min(water_needed,max(0.,a.volume.water));
        let free_water=max(0.,a.volume.water);
        if free_water>0.{pore_momentum+=(take/free_water)*a.momentum[0];}
        water_needed=max(0.,water_needed-take);
    }
    // The rows are refilled from the bottom: each row's old bed and captured
    // sand in turn, then each row's remaining suspension. Only the snapshot
    // is read, and a column writes its own cells.
    var bed_row=start;var bed_kind=0u;var free_row=start;var bed_done=false;
    var piece:ArrivalParcel;var removed_water=pore_total;
    var bed_flux=vec2<f32>(0.);var mobile_flux=vec2<f32>(0.);
    for(var row=start;row<=last;row++){
        let j=row*cfg.nx+x;var output:ArrivalParcel;
        // The last row takes all that remains: no residue is discarded.
        var room=select(liquid(arrival[j]),1.e10,row==last);
        for(var segment=0u;segment<4u*cfg.ny+4u;segment++){
            if room<=0.{break;}
            if parcel_volume(piece)<=0.{
                if !bed_done&&bed_row<=last{
                    let source=arrival[bed_row*cfg.nx+x];
                    if bed_kind==0u{piece=old_bed_parcel(source);bed_kind=1u;}
                    else{piece=captured_parcel(source,captured[bed_row],captured_powder[bed_row],whole&&bed_row>=first);bed_kind=0u;bed_row++;}
                    continue;
                }
                bed_done=true;
                if free_row>last{break;}
                let source=arrival[free_row*cfg.nx+x];
                let take=min(removed_water,max(0.,source.volume.water));
                removed_water=max(0.,removed_water-take);
                piece=free_parcel(source,captured[free_row],captured_powder[free_row],take);free_row++;continue;
            }
            let available=parcel_volume(piece);let take=min(room,available);
            if take==available{output=parcel_sum(output,piece);piece=parcel_scaled(piece,0.);}
            else{let share=take/available;output=parcel_sum(output,parcel_scaled(piece,share));piece=parcel_scaled(piece,1.-share);}
            room=max(0.,room-take);
        }
        // Only relocation is a displacement, not the change of class: what
        // the rows up to this one held less what they are given crossed this
        // row's upper face.
        let source=arrival[j];let captured_grains=captured[row]*grains(source);
        bed_flux+=source.bed.xy+captured_grains-output.bed.xy;
        mobile_flux+=(1.-captured[row])*grains(source)-output.mobile.yz;
        if row<last{faces[2u*j+1u].displacement+=vec4<f32>(bed_flux,mobile_flux)*cfg.h/cfg.dt;}
        cells[j].volume.water=output.mobile.x;cells[j].volume.fine=output.mobile.y;cells[j].volume.coarse=output.mobile.z;
        cells[j].bed=output.bed;cells[j].powder=CellPowder(output.powder.x,output.powder.y,output.powder.z);
        for(var phase=0u;phase<3u;phase++){cells[j].momentum[phase]=output.momentum[phase];}
        put_stress(j,vec4<f32>(output.stress,min(cells[j].stress.load,output.stress.x)),cells[j].stress.suction);
    }
    impulse+=RHO.x*pore_momentum;
    terrain[x].z=-impulse.x*cfg.h/cfg.dt;terrain[x].w=-impulse.y*cfg.h/cfg.dt;
}
