// The plug descent: the voids that a joint's take leaves in a packed body
// rise through it and the body comes down as a plug, the kinematic model of
// silo flow as a transport step (plug_descent). plug_record runs before the
// carry (stage TransportPrepare), plug_descent after it (stage TraceGas).

// The walk's threads and column-owned row handoff capacity (covers both
// 216x144 and 144x216).
const PLUG_THREADS:u32=256u;
const PLUG_COLS:u32=216u;
const PLUG_WORDS:u32=7u;
fn plug_deposit(c:Cell)->bool{return max(c.bed.x,0.)+max(c.bed.y,0.)>0.;}
// A packed body's cell: free grains at the critical packing of their wet
// part or more, or a body's last layer (body_faces); no deposit, and no
// bubble (gas a grain layer thick), which interrupts the skeleton: by its
// wet part alone its cell would count as packed and be fed grains until dry.
fn plug_body(i:u32)->bool{
    let c=cells[i];if plug_deposit(c)||c.volume.gas*cfg.h>=GRAIN_LAYER{return false;}
    let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if solid<=0.||wet<=0.{return false;}
    if solid>=CRITICAL_PACKING*wet{return true;}
    let entry=scratch[BODY_ROW*cfg.n+i];
    return entry.y>=0.&&entry.x<1.;
}
// Sand that can stand in a chain: no bubble and no deposit.
fn plug_sand(c:Cell)->bool{
    return !plug_deposit(c)&&c.volume.gas*cfg.h<GRAIN_LAYER&&teeth_sandy(c);
}
// A link of a chain: a body cell, or the slab's loose sand that stands on
// one through sand alone (a funnel's floor and walls, a column loosened by
// its deficits), which a stream takes first: out of the chain it would
// stay while the packed cell under it drained to water. A joint's column
// is no link: its sand slides at the chain's rate (joint_mouth).
fn plug_link(i:u32)->bool{
    if plug_body(i){return true;}
    if !plug_sand(cells[i])||column_cell(i){return false;}
    var p=point(i)-axis(1u);
    while valid(p){
        let j=at(p);
        if plug_body(j){return true;}
        if !plug_sand(cells[j])||column_cell(j){return false;}
        p-=axis(1u);
    }
    return false;
}
// The packing up to which sand counts as at the critical state: a margin of
// rounding over the critical packing, for cells that settled to it.
const DILATED_PACKING:f32=1.002*CRITICAL_PACKING;
// Whether the sand of cell i has given way to a draw from below and flows
// with it: sand at the critical packing or looser, or packed sand that the
// draw has dilated (SET_IN_ROW, plug_set_in). Packed sand cannot follow a
// draw at once: its grains must part first, and they part only as water
// enters between them.
fn plug_flows(i:u32)->bool{
    let c=cells[i];let g=grains(c);let solid=g.x+g.y;
    return solid<=DILATED_PACKING*(max(c.volume.water,0.)+solid)||geometry[SET_IN_ROW*cfg.n+i].x>=1.;
}
// A packed link that is drawn on for `want` gives way a little more. The
// front that dilates it to the critical packing crosses its cell at Van
// Rhee's rate for a roof (the breaching of a face that looks down: water
// enters no faster), and no faster than the draw takes up the grains that
// the dilation frees, the cell's margin over the critical packing (with the
// breaching law's floor). A column H high so follows a steady draw after H
// over that rate, and a draw that moves on before then leaves the column
// standing as it was.
fn plug_set_in(i:u32,c:Cell,want:f32){
    let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if solid<=0.||wet<=0.{return;}
    let phi=min(solid/wet,CLOSE_PACKING);
    let front=breaching_rate(g,white_powder(c),phi)*inclination_factor(-1.,grain_friction(g))*cfg.dt/cfg.h;
    let freed=max(phi-CRITICAL_PACKING,.01)*wet;
    geometry[SET_IN_ROW*cfg.n+i].x=min(geometry[SET_IN_ROW*cfg.n+i].x+min(want/freed,front),1.);
}
// scratch[PLUG_ROW n + i] = (fine, coarse, link, wet) before the carry:
// the cell's free grains, whether it is a link, and its wet part. Clears
// PLUG_INFLOW_ROW, where advect_groups sums the grains the carry brings in.
@compute @workgroup_size(128) fn plug_record(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    let c=cells[i];let g=grains(c);
    scratch[PLUG_ROW*cfg.n+i]=vec4<f32>(g.x,g.y,select(0.,1.,plug_link(i)),max(c.volume.water,0.)+g.x+g.y);
    scratch[PLUG_INFLOW_ROW*cfg.n+i]=vec4<f32>(0.);
}
// Momentum scaled with its phase's volume, for a reduction only: scaled
// up, the leftover of a near-empty phase is amplified.
fn plug_scaled(m:vec2<f32>,old:f32,now:f32)->vec2<f32>{
    if old<=0.{return vec2<f32>(0.);}
    return m*(now/old);
}
// Per-row supply scratch for `plug_supply` (row y+1's supply values).
var<workgroup> plug_supply_row:array<f32,PLUG_COLS>;
// Bitmask of rows holding a link (up to 288 rows) and per-column handoff
// from row y-1 to row y for `plug_walk`. Staging all cross-thread handoff
// in workgroup memory lets each thread exclusively own its columns' cells
// across rows with pure workgroupBarrier() synchronisation (no storageBarrier).
var<workgroup> plug_marks:array<atomic<u32>,PLUG_WORDS>;
var<workgroup> plug_linked:array<u32,PLUG_WORDS>;
var<workgroup> pw_ask:array<f32,PLUG_COLS>;
var<workgroup> pw_down:array<vec4<f32>,PLUG_COLS>;
var<workgroup> pw_up:array<vec4<f32>,PLUG_COLS>;
var<workgroup> pw_vel:array<vec4<f32>,PLUG_COLS>;
var<workgroup> pw_pow:array<u32,PLUG_COLS>;
// The share of an ask that falls to the cell straight above the asker; a
// wall's share folds into it (the walls are mirrors).
fn plug_centre_weight(x:i32,k:f32)->f32{
    return 1.-2.*k+select(0.,k,x==0)+select(0.,k,x==i32(cfg.nx)-1);
}
// What a link gives the three cells below it, what it asks of the row
// above and the surplus it hands the three cells above, with fine shares.
struct PlugStep{left:f32,centre:f32,right:f32,fine:f32,ask:f32,up:vec3<f32>,up_fine:f32}
// The shares (left, centre, right) in which link i hands its surplus up:
// as the carry's net grain inflow through its left, its bottom and top,
// and its right faces; a side's share folds into the centre where no link
// stands diagonally above. The surplus returns the way it came: sent
// straight up, the surplus of a slot that the drawdown fills from both
// sides would climb the slot's own column and pack it dry.
fn plug_return(i:u32,x:i32,y:i32)->vec3<f32>{
    let flow=max(scratch[PLUG_INFLOW_ROW*cfg.n+i],vec4<f32>(0.));
    var w=vec3<f32>(flow.x,flow.z+flow.w,flow.y);
    if x==0||scratch[PLUG_ROW*cfg.n+at(vec2<i32>(x-1,y+1))].z<=0.{w.y+=w.x;w.x=0.;}
    if x+1>=i32(cfg.nx)||scratch[PLUG_ROW*cfg.n+at(vec2<i32>(x+1,y+1))].z<=0.{w.y+=w.z;w.z=0.;}
    let total=w.x+w.y+w.z;
    if total<=0.{return vec3<f32>(0.,1.,0.);}
    return w/total;
}
fn plug_room(c:Cell)->f32{
    let g=grains(c);let solid=g.x+g.y;
    return max(CLOSE_PACKING*(max(c.volume.water,0.)+solid)-solid,0.);
}
// Loose sand of a chain, which keeps what it gains: by the record its
// grains did not touch, and it is not a layer hanging under its body.
fn plug_settles(i:u32,record:vec4<f32>)->bool{
    if record.x+record.y>=CRITICAL_PACKING*record.w{return false;}
    let entry=scratch[BODY_ROW*cfg.n+i];
    return !(entry.x<1.&&entry.y==3.);
}
// The grains the open cell over a chain's top link can take until they
// touch: water without a bubble, deposit or a joint's column.
fn plug_open_room(j:u32)->f32{
    let c=cells[j];
    if plug_deposit(c)||c.volume.gas*cfg.h>=GRAIN_LAYER||column_cell(j){return 0.;}
    let g=grains(c);let solid=g.x+g.y;
    return max(CRITICAL_PACKING*(max(c.volume.water,0.)+solid)-solid,0.);
}
// The fine share of a cell's grains by the record, -1 without grains.
fn plug_share(x:i32,y:i32)->f32{
    let r=scratch[PLUG_ROW*cfg.n+at(vec2<i32>(x,y))];let solid=r.x+r.y;
    if solid<=0.{return -1.;}return r.x/solid;
}
// The fine share `face` of `amount`, as far as g holds each species.
fn plug_taken(g:vec2<f32>,amount:f32,face:f32)->f32{
    if amount<=0.{return 0.;}
    return clamp(face,max(0.,1.-g.y/amount),min(1.,g.x/amount));
}
// The fine share of the `amount` of a cell's grains g that crosses a face
// to join grains of share `ahead`, followed in through the opposite face
// by grains of share `behind` (by the record; negative without grains). A
// plug's grains keep their neighbours, so the descent must mix nothing: a
// cell whose share lies between the two holds the boundary of two layers
// and gives the layer ahead, as far as leaves its share between its own
// and `behind` (the limited downwind flux of Despres and Lagoutiere 2001,
// which carries a step unchanged); an extreme of its column gives its mean.
fn plug_face(g:vec2<f32>,amount:f32,ahead:f32,behind:f32)->f32{
    let solid=g.x+g.y;
    if amount<=0.||solid<=0.{return 0.;}
    let own=g.x/solid;
    var face=own;
    if ahead>=0.{
        face=ahead;
        if behind>=0.{
            let rest=max(solid-amount,0.)/amount;
            face=clamp(ahead,own-rest*(max(own,behind)-own),own+rest*(own-min(own,behind)));
        }
    }
    return plug_taken(g,amount,face);
}
// Cell i keeps fine f and coarse co of its grains g; their momenta and the
// white's powder scale by species.
fn plug_keep(i:u32,c:Cell,g:vec2<f32>,f:f32,co:f32){
    var kf=0.;if g.x>0.{kf=f/g.x;}var kc=0.;if g.y>0.{kc=co/g.y;}
    cells[i].volume.fine=f;cells[i].volume.coarse=co;
    cells[i].momentum[1]=c.momentum[1]*kf;cells[i].momentum[2]=c.momentum[2]*kc;
    cells[i].powder.mobile=c.powder.mobile*kf;cells[i].powder.free=c.powder.free*kf;
}
// Link i at (x, y) gives the row below what it asked (`asked`: the asks of
// the cells below it to the left, straight below and to the right; the
// grains leave with their velocity and powder, and water takes their place
// with the cell's water momentum unchanged), then sets its ask and its
// surplus.
//
// It gives only if its sand has given way (plug_flows), and then in the
// measure of its supply (SET_IN_ROW.y, plug_descent): the share of a draw on
// it that sand which has given way passes on to the top, where the surface
// can sink. The rest of the draw stays where it was drawn, at the foot,
// whose face goes on breaching: no void opens inside a packing that cannot
// fill it, and nothing in it is loosened. A link that has not given way
// gives nothing and is dilated by the draw (plug_set_in).
fn plug_give(i:u32,x:i32,y:i32,record:vec4<f32>,k:f32,asked:vec3<f32>)->PlugStep{
    let nx=i32(cfg.nx);let c=cells[i];var out=PlugStep();
    if plug_deposit(c){return out;}
    let g=grains(c);let solid=g.x+g.y;let water=max(c.volume.water,0.);var given=0.;var want=0.;
    let flows=plug_flows(i);
    if y>0&&solid>0.{
        var want_left=0.;var want_right=0.;
        if x>0{want_left=k*asked.x;}
        if x+1<nx{want_right=k*asked.z;}
        var want_centre=plug_centre_weight(x,k)*asked.y;
        // Two links side by side each owe the cell under the other its
        // diagonal share, in the measure of their own supply: only the
        // difference of what they give crosses between their columns, and
        // each gives the rest to the cell under itself. The voids' sideways
        // spread is a velocity (u = -K dv/dx, zero where the columns descend
        // alike), not an exchange of grains. (A neighbour that has not given
        // way has no supply and nets nothing.)
        let supply=geometry[SET_IN_ROW*cfg.n+i].y;
        if flows&&supply>0.{
            if x>0{
                let j=at(vec2<i32>(x-1,y));
                if scratch[PLUG_ROW*cfg.n+j].z>0.{
                    let crossed=min(supply*want_left,geometry[SET_IN_ROW*cfg.n+j].y*k*asked.y)/supply;
                    want_left-=crossed;want_centre+=crossed;
                }
            }
            if x+1<nx{
                let j=at(vec2<i32>(x+1,y));
                if scratch[PLUG_ROW*cfg.n+j].z>0.{
                    let crossed=min(supply*want_right,geometry[SET_IN_ROW*cfg.n+j].y*k*asked.y)/supply;
                    want_right-=crossed;want_centre+=crossed;
                }
            }
        }
        want=want_left+want_centre+want_right;
        // A trace of sand is what rounding left of a difference: it gives
        // nothing.
        if flows{given=min(want*supply,solid);if solid<TRACE_VOLUME{given=0.;}}
        else if want>0.{plug_set_in(i,c,want);}
        if want>0.{let share=given/want;out.left=want_left*share;out.centre=want_centre*share;out.right=want_right*share;}
        if given>0.{
            // The grains given join the cells below (their shares weighted
            // by what each takes); the cell above's follow them in.
            var ahead=0.;var weight=0.;
            let parts=vec3<f32>(out.left,out.centre,out.right);
            for(var d=0;d<3;d++){
                if parts[d]<=0.{continue;}
                let below=plug_share(x+d-1,y-1);
                if below>=0.{ahead+=parts[d]*below;weight+=parts[d];}
            }
            if weight>0.{ahead/=weight;}else{ahead=-1.;}
            var behind=-1.;if y+1<i32(cfg.ny){behind=plug_share(x,y+1);}
            out.fine=plug_face(g,given,ahead,behind);
            plug_keep(i,c,g,g.x-given*out.fine,g.y-given*(1.-out.fine));
            cells[i].volume.water=water+given;
        }
    }
    var ask=given;
    // The deficit is what restores the cell's packing before the carry over
    // its wet part now: gas that took part of the cell took its water, not
    // room for grains.
    let above=at(vec2<i32>(x,y+1));
    if y+1<i32(cfg.ny)&&record.w>0.{
        let linked=scratch[PLUG_ROW*cfg.n+above].z>0.;
        let restore=(record.x+record.y)*min((water+solid)/record.w,1.);
        // A link that has given way asks for the whole draw on it and for
        // what the carry took from it: the links over it give that in the
        // measure of their own supply, which is what it gave. One that has
        // not given way takes the draw on itself, and only one that nothing
        // draws on asks for what it lacks.
        if linked{
            if flows{ask=max(want+restore-solid,0.);}
            else if want>0.{ask=0.;}
            else{ask=max(restore-solid,0.);}
        }
        // The link hands up what the carry left it beyond that packing. The
        // mechanics slides the body over a joint at the voids' speed
        // (void_speed), the mouth passes only its breaching rate, and a
        // packed skeleton does not compact under tens of pascals: kept, the
        // surplus would pack the plug from its mouth upward. Only sand
        // that was packed, or hangs under its body, holds its recorded
        // packing: loose sand that gains grains is settling and keeps them
        // until they touch, and a chain's top link under open water hands
        // its surplus into that water. A receiver hears from up to three
        // givers, so each gives it at most a third of its room.
        let left_now=solid-given;
        var hold=restore;
        if plug_settles(i,record){hold=max(restore,CRITICAL_PACKING*(water+solid));}
        var surplus=min(max(left_now-hold,0.),left_now);
        var shares=vec3<f32>(0.,1.,0.);
        if surplus>0.{
            if linked{
                shares=plug_return(i,x,y);
                for(var d=0;d<3;d++){
                    if shares[d]>1e-12{surplus=min(surplus,plug_room(cells[at(vec2<i32>(x+d-1,y+1))])/(3.*shares[d]));}
                }
            }else{surplus=min(surplus,plug_open_room(above));}
        }
        if surplus>0.{
            let now=cells[i];let gn=grains(now);
            // The surplus joins the cell above (a side donor for a diagonal
            // part), and the cell below's follows it in.
            var ahead=0.;var weight=0.;let up_share=plug_share(x,y+1);
            if up_share>=0.{ahead+=shares.y*up_share;weight+=shares.y;}
            if shares.x>0.{let side=plug_share(x-1,y);if side>=0.{ahead+=shares.x*side;weight+=shares.x;}}
            if shares.z>0.{let side=plug_share(x+1,y);if side>=0.{ahead+=shares.z*side;weight+=shares.z;}}
            if weight>0.{ahead/=weight;}else{ahead=-1.;}
            var behind=-1.;if y>0{behind=plug_share(x,y-1);}
            let fine=plug_face(gn,surplus,ahead,behind);
            plug_keep(i,now,gn,gn.x-surplus*fine,gn.y-surplus*(1.-fine));
            cells[i].volume.water=now.volume.water+surplus;
            out.up=surplus*shares;out.up_fine=fine;
        }
    }
    // The refill replaces water: no more than the cell holds.
    out.ask=min(ask,water+given);
    return out;
}
// Cell r takes the parts that the givers at columns x-1, x and x+1 of row
// y-1 pass it, with their fine shares, velocities, and powder shares already
// handed across workgroup memory: the grains arrive with their givers'
// velocities and powder, and the same volume of its water leaves.
fn plug_receive_local(
    r:u32,
    parts:vec3<f32>,
    fines:vec3<f32>,
    vf:array<vec2<f32>,3>,
    vc:array<vec2<f32>,3>,
    pw:array<vec2<f32>,3>,
){
    var received=0.;var fine=0.;var coarse=0.;var fine_momentum=vec2<f32>(0.);var coarse_momentum=vec2<f32>(0.);
    var powder=vec2<f32>(0.);
    for(var d=0;d<3;d++){
        let part=parts[d];if part<=0.{continue;}
        let part_fine=part*fines[d];let part_coarse=part-part_fine;
        received+=part;fine+=part_fine;coarse+=part_coarse;
        fine_momentum+=part_fine*vf[d];
        powder+=part_fine*pw[d];
        coarse_momentum+=part_coarse*vc[d];
    }
    if received<=0.{return;}
    let c=cells[r];let g=grains(c);let water=max(c.volume.water,0.);
    cells[r].volume.fine=g.x+fine;cells[r].volume.coarse=g.y+coarse;
    cells[r].volume.water=max(water-received,0.);
    cells[r].momentum[0]=plug_scaled(c.momentum[0],water,max(water-received,0.));
    cells[r].momentum[1]=c.momentum[1]+fine_momentum;cells[r].momentum[2]=c.momentum[2]+coarse_momentum;
    cells[r].powder.mobile=c.powder.mobile+powder.x;cells[r].powder.free=c.powder.free+powder.y;
}
// First half of the kinematic silo walk: from the ceiling down, each cell's
// supply (the share of a draw on it that reaches the top through links that
// have given way). Row y+1's supply is kept in workgroup memory so `geometry`
// is write-only here; the dispatch boundary before `plug_walk` publishes it.
// Every column fits in one thread (`PLUG_COLS = 216 <= PLUG_THREADS = 256`).
@compute @workgroup_size(PLUG_THREADS) fn plug_supply(@builtin(local_invocation_id) id:vec3<u32>){
    let nx=i32(cfg.nx);let ny=i32(cfg.ny);let k=VOID_SPREAD/cfg.h;let t=id.x;
    let in_col=t<cfg.nx;let x=i32(t);
    if in_col{plug_supply_row[t]=1.;}
    workgroupBarrier();
    let centre_w=select(0.,plug_centre_weight(x,k),in_col);
    for(var y=ny-1;y>=0;y--){
        var s_out=0.;var l_out=0.;
        if in_col{
            let i=at(vec2<i32>(x,y));
            let link=scratch[PLUG_ROW*cfg.n+i].z;var supply=0.;
            if link<=0.{
                let c=cells[i];
                if !plug_deposit(c)&&c.volume.gas*cfg.h<GRAIN_LAYER&&!teeth_sandy(c){
                    supply=1.;
                    if y+1<ny{
                        let j=at(vec2<i32>(x,y+1));
                        if scratch[PLUG_ROW*cfg.n+j].z>0.||plug_supply_row[t]<1.{supply=0.;}
                    }
                }
            }else if plug_flows(i){
                let has_above=y+1<ny;
                if x>0{supply+=k*select(1.,plug_supply_row[t-1u],has_above);}
                supply+=centre_w*select(1.,plug_supply_row[t],has_above);
                if x+1<nx{supply+=k*select(1.,plug_supply_row[t+1u],has_above);}
            }
            s_out=supply;l_out=link;
        }
        workgroupBarrier();
        if in_col{
            let i=at(vec2<i32>(x,y));
            plug_supply_row[t]=s_out;
            geometry[SET_IN_ROW*cfg.n+i].y=s_out;
            geometry[SET_IN_ROW*cfg.n+i].z=l_out;
        }
        workgroupBarrier();
    }
}
fn plug_row_linked(row:u32)->bool{
    return ((workgroupUniformLoad(&plug_linked[row>>5u])>>(row&31u))&1u)!=0u;
}
// Second half of the kinematic silo walk: one workgroup from the floor to
// one row past the ceiling, column-owned across rows so every `cells` write
// is only ever read back by the same thread, and row y-1 -> row y handoff
// goes exclusively through workgroup memory (`workgroupBarrier()` only).
@compute @workgroup_size(PLUG_THREADS) fn plug_walk(@builtin(local_invocation_id) id:vec3<u32>){
    let k=VOID_SPREAD/cfg.h;let t=id.x;
    if t<PLUG_WORDS{atomicStore(&plug_marks[t],0u);}
    workgroupBarrier();
    for(var i=t;i<cfg.n;i+=4u*PLUG_THREADS){
        var z=vec4<f32>(0.);
        for(var u=0u;u<4u;u++){let c=i+u*PLUG_THREADS;if c<cfg.n{z[u]=scratch[PLUG_ROW*cfg.n+c].z;}}
        for(var u=0u;u<4u;u++){
            let c=i+u*PLUG_THREADS;
            if c<cfg.n&&z[u]>0.{
                let row=c/cfg.nx;
                atomicOr(&plug_marks[row>>5u],1u<<(row&31u));
            }
        }
    }
    workgroupBarrier();
    if t<PLUG_WORDS{plug_linked[t]=atomicLoad(&plug_marks[t]);}
    workgroupBarrier();
    let in_col=t<cfg.nx;let x=i32(t);
    for(var y=0u;y<=cfg.ny;y++){
        let has_prev=y>0u&&plug_row_linked(y-1u);
        let has_curr=y<cfg.ny&&plug_row_linked(y);
        if !has_prev&&!has_curr{continue;}
        var l_asked=vec3<f32>(0.);
        var l_down=vec3<f32>(0.);
        var l_down_fine=vec3<f32>(0.);
        var l_up=vec3<f32>(0.);
        var l_up_fine=vec3<f32>(0.);
        var l_vf:array<vec2<f32>,3>;
        var l_vc:array<vec2<f32>,3>;
        var l_pw:array<vec2<f32>,3>;
        if has_prev&&in_col{
            if t>0u{
                let gx=t-1u;
                l_asked.x=pw_ask[gx];
                let dn=pw_down[gx];let up=pw_up[gx];
                l_down.x=dn.z;l_down_fine.x=dn.w;
                l_up.x=up.z;l_up_fine.x=up.w;
                if dn.z>0.||up.z>0.{
                    let v=pw_vel[gx];
                    l_vf[0]=v.xy;l_vc[0]=v.zw;l_pw[0]=unpack2x16unorm(pw_pow[gx]);
                }
            }
            {
                let gx=t;
                l_asked.y=pw_ask[gx];
                let dn=pw_down[gx];let up=pw_up[gx];
                l_down.y=dn.y;l_down_fine.y=dn.w;
                l_up.y=up.y;l_up_fine.y=up.w;
                if dn.y>0.||up.y>0.{
                    let v=pw_vel[gx];
                    l_vf[1]=v.xy;l_vc[1]=v.zw;l_pw[1]=unpack2x16unorm(pw_pow[gx]);
                }
            }
            if t+1u<cfg.nx{
                let gx=t+1u;
                l_asked.z=pw_ask[gx];
                let dn=pw_down[gx];let up=pw_up[gx];
                l_down.z=dn.x;l_down_fine.z=dn.w;
                l_up.z=up.x;l_up_fine.z=up.w;
                if dn.x>0.||up.x>0.{
                    let v=pw_vel[gx];
                    l_vf[2]=v.xy;l_vc[2]=v.zw;l_pw[2]=unpack2x16unorm(pw_pow[gx]);
                }
            }
        }
        workgroupBarrier();
        if in_col{
            let yi=i32(y);
            if has_prev{
                if y>1u&&(l_down.x>0.||l_down.y>0.||l_down.z>0.){
                    plug_receive_local(at(vec2<i32>(x,yi-2)),l_down,l_down_fine,l_vf,l_vc,l_pw);
                }
                if y<cfg.ny&&(l_up.x>0.||l_up.y>0.||l_up.z>0.){
                    plug_receive_local(at(vec2<i32>(x,yi)),l_up,l_up_fine,l_vf,l_vc,l_pw);
                }
            }
            if has_curr{
                let i=at(vec2<i32>(x,yi));let record=scratch[PLUG_ROW*cfg.n+i];
                var step=PlugStep();if record.z>0.{step=plug_give(i,x,yi,record,k,l_asked);}
                pw_ask[t]=step.ask;
                pw_down[t]=vec4<f32>(step.left,step.centre,step.right,step.fine);
                pw_up[t]=vec4<f32>(step.up.x,step.up.y,step.up.z,step.up_fine);
                if step.left>0.||step.centre>0.||step.right>0.||step.up.x>0.||step.up.y>0.||step.up.z>0.{
                    let after=cells[i];let ag=grains(after);
                    var vf=vec2<f32>(0.);var vc=vec2<f32>(0.);var ps=vec2<f32>(0.);
                    if ag.x>TRACE_VOLUME{
                        vf=after.momentum[1]/ag.x;
                        let share=white_powder(after);
                        ps=clamp(vec2<f32>(share.total,share.free),vec2<f32>(0.),vec2<f32>(1.));
                    }
                    if ag.y>TRACE_VOLUME{vc=after.momentum[2]/ag.y;}
                    pw_vel[t]=vec4<f32>(vf,vc);
                    pw_pow[t]=pack2x16unorm(ps);
                }
            }
        }
        workgroupBarrier();
    }
}
