// The plug descent: the voids that a joint's take leaves in a packed body
// rise through it and the body comes down as a plug, the kinematic model of
// silo flow as a transport step (plug_descent). plug_record runs before the
// carry (stage TransportPrepare), plug_descent after it (stage TraceGas).

// The walk's threads, and the cells whose steps it keeps: more than a row
// and two cells of any grid.
const PLUG_THREADS:u32=256u;
const PLUG_KEPT:u32=384u;
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
// geometry[PLUG_ROW n + i] = (fine, coarse, link, wet) before the carry:
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
// The most rows of a grid. plug_linked: the rows that hold a link. The
// other arrays: the steps of the last PLUG_KEPT cells of the walk, a cell's
// at its raster index modulo PLUG_KEPT.
const PLUG_ROWS:u32=256u;
var<workgroup> plug_marks:array<atomic<u32>,PLUG_ROWS>;
var<workgroup> plug_linked:array<u32,PLUG_ROWS>;
var<workgroup> plug_ask:array<f32,PLUG_KEPT>;
var<workgroup> plug_left:array<f32,PLUG_KEPT>;
var<workgroup> plug_centre:array<f32,PLUG_KEPT>;
var<workgroup> plug_right:array<f32,PLUG_KEPT>;
var<workgroup> plug_fine:array<f32,PLUG_KEPT>;
var<workgroup> plug_up_left:array<f32,PLUG_KEPT>;
var<workgroup> plug_up_centre:array<f32,PLUG_KEPT>;
var<workgroup> plug_up_right:array<f32,PLUG_KEPT>;
var<workgroup> plug_up_fine:array<f32,PLUG_KEPT>;
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
fn plug_give(i:u32,x:i32,y:i32,record:vec4<f32>,k:f32,asked:vec3<f32>)->PlugStep{
    let nx=i32(cfg.nx);let c=cells[i];var out=PlugStep();
    if plug_deposit(c){return out;}
    let g=grains(c);let solid=g.x+g.y;let water=max(c.volume.water,0.);var given=0.;
    if y>0&&solid>0.{
        var want_left=0.;var want_right=0.;
        if x>0{want_left=k*asked.x;}
        if x+1<nx{want_right=k*asked.z;}
        var want_centre=plug_centre_weight(x,k)*asked.y;
        // Two links side by side each owe the cell under the other its
        // diagonal share: only the difference crosses between their columns.
        // The voids' sideways spread is a velocity (u = -K dv/dx, zero
        // where the columns descend alike), not an exchange of grains.
        if x>0&&scratch[PLUG_ROW*cfg.n+at(vec2<i32>(x-1,y))].z>0.{
            let crossed=min(want_left,k*asked.y);want_left-=crossed;want_centre+=crossed;
        }
        if x+1<nx&&scratch[PLUG_ROW*cfg.n+at(vec2<i32>(x+1,y))].z>0.{
            let crossed=min(want_right,k*asked.y);want_right-=crossed;want_centre+=crossed;
        }
        let want=want_left+want_centre+want_right;
        given=min(want,solid);
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
        if linked{ask=max(given+restore-solid,0.);}
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
                    if shares[d]>0.{surplus=min(surplus,plug_room(cells[at(vec2<i32>(x+d-1,y+1))])/(3.*shares[d]));}
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
// gy pass it, with their fine shares: the grains arrive with their givers'
// velocities and powder, and the same volume of its water leaves.
fn plug_receive(r:u32,x:i32,gy:i32,parts:vec3<f32>,fines:vec3<f32>){
    var received=0.;var fine=0.;var fine_momentum=vec2<f32>(0.);var coarse_momentum=vec2<f32>(0.);
    var powder=vec2<f32>(0.);
    for(var d=0;d<3;d++){
        let part=parts[d];if part<=0.{continue;}
        let giver=cells[at(vec2<i32>(x+d-1,gy))];let gg=grains(giver);
        let part_fine=part*fines[d];let part_coarse=part-part_fine;
        received+=part;fine+=part_fine;
        if gg.x>0.{
            fine_momentum+=part_fine*giver.momentum[1]/gg.x;
            let share=white_powder(giver);powder+=part_fine*vec2<f32>(share.total,share.free);
        }
        if gg.y>0.{coarse_momentum+=part_coarse*giver.momentum[2]/gg.y;}
    }
    if received<=0.{return;}
    let c=cells[r];let g=grains(c);let water=max(c.volume.water,0.);
    cells[r].volume.fine=g.x+fine;cells[r].volume.coarse=g.y+(received-fine);
    cells[r].volume.water=max(water-received,0.);
    cells[r].momentum[0]=plug_scaled(c.momentum[0],water,max(water-received,0.));
    cells[r].momentum[1]=c.momentum[1]+fine_momentum;cells[r].momentum[2]=c.momentum[2]+coarse_momentum;
    cells[r].powder.mobile=c.powder.mobile+powder.x;cells[r].powder.free=c.powder.free+powder.y;
}
// The walk, one workgroup from the floor to the ceiling. In the kinematic
// model of a thin silo (Nedderman and Tuzun 1979; K = 0.26 b, Hung,
// Aussillous and Capart 2018) the voids left by departing grains rise
// through the packing, spreading sideways by K per unit height, and the
// grains above fall into them: the column over a joint comes down as a
// plug of unchanged packing and the surface sinks into a funnel. The
// carry's cell-mean velocities leave the deficits where they arise, to
// loosen the column until a bubble enters it. So each link asks the three
// cells above for its deficit, weighted k, 1-2k, k with k = K/h, and gives
// what it is asked up to its grains; with no link above, it keeps its
// deficit and the surface sinks. A bubble cuts a chain, the deposit ends
// it, and sand that stands on nothing packed moves by the carry alone.
//
// The cells are walked in raster order, a thread to a cell and `width`
// cells a turn: a row a turn on a grid no wider than the workgroup, else
// PLUG_THREADS cells, so that no thread serves two cells in a turn (a turn
// lasts as long as its slowest thread). A cell's turn needs the steps of
// the three cells below it and leaves the cells beside it alone: on a wider
// grid those three lie a row less a cell back or more, in an earlier turn.
// In its turn the cell at (x, y) passes on what row y-1 gave (down to the
// cell at (x, y-2), up to itself), then gives in its own right; one row
// past the top only the first remains. Each cell so takes what the row
// below hands up, gives, and takes what the row above gives, in that order.
@compute @workgroup_size(PLUG_THREADS) fn plug_descent(@builtin(local_invocation_id) id:vec3<u32>){
    let nx=i32(cfg.nx);let ny=i32(cfg.ny);let k=VOID_SPREAD/cfg.h;let t=id.x;
    for(var y=t;y<cfg.ny;y+=PLUG_THREADS){atomicStore(&plug_marks[y],0u);}
    workgroupBarrier();
    for(var i=t;i<cfg.n;i+=4u*PLUG_THREADS){
        var z=vec4<f32>(0.);
        for(var u=0u;u<4u;u++){let c=i+u*PLUG_THREADS;if c<cfg.n{z[u]=scratch[PLUG_ROW*cfg.n+c].z;}}
        for(var u=0u;u<4u;u++){let c=i+u*PLUG_THREADS;if c<cfg.n&&z[u]>0.{atomicOr(&plug_marks[c/cfg.nx],1u);}}
    }
    workgroupBarrier();
    for(var y=t;y<cfg.ny;y+=PLUG_THREADS){plug_linked[y]=atomicLoad(&plug_marks[y]);}
    workgroupBarrier();
    let width=min(cfg.nx,PLUG_THREADS);let cells_walked=cfg.n+cfg.nx;
    for(var first=0u;first<cells_walked;first+=width){
        // A turn whose rows and the row below them hold no link has nothing
        // to pass on and nothing to give.
        let low=first/cfg.nx;let high=min((first+width-1u)/cfg.nx,cfg.ny-1u);
        var busy=false;
        if low>0u&&low<=cfg.ny&&workgroupUniformLoad(&plug_linked[low-1u])!=0u{busy=true;}
        if low<cfg.ny&&workgroupUniformLoad(&plug_linked[low])!=0u{busy=true;}
        if workgroupUniformLoad(&plug_linked[high])!=0u{busy=true;}
        if !busy{continue;}
        let cell=first+t;let walked=t<width&&cell<cells_walked;
        let x=i32(cell%cfg.nx);let y=i32(cell/cfg.nx);
        // The steps of the three cells below, if their row holds a link.
        var asked=vec3<f32>(0.);var down=vec3<f32>(0.);var down_fine=vec3<f32>(0.);
        var up=vec3<f32>(0.);var up_fine=vec3<f32>(0.);
        let handed=walked&&y>0&&plug_linked[y-1]!=0u;
        if handed{
            let below=cell-cfg.nx;
            if x>0{
                let r=(below-1u)%PLUG_KEPT;
                asked.x=plug_ask[r];down.x=plug_right[r];down_fine.x=plug_fine[r];up.x=plug_up_right[r];up_fine.x=plug_up_fine[r];
            }
            let r=below%PLUG_KEPT;
            asked.y=plug_ask[r];down.y=plug_centre[r];down_fine.y=plug_fine[r];up.y=plug_up_centre[r];up_fine.y=plug_up_fine[r];
            if x+1<nx{
                let q=(below+1u)%PLUG_KEPT;
                asked.z=plug_ask[q];down.z=plug_left[q];down_fine.z=plug_fine[q];up.z=plug_up_left[q];up_fine.z=plug_up_fine[q];
            }
        }
        workgroupBarrier();
        if handed{
            if y>1{plug_receive(at(vec2<i32>(x,y-2)),x,y-1,down,down_fine);}
            if y<ny{plug_receive(at(vec2<i32>(x,y)),x,y-1,up,up_fine);}
        }
        if walked&&y<ny&&plug_linked[y]!=0u{
            let i=at(vec2<i32>(x,y));let record=scratch[PLUG_ROW*cfg.n+i];
            var step=PlugStep();if record.z>0.{step=plug_give(i,x,y,record,k,asked);}
            let r=cell%PLUG_KEPT;
            plug_ask[r]=step.ask;
            plug_left[r]=step.left;plug_centre[r]=step.centre;plug_right[r]=step.right;plug_fine[r]=step.fine;
            plug_up_left[r]=step.up.x;plug_up_centre[r]=step.up.y;plug_up_right[r]=step.up.z;plug_up_fine[r]=step.up_fine;
        }
        workgroupBarrier();
    }
}
