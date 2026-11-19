// The bubbles' identities. Two pieces of air parted by water are two bubbles:
// where they meet side by side their soap film stands upright, carries no
// load and does not break (Karnakov et al. 2022), but left with one identity
// they would merge on contact. After each update's transport (stage Labels)
// the pieces that have parted take different identities, stacked bubbles
// join once their film has drained, and held bubbles are anchored.

const LABEL_SLOTS:u32=GAS_SLOTS;
const NO_LABEL:u32=0xffffffffu;
const LABEL_MARGIN:i32=2;
// The buffer in words: parent[GAS_SLOTS n], each (cell, slot) node's
// union-find parent; compact[GAS_SLOTS n], a root's area, then its record;
// the record counter; MAX_COMPONENTS records; then, kept between updates,
// each cell's pressed and held counts by identity, its bits of the anchored
// identities and their records. A record: root, slot, area, bounding box
// (min x, min y, max x, max y), new slot; by other identity its contact
// above, below and beside and the lowest root it touches (roots, unlike
// record numbers, do not follow the threads' order); whether it is held; its
// pressed and held counts (`age`, `settle`); its contact line's five sums and
// slope (f32 bits). All starts as NO_LABEL but the held counts: they start
// drained (labels.rs).
@group(0) @binding(16) var<storage,read_write> labels:array<atomic<u32>>;
// Areas and contact heights are summed as integers, in 1/LABEL_AREA_UNIT and
// 1/CONTACT_UNIT of a cell: such sums do not depend on the threads' order.
const LABEL_AREA_UNIT:f32=1024.;
const LABEL_RECORD:u32=17u+4u*GAS_SLOTS;
const CONTACT_UNIT:f32=64.;
// The time (s) the film between two pressed bubbles takes to drain and break:
// on a real picture a bubble sat that long under the frame's air layer.
const FILM_DRAINAGE:f32=2.;
fn label_drain_updates()->u32{return u32(round(FILM_DRAINAGE/cfg.dt));}
fn label_age(k:u32)->u32{return label_field(k,9u+4u*GAS_SLOTS);}
fn label_cell_age(i:u32,l:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*MAX_COMPONENTS+GAS_SLOTS*i+l;}
fn label_above(k:u32,m:u32)->u32{return label_field(k,8u+m);}
fn label_below(k:u32,m:u32)->u32{return label_field(k,8u+GAS_SLOTS+m);}
fn label_beside(k:u32,m:u32)->u32{return label_field(k,8u+2u*GAS_SLOTS+m);}
fn label_touching(k:u32,m:u32)->u32{return label_field(k,8u+3u*GAS_SLOTS+m);}
fn label_held(k:u32)->u32{return label_field(k,8u+4u*GAS_SLOTS);}
fn label_settle(k:u32)->u32{return label_field(k,10u+4u*GAS_SLOTS);}
fn label_cell_settle(i:u32,l:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*MAX_COMPONENTS+GAS_SLOTS*cfg.n+GAS_SLOTS*i+l;}
fn label_cell_anchor(i:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*MAX_COMPONENTS+2u*GAS_SLOTS*cfg.n+i;}
// Most cells hold no gas, and a kernel that asked each of a cell's slots
// would spend its time on them. So the cell's word of anchored identities
// (bits 0-8) also says which slots hold gas (CELL_HELD, from clear_trace_gas
// and label_apply), which slots' nodes stand in the union-find (CELL_NODES,
// from label_init) and whether its counts may be other than zero
// (CELL_COUNTED): a kernel reads the word and visits those slots alone. A
// word without CELL_KEPT (at the start, after a turn has moved the gas under
// it in invert_cells, or from a state saved before these bits) reads as all
// of them set, which visits every slot. The flow solve's tension tensor
// reads the slots that hold gas from the same word (corner_slots).
const CELL_SLOTS:u32=(1u<<GAS_SLOTS)-1u;
const CELL_HELD:u32=GAS_SLOTS;
const CELL_NODES:u32=2u*GAS_SLOTS;
const CELL_COUNTED:u32=1u<<(3u*GAS_SLOTS);
const CELL_KEPT:u32=1u<<31u;
fn label_cell_word(i:u32)->u32{
    let word=atomicLoad(&labels[label_cell_anchor(i)]);
    if (word&CELL_KEPT)==0u{return word|~CELL_SLOTS;}
    return word;
}
// Stores cell i's word if it differs from `before`, the word as read.
fn label_cell_keep(i:u32,before:u32,word:u32){
    if word!=before{atomicStore(&labels[label_cell_anchor(i)],word);}
}
// The lowest slot of a set of slots, which it leaves.
fn label_next_slot(slots:ptr<function,u32>)->u32{
    let l=countTrailingZeros(*slots);*slots&=*slots-1u;return l;
}
fn label_cell_record(i:u32,l:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*MAX_COMPONENTS+(2u*GAS_SLOTS+1u)*cfg.n+GAS_SLOTS*i+l;}
fn label_contact(k:u32,m:u32)->u32{return label_field(k,11u+4u*GAS_SLOTS+m);}
fn label_contact_slope(k:u32)->u32{return label_field(k,16u+4u*GAS_SLOTS);}
fn label_node(i:u32,l:u32)->u32{return i*GAS_SLOTS+l;}
fn label_compact(node:u32)->u32{return GAS_SLOTS*cfg.n+node;}
fn label_counter()->u32{return 2u*GAS_SLOTS*cfg.n;}
fn label_field(k:u32,field:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*k+field;}
// A cell holding a grain layer or more of an identity is a member of its
// component. Union-find with atomic minima joins the members: a root is its
// component's lowest node, whatever the threads' order.
fn label_member(i:u32,l:u32)->bool{return geometry[layer_index(i,l)].x*cfg.h>=GRAIN_LAYER;}
fn label_find(start:u32)->u32{
    var x=start;
    loop{let p=atomicLoad(&labels[x]);if p==x{return x;}x=p;}
    return x;
}
fn label_union(first:u32,second:u32){
    var a=first;var b=second;
    loop{
        a=label_find(a);b=label_find(b);
        if a==b{return;}
        if a<b{let t=a;a=b;b=t;}
        let old=atomicMin(&labels[a],b);
        if old==a{return;}
        a=old;
    }
}
@compute @workgroup_size(128) fn label_init(@builtin(global_invocation_id) id:vec3<u32>){
    let t=id.x;
    if t==0u{atomicStore(&labels[label_counter()],0u);}
    if t<LABEL_RECORD*MAX_COMPONENTS{
        let field=t%LABEL_RECORD;
        let touching=field>=8u+3u*GAS_SLOTS&&field<8u+4u*GAS_SLOTS;
        atomicStore(&labels[label_field(t/LABEL_RECORD,field)],select(0u,NO_LABEL,field==3u||field==4u||touching));
    }
    let i=t;if i>=cfg.n{return;}
    // A member's node starts as its own root with no area; a node that
    // stood and is no member leaves the union-find. The other slots' nodes
    // are out of it already, and their areas are read by no one.
    let before=label_cell_word(i);let stood=(before>>CELL_NODES)&CELL_SLOTS;
    var slots=((before>>CELL_HELD)&CELL_SLOTS)|stood;var members=0u;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);let node=label_node(i,l);
        if ((before>>(CELL_HELD+l))&1u)!=0u&&label_member(i,l){
            members|=1u<<l;atomicStore(&labels[node],node);atomicStore(&labels[label_compact(node)],0u);
        }else if ((stood>>l)&1u)!=0u{atomicStore(&labels[node],NO_LABEL);}
    }
    label_cell_keep(i,before,(before&~(CELL_SLOTS<<CELL_NODES))|(members<<CELL_NODES));
}
@compute @workgroup_size(128) fn label_merge(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=point(i);
    var slots=(label_cell_word(i)>>CELL_NODES)&CELL_SLOTS;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);
        for(var a=0u;a<2u;a++){
            let q=p+axis(a);if !valid(q){continue;}
            let other=label_node(at(q),l);
            if atomicLoad(&labels[other])!=NO_LABEL{label_union(label_node(i,l),other);}
        }
    }
}
@compute @workgroup_size(128) fn label_roots(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    var slots=(label_cell_word(i)>>CELL_NODES)&CELL_SLOTS;
    loop{
        if slots==0u{break;}
        let node=label_node(i,label_next_slot(&slots));
        atomicStore(&labels[node],label_find(node));
    }
}
// Each component's area, gathered at its root.
@compute @workgroup_size(128) fn label_sizes(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    var slots=(label_cell_word(i)>>CELL_HELD)&CELL_SLOTS;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);
        var root=atomicLoad(&labels[label_node(i,l)]);
        let amount=geometry[layer_index(i,l)].x;if amount<=0.{continue;}
        if root==NO_LABEL{root=label_follow(i,l);if root==NO_LABEL{continue;}}
        atomicAdd(&labels[label_compact(root)],u32(round(amount*LABEL_AREA_UNIT)));
    }
}
// A record for every component at least as large as a round bubble of
// diameter l_c (2.8 mm2), the smallest that holds a joint open. A smaller one
// is a shred, which the transport would smear over some ten cells: it keeps
// its identity and rejoins its bubble on contact. Records for shreds would
// run out, and the threads' order would decide which bubbles go without.
@compute @workgroup_size(128) fn label_records(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    let bubble=u32(round(.25*3.14159265*CAPILLARY_LENGTH*CAPILLARY_LENGTH/(cfg.h*cfg.h)*LABEL_AREA_UNIT));
    var slots=(label_cell_word(i)>>CELL_NODES)&CELL_SLOTS;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);
        let node=label_node(i,l);if atomicLoad(&labels[node])!=node{continue;}
        let compact=label_compact(node);
        if atomicLoad(&labels[compact])<bubble{atomicStore(&labels[compact],NO_LABEL);continue;}
        let k=atomicAdd(&labels[label_counter()],1u);
        if k>=MAX_COMPONENTS{atomicStore(&labels[compact],NO_LABEL);continue;}
        atomicStore(&labels[compact],k);
        atomicStore(&labels[label_field(k,0u)],node);atomicStore(&labels[label_field(k,1u)],l);
    }
}
// Whether what lies at `up` and above holds a bubble pressed against it: the
// frame's top, or sand in contact that the plates hold against the push. A
// packed layer of height H pushed up over the gap b holds by both plates'
// friction, 2 mu_s K H against b (Janssen's arch): it must be b / (2 mu_s K)
// = 2.75 mm tall or reach the frame's top, else it would ride its bubble.
fn label_supported(up:vec2<i32>)->bool{
    let rows=i32(ceil(PLATE_GAP/(2.*STATIC_FRICTION*LATERAL_STRESS_RATIO*cfg.h)));
    for(var k=0;k<rows;k++){
        let q=up+vec2<i32>(0,k);
        if !valid(q){return true;}
        if !pores_only(cells[at(q)]){return false;}
    }
    return true;
}
// The root of a component of slot l in a cell beside cell i, or NO_LABEL: gas
// below a grain layer follows it, in the relabelling and in the bubble's
// area, which would else flicker as the rim cells cross the grain layer.
fn label_follow(i:u32,l:u32)->u32{
    let p=point(i);
    for(var a=0u;a<4u;a++){
        let q=p+select(-axis(a%2u),axis(a%2u),a>=2u);if !valid(q){continue;}
        let parent=atomicLoad(&labels[label_node(at(q),l)]);if parent!=NO_LABEL{return parent;}
    }
    return NO_LABEL;
}
// Fills the records: area, box, contacts, hold and the cells' counts.
@compute @workgroup_size(128) fn label_measure(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=vec2<u32>(point(i));
    var slots=(label_cell_word(i)>>CELL_HELD)&CELL_SLOTS;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);
        var parent=atomicLoad(&labels[label_node(i,l)]);
        if parent==NO_LABEL{
            let amount=geometry[layer_index(i,l)].x;if amount<=0.{continue;}
            let root=label_follow(i,l);if root==NO_LABEL{continue;}
            let k=atomicLoad(&labels[label_compact(root)]);if k>=MAX_COMPONENTS{continue;}
            atomicAdd(&labels[label_field(k,2u)],u32(round(amount*LABEL_AREA_UNIT)));
            continue;
        }
        let k=atomicLoad(&labels[label_compact(parent)]);if k>=MAX_COMPONENTS{continue;}
        atomicAdd(&labels[label_field(k,2u)],u32(round(geometry[layer_index(i,l)].x*LABEL_AREA_UNIT)));
        atomicMin(&labels[label_field(k,3u)],p.x);atomicMin(&labels[label_field(k,4u)],p.y);
        atomicMax(&labels[label_field(k,5u)],p.x);atomicMax(&labels[label_field(k,6u)],p.y);
        // Where another bubble meets this one around the cell, their film
        // runs across the line joining their gases' centres in the 3x3: the
        // line's parts up, down and across, weighted by the smaller gas.
        for(var m=0u;m<LABEL_SLOTS;m++){
            if m==l{continue;}
            var touching=NO_LABEL;
            var own=vec3<f32>(0.);var other=vec3<f32>(0.);
            for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
                let q=vec2<i32>(p)+vec2<i32>(x,y);if !valid(q){continue;}
                let j=at(q);let offset=vec2<f32>(f32(x),f32(y));
                let a=geometry[layer_index(j,l)].x;let b=geometry[layer_index(j,m)].x;
                own+=a*vec3<f32>(offset,1.);other+=b*vec3<f32>(offset,1.);
                let node=atomicLoad(&labels[label_node(j,m)]);
                if node<touching&&atomicLoad(&labels[label_compact(node)])<MAX_COMPONENTS{touching=node;}
            }}
            if touching==NO_LABEL||own.z<=0.||other.z<=0.{continue;}
            let d=other.xy/other.z-own.xy/own.z;let weight=min(own.z,other.z)*LABEL_AREA_UNIT;
            atomicAdd(&labels[select(label_below(k,m),label_above(k,m),d.y>0.)],u32(round(abs(d.y)*weight)));
            atomicAdd(&labels[label_beside(k,m)],u32(round(abs(d.x)*weight)));
            atomicMin(&labels[label_touching(k,m)],touching);
        }
        let up=vec2<i32>(p)+vec2<i32>(0,1);
        if label_supported(up){
            atomicMax(&labels[label_held(k)],1u);
            // The contact's height in this column: the top of the gas, which
            // fills the cell from below where the bubble goes on below it,
            // else lies against what holds it.
            let down=vec2<i32>(p)-vec2<i32>(0,1);
            var top=f32(p.y)+1.;
            if valid(down)&&atomicLoad(&labels[label_node(at(down),l)])!=NO_LABEL{top=f32(p.y)+min(geometry[layer_index(i,l)].x,1.);}
            let height=u32(round(top*CONTACT_UNIT));
            atomicAdd(&labels[label_contact(k,0u)],1u);atomicAdd(&labels[label_contact(k,1u)],p.x);
            atomicAdd(&labels[label_contact(k,2u)],height);atomicAdd(&labels[label_contact(k,3u)],p.x*p.x);
            atomicAdd(&labels[label_contact(k,4u)],p.x*height);
        }
        let age=atomicLoad(&labels[label_cell_age(i,l)]);
        if age!=NO_LABEL{atomicMax(&labels[label_age(k)],age);}
        let settle=atomicLoad(&labels[label_cell_settle(i,l)]);
        if settle!=NO_LABEL{atomicMax(&labels[label_settle(k)],settle);}
    }
}
var<workgroup> color_area:array<u32,MAX_COMPONENTS>;
var<workgroup> color_root:array<u32,MAX_COMPONENTS>;
var<workgroup> color_slot:array<u32,MAX_COMPONENTS>;
var<workgroup> color_box:array<vec4<i32>,MAX_COMPONENTS>;
// rank[record], order[rank] = record; kept = 1 for the first of its identity;
// choice = its identity once decided; forbidden = the identities of the
// keepers near it; host = the bubble it joins, if any.
var<workgroup> color_rank:array<u32,MAX_COMPONENTS>;
var<workgroup> color_order:array<u32,MAX_COMPONENTS>;
var<workgroup> color_kept:array<u32,MAX_COMPONENTS>;
var<workgroup> color_choice:array<u32,MAX_COMPONENTS>;
var<workgroup> color_decided:array<u32,MAX_COMPONENTS>;
var<workgroup> color_forbidden:array<u32,MAX_COMPONENTS>;
var<workgroup> color_host:array<u32,MAX_COMPONENTS>;
var<workgroup> color_held:array<u32,MAX_COMPONENTS>;
var<workgroup> color_pending:atomic<u32>;
// Near: bounding boxes within LABEL_MARGIN cells of each other.
fn label_near(a:u32,b:u32)->bool{
    let ba=color_box[a];let bb=color_box[b];
    return ba.x-LABEL_MARGIN<=bb.z&&bb.x<=ba.z+LABEL_MARGIN&&ba.y-LABEL_MARGIN<=bb.w&&bb.y<=ba.w+LABEL_MARGIN;
}
fn label_before(a:u32,b:u32)->bool{
    if color_area[a]!=color_area[b]{return color_area[a]>color_area[b];}
    return color_root[a]<color_root[b];
}
// Chooses the identities, one thread per record. In the order of assignment
// (larger first, then the lower root) the first component of each identity
// keeps it; any other keeps its own unless a component near it that has one
// carries it, and then takes the lowest none of them carries, as soon as
// every earlier component near it has decided. The pass also joins stacked
// bubbles, counts the pressed and held updates and fits the contact lines.
@compute @workgroup_size(MAX_COMPONENTS) fn label_color(@builtin(local_invocation_index) t:u32){
    let count=min(atomicLoad(&labels[label_counter()]),MAX_COMPONENTS);
    if t==0u{atomicStore(&color_pending,0u);}
    if t<count{
        color_area[t]=atomicLoad(&labels[label_field(t,2u)]);
        color_held[t]=atomicLoad(&labels[label_held(t)]);
        color_root[t]=atomicLoad(&labels[label_field(t,0u)]);
        color_slot[t]=atomicLoad(&labels[label_field(t,1u)]);
        color_box[t]=vec4<i32>(i32(atomicLoad(&labels[label_field(t,3u)])),i32(atomicLoad(&labels[label_field(t,4u)])),
            i32(atomicLoad(&labels[label_field(t,5u)])),i32(atomicLoad(&labels[label_field(t,6u)])));
    }
    workgroupBarrier();
    if t<count{
        var rank=0u;
        for(var j=0u;j<count;j++){if label_before(j,t){rank++;}}
        color_rank[t]=rank;color_order[rank]=t;
    }
    workgroupBarrier();
    // A bubble pressed against a larger one above or below it (the line
    // joining their gases' centres runs more up than across) joins it once
    // they have been so pressed, the upper one held from above, for the
    // film's drainage time: the film lies flat, carries the lower bubble's
    // buoyancy, drains and breaks. Bubbles rising free or sloshed by the
    // shake part before that. Never joined, a bubble pressed up into those
    // above leaves its gas in their cells and the raft frays to a foam.
    if t<count{
        var host=NO_LABEL;var most=0u;
        for(var m=0u;m<LABEL_SLOTS;m++){
            let root=atomicLoad(&labels[label_touching(t,m)]);if root==NO_LABEL{continue;}
            let h=atomicLoad(&labels[label_compact(root)]);
            if h>=count||color_rank[h]>=color_rank[t]{continue;}
            let above=atomicLoad(&labels[label_above(t,m)]);let below=atomicLoad(&labels[label_below(t,m)]);
            let beside=atomicLoad(&labels[label_beside(t,m)]);
            // The other above and held, or below and this one held.
            var stacked=0u;
            if above>=below&&color_held[h]!=0u{stacked=above;}
            if below>above&&color_held[t]!=0u{stacked=below;}
            if stacked<=beside||stacked<=most{continue;}
            host=h;most=stacked;
        }
        // One more update of drainage; once drained it joins and counts anew.
        var age=0u;if host!=NO_LABEL{age=atomicLoad(&labels[label_age(t)])+1u;}
        if age<label_drain_updates(){host=NO_LABEL;}else{age=0u;}
        atomicStore(&labels[label_age(t)],age);
        color_host[t]=host;
        var settle=0u;if color_held[t]!=0u{settle=min(atomicLoad(&labels[label_settle(t)])+1u,label_drain_updates());}
        atomicStore(&labels[label_settle(t)],settle);
        // The slope of the line through its contact (least squares).
        let n=f32(atomicLoad(&labels[label_contact(t,0u)]));var slope=0.;
        if n>=2.{
            let sx=f32(atomicLoad(&labels[label_contact(t,1u)]));let sy=f32(atomicLoad(&labels[label_contact(t,2u)]))/CONTACT_UNIT;
            let sxx=f32(atomicLoad(&labels[label_contact(t,3u)]));let sxy=f32(atomicLoad(&labels[label_contact(t,4u)]))/CONTACT_UNIT;
            let spread=n*sxx-sx*sx;if spread>0.{slope=(n*sxy-sx*sy)/spread;}
        }
        atomicStore(&labels[label_contact_slope(t)],bitcast<u32>(slope));
    }
    workgroupBarrier();
    if t<count{
        let merged=color_host[t]!=NO_LABEL;
        var keeper=!merged;
        for(var r=0u;r<color_rank[t];r++){
            let earlier=color_order[r];
            if color_host[earlier]==NO_LABEL&&color_slot[earlier]==color_slot[t]{keeper=false;break;}
        }
        color_kept[t]=select(0u,1u,keeper);color_decided[t]=select(0u,1u,keeper);color_choice[t]=color_slot[t];
        if !keeper{atomicAdd(&color_pending,1u);}
    }
    workgroupBarrier();
    if t<count&&color_kept[t]==0u{
        var forbidden=0u;
        for(var j=0u;j<count;j++){
            if color_kept[j]!=0u&&label_near(t,j){forbidden|=1u<<color_choice[j];}
        }
        color_forbidden[t]=forbidden;
    }
    loop{
        workgroupBarrier();
        if workgroupUniformLoad(&color_pending)==0u{break;}
        var ready=false;var choice=0u;
        if t<count&&color_decided[t]==0u&&color_host[t]!=NO_LABEL{
            let host=color_host[t];
            if color_decided[host]!=0u{ready=true;choice=color_choice[host];}
        }
        else if t<count&&color_decided[t]==0u{
            ready=true;var forbidden=color_forbidden[t];
            for(var j=0u;j<count;j++){
                if color_kept[j]!=0u||color_rank[j]>=color_rank[t]||!label_near(t,j){continue;}
                if color_decided[j]==0u{ready=false;break;}
                forbidden|=1u<<color_choice[j];
            }
            if ready{
                let slot=color_slot[t];choice=slot;
                if (forbidden&(1u<<slot))!=0u{
                    for(var l=0u;l<LABEL_SLOTS;l++){if (forbidden&(1u<<l))==0u{choice=l;break;}}
                }
            }
        }
        workgroupBarrier();
        if ready{color_choice[t]=choice;color_decided[t]=1u;atomicSub(&color_pending,1u);}
    }
    if t<count{atomicStore(&labels[label_field(t,7u)],color_choice[t]);}
}
fn label_target(i:u32,l:u32)->u32{
    let parent=atomicLoad(&labels[label_node(i,l)]);
    if parent==NO_LABEL{return NO_LABEL;}
    let k=atomicLoad(&labels[label_compact(parent)]);
    if k>=MAX_COMPONENTS{return l;}
    return atomicLoad(&labels[label_field(k,7u)]);
}
// Moves the gas to its new identities; keeps the cells' counts and anchors.
@compute @workgroup_size(128) fn label_apply(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=point(i);
    var amounts:array<f32,LABEL_SLOTS>;var ages:array<u32,LABEL_SLOTS>;var settles:array<u32,LABEL_SLOTS>;var moved=false;
    var anchored=0u;
    for(var l=0u;l<LABEL_SLOTS;l++){amounts[l]=0.;ages[l]=0u;settles[l]=0u;}
    let before=label_cell_word(i);let held=(before>>CELL_HELD)&CELL_SLOTS;var slots=held;
    loop{
        if slots==0u{break;}
        let l=label_next_slot(&slots);
        let amount=geometry[layer_index(i,l)].x;if amount<=0.{continue;}
        var destination=label_target(i,l);
        let parent=atomicLoad(&labels[label_node(i,l)]);
        if parent!=NO_LABEL&&destination!=NO_LABEL{
            let k=atomicLoad(&labels[label_compact(parent)]);
            if k<MAX_COMPONENTS{
                ages[destination]=max(ages[destination],atomicLoad(&labels[label_age(k)]));
                settles[destination]=max(settles[destination],atomicLoad(&labels[label_settle(k)]));
                if label_anchored(k)&&!label_exposed(i,l){anchored|=1u<<destination;atomicStore(&labels[label_cell_record(i,destination)],k);}
            }
        }
        if destination==NO_LABEL{
            // Gas below a grain layer follows a neighbouring component, held
            // with it: it is no end of the bubble that buoyancy could lift.
            destination=l;
            let root=label_follow(i,l);
            if root!=NO_LABEL{
                let k=atomicLoad(&labels[label_compact(root)]);
                if k<MAX_COMPONENTS{
                    destination=atomicLoad(&labels[label_field(k,7u)]);
                    if label_anchored(k){anchored|=1u<<destination;atomicStore(&labels[label_cell_record(i,destination)],k);}
                }
            }
        }
        if destination!=l{moved=true;}
        amounts[destination]+=amount;
    }
    // Counts that are zero and were stored as zero stand.
    var counted=false;
    for(var l=0u;l<LABEL_SLOTS;l++){if ages[l]!=0u||settles[l]!=0u{counted=true;}}
    if counted||(before&CELL_COUNTED)!=0u{
        for(var l=0u;l<LABEL_SLOTS;l++){
            atomicStore(&labels[label_cell_age(i,l)],ages[l]);
            atomicStore(&labels[label_cell_settle(i,l)],settles[l]);
        }
    }
    var holds=held;
    if moved{
        holds=0u;
        for(var l=0u;l<LABEL_SLOTS;l++){
            let entry=geometry[layer_index(i,l)];
            geometry[layer_index(i,l)]=vec4<f32>(amounts[l],f32(l+1u),entry.z,entry.w);
            if amounts[l]>0.{holds|=1u<<l;}
        }
    }
    label_cell_keep(i,before,anchored|(holds<<CELL_HELD)|(before&(CELL_SLOTS<<CELL_NODES))|select(0u,CELL_COUNTED,counted)|CELL_KEPT);
}
// The area (cells) of the bubble of slot l that cell i is a member of, or 0.
// The slab reads it at the start of the next update, before anything moves.
fn label_area(i:u32,l:u32)->f32{
    let parent=atomicLoad(&labels[label_node(i,l)]);if parent==NO_LABEL{return 0.;}
    let k=atomicLoad(&labels[label_compact(parent)]);if k>=MAX_COMPONENTS{return 0.;}
    return f32(atomicLoad(&labels[label_field(k,2u)]))/LABEL_AREA_UNIT;
}

// Open water: no deposit and less than a grain layer of gas and of sand.
fn label_open(c:Cell)->bool{
    if max(c.bed.x,0.)+max(c.bed.y,0.)>0.||c.volume.gas*cfg.h>=GRAIN_LAYER{return false;}
    let g=grains(c);return (g.x+g.y)*cfg.h<GRAIN_LAYER;
}
// Whether cell i's gas of slot l has come out from under what holds its
// bubble: above the bubble's gas in this column lies open water, at least a
// capillary length wide. A held bubble creeps as one body only where
// something presses on it; under open water its end rises by its buoyancy,
// rho g times the puddle's thickness of about 2 l_c (3.4 mm), some 33 Pa
// against the 0.1 Pa of the streams' inflow. Gas enters a gap of width w
// against the meniscus pressure 2 sigma / w, which that head pays only for
// w >= l_c: the joints' throats, a cell or so wide, keep their bubbles held.
fn label_exposed(i:u32,l:u32)->bool{
    var q=point(i)+vec2<i32>(0,1);
    loop{
        if !valid(q){return false;}
        if atomicLoad(&labels[label_node(at(q),l)])==NO_LABEL{break;}
        q.y+=1;
    }
    if !label_open(cells[at(q)]){return false;}
    var width=1.;
    for(var s=-1;s<=1;s+=2){
        var r=q+vec2<i32>(s,0);
        loop{
            if width*cfg.h>=CAPILLARY_LENGTH||!valid(r)||!label_open(cells[at(r)]){break;}
            width+=1.;r.x+=s;
        }
    }
    return width*cfg.h>=CAPILLARY_LENGTH;
}
// Whether the bubble of record k is anchored: held from above for the film's
// drainage time, by when its arrival has relaxed.
fn label_anchored(k:u32)->bool{
    return atomicLoad(&labels[label_held(k)])!=0u&&atomicLoad(&labels[label_settle(k)])>=label_drain_updates();
}
fn label_anchored_at(i:u32,l:u32)->bool{return ((atomicLoad(&labels[label_cell_anchor(i)])>>l)&1u)!=0u;}
// The velocity of cell i's gas of slot l, an anchored bubble's, where the
// flow would carry it at `carried`: along its contact only, up the contact's
// slope against the apparent gravity, no faster than the flow or than its
// soap films let it creep. A surfactant-laden surface does not slip
// (Marangoni): moving at U the bubble shears the films it lays on both
// plates, of thickness 4^(2/3) 1.337 (b/2) Ca^(2/3), Ca = eta U / sigma
// (Ratulowski and Chang 1990), whose stress grows only as U^(1/3). Against
// the buoyancy F along the contact it creeps at (0.842 F b^2)^3 / (eta
// sigma^2) = 0.48 sin^3(slope) m/s: 2.5 mm/s at 10 degrees, and nanometres a
// second under the fraction of a pascal the streams' inflow sets across it.
fn label_anchored_velocity(i:u32,l:u32,carried:vec2<f32>)->vec2<f32>{
    let k=atomicLoad(&labels[label_cell_record(i,l)]);if k>=MAX_COMPONENTS{return vec2<f32>(0.);}
    let slope=bitcast<f32>(atomicLoad(&labels[label_contact_slope(k)]));
    var tangent=vec2<f32>(1.,slope)/sqrt(1.+slope*slope);
    var push=RHO.x*(cfg.gravity*tangent.y-cfg.lateral*tangent.x);
    if push<0.{tangent=-tangent;push=-push;}
    let creep=pow(.84225*push*PLATE_GAP*PLATE_GAP,3.)/(WATER_VISCOSITY*SURFACE_TENSION*SURFACE_TENSION);
    return tangent*clamp(dot(carried,tangent),0.,creep);
}
