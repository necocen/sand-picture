// Transport groups (stage TransportPrepare): a deposit cell whose free
// volume is small against the flow through it joins the larger free region
// beside it, and the group carries as one control volume whose inner faces
// carry nothing. A partition of the cells only: grouping moves no material.
// Then the groups' payloads, which each subdivision gathers anew.

// members[i] heads the list of owner i, members[cfg.n + i] follows i in
// its list; cfg.n ends a list.
@group(0) @binding(12) var<storage,read_write> members:array<atomic<u32>>;
fn free_inventory(i:u32)->f32{
    let v=mobile_volumes(cells[i]);return v.x+v.y+v.z+v.w;
}
// scratch[GROUP_MAP_ROW n + i] = (i's owner, the owner's other members,
// i's free inventory, the group's on its owner).
fn group_owner(i:u32)->u32{return u32(scratch[GROUP_MAP_ROW*cfg.n+i].x);}
fn group_count(i:u32)->u32{return 1u+u32(scratch[GROUP_MAP_ROW*cfg.n+i].y);}
fn group_member(i:u32,k:u32)->u32{
    if k==0u{return i;}var j=atomicLoad(&members[i]);
    for(var n=1u;n<k;n++){j=atomicLoad(&members[cfg.n+j]);}
    return j;
}
fn internal_face(f:u32)->bool{
    let q=point(f/2u)+axis(f%2u);if !valid(q){return false;}
    return group_owner(f/2u)==group_owner(at(q));
}
// A cell asks to merge when it holds deposit and the common volume leaving
// it in an update is at least four times its free inventory: alone, it
// would need that many subdivisions.
@compute @workgroup_size(128) fn request_groups(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=point(i);let free=free_inventory(i);var outgoing=0.;
    atomicStore(&members[i],cfg.n);atomicStore(&members[cfg.n+i],cfg.n);
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let q=p+select(-axis(a),axis(a),side==1u);if !valid(q){continue;}
        let j=at(q);let f=2u*select(j,i,side==1u)+a;
        outgoing+=max(select(-1.,1.,side==1u)*faces[f].volume.w,0.);
    }}
    // Compared as products: the reciprocal of a subnormal inventory turns
    // no flow into NaN.
    let needs_merge=cells[i].bed.z>0.&&outgoing>0.&&(cfg.dt/cfg.h)*outgoing>=4.*free;
    scratch[GROUP_REQUEST_ROW*cfg.n+i]=vec4<f32>(select(0.,1.,needs_merge),0.,free,outgoing);
}
// The parent of a cell that asks: the neighbour across an open face with
// the largest free inventory above its own, the larger index on a tie, so
// that the chains of parents hold no cycle.
@compute @workgroup_size(128) fn choose_groups(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let own=scratch[GROUP_REQUEST_ROW*cfg.n+i];let p=point(i);
    var parent=i;var largest=own.z;
    if own.x>0.{
        for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
            let q=p+select(-axis(a),axis(a),side==1u);if !valid(q){continue;}
            let j=at(q);let candidate=scratch[GROUP_REQUEST_ROW*cfg.n+j];let f=2u*select(j,i,side==1u)+a;
            if free_area(f)>0.&&(candidate.z>largest||(candidate.z==largest&&j>parent)){
                parent=j;largest=candidate.z;
            }
        }}
    }
    scratch[GROUP_PARENT_ROW*cfg.n+i]=vec4<f32>(f32(parent),0.,0.,0.);
}
@compute @workgroup_size(128) fn resolve_groups(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}var root=i;
    loop{let parent=u32(scratch[GROUP_PARENT_ROW*cfg.n+root].x);if parent==root{break;}root=parent;}
    scratch[GROUP_MAP_ROW*cfg.n+i]=vec4<f32>(f32(root),0.,scratch[GROUP_REQUEST_ROW*cfg.n+i].z,0.);
    if root!=i{
        let previous=atomicExchange(&members[root],i);
        atomicStore(&members[cfg.n+i],previous);
    }
}
@compute @workgroup_size(128) fn accept_groups(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n||group_owner(i)!=i{return;}
    // Only the root writes its list. Sorted by index, the sums over its
    // members do not depend on the order the atomics inserted them.
    var head=atomicLoad(&members[i]);var sorted=cfg.n;var count=0u;
    loop{
        if head==cfg.n{break;}let item=head;head=atomicLoad(&members[cfg.n+item]);
        var previous=cfg.n;var cursor=sorted;
        loop{if cursor>=item{break;}previous=cursor;cursor=atomicLoad(&members[cfg.n+cursor]);}
        atomicStore(&members[cfg.n+item],cursor);
        if previous==cfg.n{sorted=item;}else{atomicStore(&members[cfg.n+previous],item);}
        count++;
    }
    atomicStore(&members[i],sorted);
    var total=scratch[GROUP_REQUEST_ROW*cfg.n+i].z;var cursor=sorted;
    loop{if cursor==cfg.n{break;}total+=scratch[GROUP_REQUEST_ROW*cfg.n+cursor].z;cursor=atomicLoad(&members[cfg.n+cursor]);}
    scratch[GROUP_MAP_ROW*cfg.n+i].y=f32(count);scratch[GROUP_MAP_ROW*cfg.n+i].w=total;
}

fn native_payload(i:u32)->GroupPayload{
    let c=cells[i];var out:GroupPayload;out.volume=mobile_volumes(c);out.momentum=c.momentum;
    out.stress=vec2<f32>(c.stress.normal,c.stress.shear);out.powder=vec2<f32>(c.powder.mobile,c.powder.free);return out;
}
// A group's payload is kept in four scratch rows: the volumes, the four
// momenta, and (free powder, normal stress, shear stress, powder).
fn group_payload(i:u32)->GroupPayload{
    let owner=group_owner(i);
    let base=GROUP_PAYLOAD_ROW*cfg.n+4u*owner;var out:GroupPayload;
    out.volume=scratch[base];out.momentum[0]=scratch[base+1u].xy;out.momentum[1]=scratch[base+1u].zw;
    out.momentum[2]=scratch[base+2u].xy;out.momentum[3]=scratch[base+2u].zw;
    let carried=scratch[base+3u];out.stress=carried.yz;out.powder=carried.wx;return out;
}
// The group's gas of identity l + 1: (amount, identity, the share of the
// sweeps' increase that the stock pays (gas_capacity), 0).
fn group_label_index(i:u32,l:u32)->u32{
    let owner=group_owner(i);
    return GROUP_LABEL_ROW*cfg.n+GAS_SLOTS*owner+l;
}
fn group_label(i:u32,l:u32)->vec4<f32>{return scratch[group_label_index(i,l)];}
// Owner i gathers its members' payload and gas identities for this
// subdivision.
fn cache_group(i:u32){
    var value=native_payload(i);let count=group_count(i);
    for(var member=1u;member<count;member++){
        let right=native_payload(group_member(i,member));value.volume+=right.volume;
        for(var phase=0u;phase<4u;phase++){value.momentum[phase]+=right.momentum[phase];}
        value.stress+=right.stress;value.powder+=right.powder;
    }
    let base=GROUP_PAYLOAD_ROW*cfg.n+4u*i;scratch[base]=value.volume;
    scratch[base+1u]=vec4<f32>(value.momentum[0],value.momentum[1]);
    scratch[base+2u]=vec4<f32>(value.momentum[2],value.momentum[3]);
    scratch[base+3u]=vec4<f32>(value.powder.y,value.stress,value.powder.x);
    for(var l=0u;l<GAS_SLOTS;l++){
        var amount=geometry[layer_index(i,l)].x;
        for(var member=1u;member<count;member++){amount+=geometry[layer_index(group_member(i,member),l)].x;}
        scratch[group_label_index(i,l)]=vec4<f32>(amount,f32(l+1u),0.,0.);
    }
}
// The momentum the amount `amount` of a phase carries out of a payload.
fn payload_momentum(c:GroupPayload,amount:f32,phase:u32)->vec2<f32>{
    if amount==0.||c.volume[phase]<=0.{return vec2<f32>(0.);}
    return (amount/c.volume[phase])*c.momentum[phase];
}
fn payload_shear(c:GroupPayload,amount:f32)->vec2<f32>{
    let solid=c.volume.y+c.volume.z;
    if amount==0.||solid<=0.{return vec2<f32>(0.);}return (amount/solid)*c.stress;
}
// The powder that a white flux `amount` (of either sign) carries out of the
// payload it leaves, all of it and its free part: the shares of the white,
// which are at most all of it (the white's own bookkeeping clips at zero
// where a cell is drained).
fn payload_powder(c:GroupPayload,amount:f32)->vec2<f32>{
    if amount==0.||c.volume.y<=0.{return vec2<f32>(0.);}
    let total=clamp(c.powder.x/c.volume.y,0.,1.);
    return amount*vec2<f32>(total,clamp(c.powder.y/c.volume.y,0.,total));
}
