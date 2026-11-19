// The subdivision's last kernel: each transport group, written by its owner
// alone, takes the limited fluxes of its outer faces and shares the result
// among its members. The fluxes and what they carry were stored before
// this pass, so no write here changes a donor that a neighbour reads.
@compute @workgroup_size(128) fn advect_groups(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let i=id.x;if i>=cfg.n||group_owner(i)!=i{return;}let old=group_payload(i);
    var incoming:GroupPayload;var outgoing=vec4<f32>(0.);var outgoing_powder=vec2<f32>(0.);
    // The identities' sums are vectors (slots 0-3, 4-7) and a scalar (slot
    // 8), and the phase loops are written out: no array is indexed at run
    // time. The kernel is written for nine identities.
    var out_low=vec4<f32>(0.);var out_high=vec4<f32>(0.);var out_last=0.;
    var in_low=vec4<f32>(0.);var in_high=vec4<f32>(0.);var in_last=0.;
    var outgoing_common=0.;
    for(var member=0u;member<group_count(i);member++){
      let p=point(group_member(i,member));
      // The grains entering the member through its (left, right, bottom,
      // top) faces, net, for the plug descent (plug_return).
      var inflow=vec4<f32>(0.);
      for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;if internal_face(f){continue;}let sign=select(-1.,1.,side==1u);
        let value=sign*transport[f].volume;
        let face=2u*a+side;
        inflow-=(value.y+value.z)*vec4<f32>(f32(face==0u),f32(face==1u),f32(face==2u),f32(face==3u));
        outgoing+=vec4<f32>(max(value.xyz,vec3<f32>(0.)),0.);
        incoming.volume+=vec4<f32>(max(-value.xyz,vec3<f32>(0.)),0.);
        let powder=sign*transport[f].powder;
        outgoing_powder+=max(powder,vec2<f32>(0.));incoming.powder+=max(-powder,vec2<f32>(0.));
        let gas=gas_sweeps(f);outgoing.w+=gas[1u-side];incoming.volume.w+=gas[side];
        if side==1u{incoming.momentum[3]-=transport[f].gas_reverse;}
        else{incoming.momentum[3]+=transport[f].momentum[3];}
        if value.x<0.{incoming.momentum[0]-=sign*transport[f].momentum[0];}
        if value.y<0.{incoming.momentum[1]-=sign*transport[f].momentum[1];}
        if value.z<0.{incoming.momentum[2]-=sign*transport[f].momentum[2];}
        var payload=transport[f].forward;if side==1u{payload=transport[f].reverse;}
        incoming.stress+=payload;
        // Component 2*origin of an identity's flux: this side's own origin
        // leaves the group, the other enters it.
        let f0=scratch[flux_index(f,0u)];let f1=scratch[flux_index(f,1u)];let f2=scratch[flux_index(f,2u)];
        let f3=scratch[flux_index(f,3u)];let f4=scratch[flux_index(f,4u)];let f5=scratch[flux_index(f,5u)];
        let f6=scratch[flux_index(f,6u)];let f7=scratch[flux_index(f,7u)];let f8=scratch[flux_index(f,8u)];
        let leaving=side==0u;
        out_low+=sign*select(vec4<f32>(f0.x,f1.x,f2.x,f3.x),vec4<f32>(f0.z,f1.z,f2.z,f3.z),leaving);
        out_high+=sign*select(vec4<f32>(f4.x,f5.x,f6.x,f7.x),vec4<f32>(f4.z,f5.z,f6.z,f7.z),leaving);
        out_last+=sign*select(f8.x,f8.z,leaving);
        in_low-=sign*select(vec4<f32>(f0.z,f1.z,f2.z,f3.z),vec4<f32>(f0.x,f1.x,f2.x,f3.x),leaving);
        in_high-=sign*select(vec4<f32>(f4.z,f5.z,f6.z,f7.z),vec4<f32>(f4.x,f5.x,f6.x,f7.x),leaving);
        in_last-=sign*select(f8.z,f8.x,leaving);
        outgoing_common+=max(sign*common_carrier(f),0.);
      }}
      scratch[PLUG_INFLOW_ROW*cfg.n+at(p)]+=inflow;
    }
    let retained=max(old.volume-outgoing,vec4<f32>(0.));var result:GroupPayload;
    result.volume=retained+incoming.volume;
    // The powder is part of the white: no more than the white the group
    // keeps, and its free part no more than the powder.
    let powder=clamp(max(old.powder.x-outgoing_powder.x,0.)+incoming.powder.x,0.,result.volume.y);
    result.powder=vec2<f32>(powder,clamp(max(old.powder.y-outgoing_powder.y,0.)+incoming.powder.y,0.,powder));
    result.momentum[0]=incoming.momentum[0];
    if old.volume.x>0.{result.momentum[0]+=(retained.x/old.volume.x)*old.momentum[0];}
    result.momentum[1]=incoming.momentum[1];
    if old.volume.y>0.{result.momentum[1]+=(retained.y/old.volume.y)*old.momentum[1];}
    result.momentum[2]=incoming.momentum[2];
    if old.volume.z>0.{result.momentum[2]+=(retained.z/old.volume.z)*old.momentum[2];}
    result.momentum[3]=incoming.momentum[3];
    if old.volume.w>0.{result.momentum[3]+=(retained.w/old.volume.w)*old.momentum[3];}
    let solid=old.volume.y+old.volume.z;var factor=0.;
    if solid>0.{factor=(retained.y+retained.z)/solid;}
    result.stress=factor*old.stress+incoming.stress;
    let stock_low=vec4<f32>(group_label(i,0u).x,group_label(i,1u).x,group_label(i,2u).x,group_label(i,3u).x);
    let stock_high=vec4<f32>(group_label(i,4u).x,group_label(i,5u).x,group_label(i,6u).x,group_label(i,7u).x);
    let stock_last=group_label(i,8u).x;
    let labels_low=max(stock_low-out_low,vec4<f32>(0.))+in_low;
    let labels_high=max(stock_high-out_high,vec4<f32>(0.))+in_high;
    let labels_last=max(stock_last-out_last,0.)+in_last;
    let over_low=max(out_low-stock_low,vec4<f32>(0.));let over_high=max(out_high-stock_high,vec4<f32>(0.));
    var overdraw=0.;
    overdraw+=over_low.x;overdraw+=over_low.y;overdraw+=over_low.z;overdraw+=over_low.w;
    overdraw+=over_high.x;overdraw+=over_high.y;overdraw+=over_high.z;overdraw+=over_high.w;
    overdraw+=max(out_last-stock_last,0.);
    var gas=0.;
    gas+=labels_low.x;gas+=labels_low.y;gas+=labels_low.z;gas+=labels_low.w;
    gas+=labels_high.x;gas+=labels_high.y;gas+=labels_high.z;gas+=labels_high.w;
    gas+=labels_last;
    // The identities are scaled to make up the bulk gas, which is carried on
    // its own; no phase and no cell is normalised.
    var gas_scale=0.;if gas>0.{gas_scale=result.volume.w/gas;}
    let count=group_count(i);let total_free=scratch[GROUP_MAP_ROW*cfg.n+i].w;
    var remainder=1.;
    // The members share everything carried by their free inventories at the
    // grouping, the owner taking the remainder; the deposit and its pore
    // water stay.
    for(var member=1u;member<=count;member++){
        let slot=select(member,0u,member==count);let j=group_member(i,slot);var fraction=remainder;
        if slot>0u{fraction=scratch[GROUP_MAP_ROW*cfg.n+j].z/total_free;remainder-=fraction;}
        // The fourth component of the first four layers is an observation,
        // kept: slot 1 adds the gas its identities were overdrawn by, slot
        // 3 keeps the largest common volume a group sent out.
        let c=cells[j];let previous=vec4<f32>(geometry[layer_index(j,0u)].w,geometry[layer_index(j,1u)].w,
            geometry[layer_index(j,2u)].w,geometry[layer_index(j,3u)].w);
        put_mobile_volumes(j,fraction*result.volume);
        cells[j].momentum[0]=fraction*result.momentum[0];cells[j].momentum[1]=fraction*result.momentum[1];
        cells[j].momentum[2]=fraction*result.momentum[2];cells[j].momentum[3]=fraction*result.momentum[3];
        // The pressure stays with its cell, with its load and suction: the
        // next update solves it again; it is no amount to carry.
        put_stress(j,vec4<f32>(c.stress.pressure,fraction*result.stress,c.stress.load),c.stress.suction);
        cells[j].powder.mobile=fraction*result.powder.x;cells[j].powder.free=fraction*result.powder.y;
        let amounts_low=fraction*(gas_scale*labels_low);let amounts_high=fraction*(gas_scale*labels_high);
        geometry[layer_index(j,0u)]=vec4<f32>(amounts_low.x,1.,0.,previous.x);
        geometry[layer_index(j,1u)]=vec4<f32>(amounts_low.y,2.,0.,previous.y+fraction*overdraw);
        geometry[layer_index(j,2u)]=vec4<f32>(amounts_low.z,3.,0.,previous.z);
        geometry[layer_index(j,3u)]=vec4<f32>(amounts_low.w,4.,0.,max(previous.w,outgoing_common));
        geometry[layer_index(j,4u)]=vec4<f32>(amounts_high.x,5.,0.,0.);
        geometry[layer_index(j,5u)]=vec4<f32>(amounts_high.y,6.,0.,0.);
        geometry[layer_index(j,6u)]=vec4<f32>(amounts_high.z,7.,0.,0.);
        geometry[layer_index(j,7u)]=vec4<f32>(amounts_high.w,8.,0.,0.);
        geometry[layer_index(j,8u)]=vec4<f32>(fraction*(gas_scale*labels_last),9.,0.,0.);
    }
}
