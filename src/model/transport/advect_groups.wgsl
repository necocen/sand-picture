// The subdivision's last kernel: each transport group, written by its owner
// alone, takes the limited fluxes of its outer faces and shares the result
// among its members. The fluxes and what they carry were stored before
// this pass, so no write here changes a donor that a neighbour reads.
@compute @workgroup_size(128) fn advect_groups(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let i=id.x;if i>=cfg.n||group_owner(i)!=i{return;}let old=group_payload(i);
    var incoming:GroupPayload;var outgoing=vec4<f32>(0.);var outgoing_powder=vec2<f32>(0.);
    var outgoing_common=0.;
    for(var member=0u;member<group_count(i);member++){
      let p=point(group_member(i,member));
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
        if value.x<0.{incoming.momentum[0]-=sign*transport[f].momentum[0];}
        if value.y<0.{incoming.momentum[1]-=sign*transport[f].momentum[1];}
        if value.z<0.{incoming.momentum[2]-=sign*transport[f].momentum[2];}
        var payload=transport[f].forward;if side==1u{payload=transport[f].reverse;}
        incoming.stress+=payload;
        outgoing_common+=max(sign*common_carrier(f),0.);
      }}
      scratch[PLUG_INFLOW_ROW*cfg.n+at(p)]+=inflow;
    }
    let retained=max(old.volume-outgoing,vec4<f32>(0.));var result:GroupPayload;
    result.volume=retained+incoming.volume;
    let powder=clamp(max(old.powder.x-outgoing_powder.x,0.)+incoming.powder.x,0.,result.volume.y);
    result.powder=vec2<f32>(powder,clamp(max(old.powder.y-outgoing_powder.y,0.)+incoming.powder.y,0.,powder));
    result.momentum[0]=incoming.momentum[0];
    if old.volume.x>0.{result.momentum[0]+=(retained.x/old.volume.x)*old.momentum[0];}
    result.momentum[1]=incoming.momentum[1];
    if old.volume.y>0.{result.momentum[1]+=(retained.y/old.volume.y)*old.momentum[1];}
    result.momentum[2]=incoming.momentum[2];
    if old.volume.z>0.{result.momentum[2]+=(retained.z/old.volume.z)*old.momentum[2];}
    result.momentum[3]=old.momentum[3];
    let solid=old.volume.y+old.volume.z;var factor=0.;
    if solid>0.{factor=(retained.y+retained.z)/solid;}
    result.stress=factor*old.stress+incoming.stress;
    let count=group_count(i);let total_free=scratch[GROUP_MAP_ROW*cfg.n+i].w;
    var remainder=1.;
    for(var member=1u;member<=count;member++){
        let slot=select(member,0u,member==count);let j=group_member(i,slot);var fraction=remainder;
        if slot>0u{fraction=scratch[GROUP_MAP_ROW*cfg.n+j].z/total_free;remainder-=fraction;}
        let c=cells[j];
        put_mobile_volumes(j,vec4<f32>(fraction*result.volume.xyz,c.volume.gas));
        cells[j].momentum[0]=fraction*result.momentum[0];cells[j].momentum[1]=fraction*result.momentum[1];
        cells[j].momentum[2]=fraction*result.momentum[2];
        put_stress(j,vec4<f32>(c.stress.pressure,fraction*result.stress,c.stress.load),c.stress.suction);
        cells[j].powder.mobile=fraction*result.powder.x;cells[j].powder.free=fraction*result.powder.y;
        geometry[layer_index(j,3u)].w=max(geometry[layer_index(j,3u)].w,outgoing_common);
    }
}
