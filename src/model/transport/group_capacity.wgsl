// The carry's capacities, by transport group (a subdivision's third and
// fifth kernels): the share of the high fluxes' increase over the low ones
// that a group's stocks can pay, and the sand and gas it has room to gain.

// For each gas identity, the share of the sweeps' increase over the low
// flux that the group's stock pays (the group label's z).
@compute @workgroup_size(128) fn gas_capacity(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let i=id.x;if i>=cfg.n||group_owner(i)!=i{return;}
    if group_payload(i).volume.w<=0.{return;}
    var increase:array<f32,GAS_SLOTS>;var low_out:array<f32,GAS_SLOTS>;
    for(var member=0u;member<group_count(i);member++){
      let p=point(group_member(i,member));
      for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;if internal_face(f){continue;}
        let sign=select(-1.,1.,side==1u);let origin=1u-side;
        let amount=common_carrier(f);let upwind=origin==select(1u,0u,amount>=0.);
        let donor=select(at(base+axis(a)),at(base),origin==0u);
        let inventory=group_payload(donor).volume;let labels=group_label_index(donor,0u);
        for(var l=0u;l<GAS_SLOTS;l++){
            let low=sign*upwind_label_flux(upwind,amount,inventory,geometry[labels+l].x);low_out[l]+=low;
            increase[l]+=max(sign*source_gas_flux(f,l,origin)-low,0.);
        }
      }}
    }
    for(var l=0u;l<GAS_SLOTS;l++){
        let slot=group_label_index(i,l);geometry[slot].z=capacity_ratio(geometry[slot].x-low_out[l],increase[l]);
    }
}
// The outgoing ratio of each phase and the receiving ratio. Hard grains do
// not pack past the random close packing: 0.64 for equal spheres (Scott
// and Kilgour 1969), about 0.62-0.66 for this mixture in the 1 mm gap
// (Desmond and Weeks 2009). The low fluxes mix bounded compositions into a
// bounded one; the high ones squeeze the wet part where a group gains
// grains or gas over its low share, through a face or by keeping back what
// the low flux would send out. Grains replace water in the wet part; gas
// takes wet volume away, and only the free water beyond the dense part.
@compute @workgroup_size(128) fn material_capacity(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let i=id.x;if i>=cfg.n||group_owner(i)!=i{return;}let v=group_payload(i).volume;
    var low_out=vec4<f32>(0.);var increase=vec4<f32>(0.);var low_nonwater_divergence=0.;var incoming_increase=0.;
    var low_divergence=vec4<f32>(0.);var incoming_grains=0.;
    var bed=0.;var dense=0.;var packed=false;
    for(var member=0u;member<group_count(i);member++){
      let j=group_member(i,member);let p=point(j);bed+=cells[j].bed.z;
      // The dense part of a cell: its grains at their body's packing if it
      // is a body's last layer (BODY_ROW), else at their own packing if
      // caged and at the freezing packing if looser. A bubble does not
      // filter the water out of a packing or a suspension.
      let cj=cells[j];let gj=grains(cj);let solid_j=gj.x+gj.y;let wet_j=max(cj.volume.water,0.)+solid_j;
      let entry=geometry[BODY_ROW*cfg.n+j];
      if entry.y>=0.&&entry.x>0.&&entry.x<1.{dense+=entry.x*wet_j;}
      else if solid_j>0.&&wet_j>0.{dense+=solid_j/max(solid_j/wet_j,FREEZING_PACKING);}
      if solid_j>0.&&wet_j>0.&&solid_j>=CRITICAL_PACKING*wet_j{packed=true;}
      for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;if internal_face(f){continue;}let sign=select(-1.,1.,side==1u);let amount=common_carrier(f);
        let donor=select(at(base+axis(a)),at(base),amount>=0.);
        let low=sign*common_low(f,donor,amount);let high=sign*transport[f].volume;
        low_out+=max(low,vec4<f32>(0.));increase+=max(max(high,vec4<f32>(0.))-max(low,vec4<f32>(0.)),vec4<f32>(0.));
        low_nonwater_divergence+=low.y+low.z+low.w;low_divergence+=low;
        let incoming=max(low.yzw-high.yzw,vec3<f32>(0.));incoming_increase+=incoming.x+incoming.y+incoming.z;
        incoming_grains+=incoming.x+incoming.y;
      }}
    }
    var ratios=vec4<f32>(1.);
    for(var phase=0u;phase<4u;phase++){ratios[phase]=capacity_ratio(v[phase]-low_out[phase],increase[phase]);}
    put_outgoing_capacity(i,ratios);
    let after=v-low_divergence;let wet=after.x+after.y+after.z;
    let incoming_gas=incoming_increase-incoming_grains;
    // A packing admits no grain through its pores: the throats between its
    // grains are a fraction of a grain (about a fifth of a diameter in a
    // close packing of equal spheres). Grains that settle or are pressed
    // against a group at the critical packing stop at its face, as at a
    // sieve: it gains grains only with the bulk flow (the low flux).
    var grain_room=CLOSE_PACKING*wet-after.y-after.z;if packed{grain_room=0.;}
    var receive=min(capacity_ratio(grain_room,incoming_grains),
        capacity_ratio(max(wet-dense,0.),incoming_gas));
    // A group with deposit gains no more sand and gas than the water that
    // the low fluxes leave it.
    if bed>0.{receive=min(receive,capacity_ratio(v.x+low_nonwater_divergence,incoming_increase));}
    put_receiving_capacity(i,receive);
}
