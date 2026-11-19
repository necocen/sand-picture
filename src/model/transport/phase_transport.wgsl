// The carry's fluxes through a face: compose_layer_flux builds the high
// flux of the gas identities and the two sands, limit_layer_flux blends low
// and high within the groups' capacities (a subdivision's fourth and sixth
// kernels).

// The low flux of a gas identity: the upwind group's share of the carrier.
fn upwind_label_flux(upwind:bool,amount:f32,inventory:vec4<f32>,label:f32)->f32{
    let total=inventory.x+inventory.y+inventory.z+inventory.w;
    if !upwind||amount==0.||total<=0.{return 0.;}
    return amount*(label/total);
}
// The high flux of face f. Gas G: each identity's sweep (layer_flux),
// blended with its low flux by its stock's capacity (gas_capacity). Sands:
// F_s = (Q - G + dt/h A_w du_s) c_s, with Q the common volume, A_w the wet
// aperture, du_s the species' slip and c_s its upwind wet concentration.
// The water is the remainder, so that the four sum to Q.
@compute @workgroup_size(128) fn compose_layer_flux(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let f=id.x;if f>=2u*cfg.n{return;}let i=f/2u;let a=f%2u;let q=point(i)+axis(a);if !valid(q)||internal_face(f){return;}
    let j=at(q);let amount=common_carrier(f);var gas=0.;let upwind=select(1u,0u,amount>=0.);
    for(var origin=0u;origin<2u;origin++){
        let donor=select(j,i,origin==0u);var candidate:array<f32,GAS_SLOTS>;var total=0.;var low_total=0.;
        let inventory=group_payload(donor).volume;
        if inventory.w<=0.{continue;}
        let labels=group_label_index(donor,0u);var lows:array<f32,GAS_SLOTS>;
        // Gas that is held back (layer_flux) or refused (gas_sieved) has no
        // candidate.
        let receiver=select(i,j,origin==0u);
        let closed=teeth_shadow(point(donor),point(receiver))||plug_wall(donor,receiver)||close_drained(cells[receiver])||(border_cell(receiver)&&!border_cell(donor))||gas_sieved(f,donor);
        for(var l=0u;l<GAS_SLOTS;l++){
            let entry=scratch[labels+l];let low=upwind_label_flux(origin==upwind,amount,inventory,entry.x);
            lows[l]=low;
            candidate[l]=select(mix(low,source_gas_flux(f,l,origin),entry.z),0.,closed||spacer_channel_closed(l,receiver));total+=candidate[l];low_total+=low;
        }
        var blend=1.;
        // Gas that moves with the common flow carries no more than the
        // carrier; a relative sweep is bounded by the stocks alone.
        if gas_carrier(f)==amount&&abs(total)>abs(amount)&&abs(total)>abs(low_total){
            blend=clamp((abs(amount)-abs(low_total))/(abs(total)-abs(low_total)),0.,1.);
        }
        for(var l=0u;l<GAS_SLOTS;l++){
            let value=select(mix(lows[l],candidate[l],blend),0.,closed||spacer_channel_closed(l,receiver));
            put_source_gas_flux(f,l,origin,value);gas+=value;
        }
    }
    let interval=cfg.dt/(carry_divisions()*cfg.h);let face=faces[f];var flux=vec4<f32>(0.);
    // On a chain's face (a joint's mouth, a column's exit) the sand crosses
    // at the chain's allowance (joint_mouth), neither less nor more,
    // whatever slip the mechanics gave the face.
    let mouth=joint_mouth(f,interval);
    // Whether the white crosses with the water (not sieved, not a plug's).
    var white_crosses=false;
    for(var phase=0u;phase<2u;phase++){
        // Grains that the face keeps back (common_low) stay behind and the
        // water passes alone. Not on a chain's face: its slot is a border's
        // or a plug's width below the grid, at least the jamming width,
        // whatever aperture the gas planes leave the face.
        let sieve_donor=select(j,i,amount>=0.);
        if mouth<0.&&sieve_holds(f,sieve_donor,amount,gas)&&(phase_locked(f)||glides(f)||grains_refused(f,sieve_donor)||standing_face(f)||(phase==1u&&coarse_locked(f))){continue;}
        let speed=face.volume[phase+1u];let opening=face.coarse_momentum.z;
        // Inside a packed skeleton the grains do not move relative to the
        // water (skeleton_face): they ride the common flow alone.
        var relative=interval*opening*face.velocity[phase+1u];
        if skeleton_face(f){relative=0.;}
        let carrier=amount-gas+relative;
        let source_cell=select(j,i,carrier>=0.);
        // A body's last layer crosses its far face no faster than its body
        // pushes it (layer_supply).
        let supply=layer_supply(f,source_cell,interval);
        if free_wet(cells[source_cell])>0.&&carrier!=0.{
            flux[phase]=carrier*wet_concentration(point(source_cell),a,carrier)[phase];
            flux[phase]=sign(flux[phase])*min(abs(flux[phase]),supply[phase]);
        }
        // A column's plug is jammed in its slot (column_plug): through its
        // faces that are not the chain's, the water moves alone.
        let other=select(i,j,source_cell==i);
        if mouth<0.&&(column_plug(source_cell)||column_plug(other)){flux[phase]=0.;}
        else if phase==0u{white_crosses=true;}
        // The chain's allowance, in the species' shares of the cell above.
        if mouth>=0.{
            let body=cells[j];let g=grains(body);let share=g[phase]/max(g.x+g.y,1e-30);
            flux[phase]=-mouth*share;
        }
    }
    // A packed face whose grains fall away breaches into the water beside
    // it at no less than its breaching rate (free_face_breach).
    if mouth<0.{
        let breach=free_face_breach(f,interval);
        if breach!=0.{
            let g=grains(cells[select(i,j,breach<0.)]);let solid=max(g.x+g.y,1e-30);
            for(var phase=0u;phase<2u;phase++){
                let least=breach*g[phase]/solid;
                if breach>0.{flux[phase]=max(flux[phase],least);}else{flux[phase]=min(flux[phase],least);}
            }
        }
    }
    // The free powder lags the white it is part of where the white crosses
    // freely (powder_drift); its lag is part of the white's flux.
    var drift=0.;
    if mouth<0.&&white_crosses{drift=powder_drift(f,i,j,interval,amount-gas,flux.x);flux.x+=drift;}
    transport[f].volume=vec4<f32>(amount-gas-flux.x-flux.y,flux.xy,gas);
    transport[f].powder=vec2<f32>(drift);
}
fn side_payload(first:GroupPayload,second:GroupPayload,forward:bool)->GroupPayload{
    if forward{return first;}return second;
}
// The flux of face f: low + blend (high - low) for every phase and gas
// identity, the blend the least capacity that the high flux draws on (a
// donor's stock where a phase leaves faster than its low flux, a receiver's
// room where it gains sand or gas), with what the flux carries.
@compute @workgroup_size(128) fn limit_layer_flux(@builtin(global_invocation_id) id:vec3<u32>){
    if carry_plan.current>=carry_plan.steps{return;}
    let f=id.x;if f>=2u*cfg.n{return;}let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q)||internal_face(f){return;}
    let j=at(q);let amount=common_carrier(f);let donor=select(j,i,amount>=0.);
    let payload_i=group_payload(i);let payload_j=group_payload(j);
    let owner_i=group_owner(i);let owner_j=group_owner(j);
    let low=common_low(f,donor,amount);let high=transport[f].volume;let drift=transport[f].powder.y;var blend=1.;
    let out_i=outgoing_capacity(owner_i);let out_j=outgoing_capacity(owner_j);
    let receive_i=receiving_capacity(owner_i);let receive_j=receiving_capacity(owner_j);
    for(var phase=0u;phase<4u;phase++){
        if max(high[phase],0.)>max(low[phase],0.){blend=min(blend,out_i[phase]);}
        if max(-high[phase],0.)>max(-low[phase],0.){blend=min(blend,out_j[phase]);}
        if phase>0u{
            if high[phase]-low[phase]>0.{blend=min(blend,receive_j);}
            if high[phase]-low[phase]<0.{blend=min(blend,receive_i);}
        }
    }
    var out:Transport;out.volume=mix(low,high,blend);var gas=0.;
    let upwind=select(1u,0u,amount>=0.);
    // Gas that is held back has a zero candidate and crosses only as far as
    // the blend keeps to the low flux: where the water's budget cannot take
    // its share, the bubble passes. Refused gas has no low flux either.
    for(var origin=0u;origin<2u;origin++){
      let inventory=select(payload_j.volume,payload_i.volume,origin==0u);let source_cell=select(j,i,origin==0u);
      if inventory.w<=0.{continue;}let labels=group_label_index(source_cell,0u);
      let sieved=gas_sieved(f,source_cell);
      for(var l=0u;l<GAS_SLOTS;l++){
        let low=select(upwind_label_flux(origin==upwind,amount,inventory,scratch[labels+l].x),0.,sieved);
        let value=mix(low,source_gas_flux(f,l,origin),blend);
        put_source_gas_flux(f,l,origin,value);gas+=value;
    }}
    out.volume.w=gas;out.volume.x=amount-gas-out.volume.y-out.volume.z;
    out.momentum[0]=payload_momentum(side_payload(payload_i,payload_j,out.volume.x>=0.),out.volume.x,0u);
    out.momentum[1]=payload_momentum(side_payload(payload_i,payload_j,out.volume.y>=0.),out.volume.y,1u);
    out.momentum[2]=payload_momentum(side_payload(payload_i,payload_j,out.volume.z>=0.),out.volume.z,2u);
    // The powder rides the white it is part of, and its free part lags it
    // by the high flux's drift as far as the blend keeps it.
    let lag=blend*drift;let carried=out.volume.y-lag;
    out.powder=payload_powder(side_payload(payload_i,payload_j,carried>=0.),carried)+vec2<f32>(lag);
    var swept=vec2<f32>(max(gas,0.),max(-gas,0.));
    if faces[f].velocity.w!=faces[f].volume.w{swept=gas_sweeps(f);}
    out.momentum[3]=payload_momentum(payload_i,swept.x,3u);
    out.gas_reverse=payload_momentum(payload_j,-swept.y,3u);
    let grain_flux=out.volume.yz;
    for(var phase=0u;phase<2u;phase++){
        let source=side_payload(payload_i,payload_j,grain_flux[phase]>=0.);let amount=abs(grain_flux[phase]);
        let stress=payload_shear(source,amount);
        if grain_flux[phase]>=0.{out.forward+=stress;}else{out.reverse+=stress;}
    }
    transport[f]=out;
}
