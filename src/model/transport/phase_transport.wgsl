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
    let j=at(q);let amount=common_carrier(f);let gas=0.;
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
    // Down into a slot between bubbles the grains cross at the batch flux
    // at most (slot_supply): the hollow above keeps the rest.
    if mouth<0.{
        let supply=slot_supply(f,interval);let down=max(-flux.x,0.)+max(-flux.y,0.);
        if down>supply{
            let keep=supply/down;
            if flux.x<0.{flux.x*=keep;}
            if flux.y<0.{flux.y*=keep;}
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
    var out:Transport;out.volume=mix(low,high,blend);
    out.volume.w=0.;out.volume.x=amount-out.volume.y-out.volume.z;
    out.momentum[0]=payload_momentum(side_payload(payload_i,payload_j,out.volume.x>=0.),out.volume.x,0u);
    out.momentum[1]=payload_momentum(side_payload(payload_i,payload_j,out.volume.y>=0.),out.volume.y,1u);
    out.momentum[2]=payload_momentum(side_payload(payload_i,payload_j,out.volume.z>=0.),out.volume.z,2u);
    // The powder rides the white it is part of, and its free part lags it
    // by the high flux's drift as far as the blend keeps it.
    let lag=blend*drift;let carried=out.volume.y-lag;
    out.powder=payload_powder(side_payload(payload_i,payload_j,carried>=0.),carried)+vec2<f32>(lag);
    out.momentum[3]=vec2<f32>(0.);
    out.gas_reverse=vec2<f32>(0.);
    let grain_flux=out.volume.yz;
    for(var phase=0u;phase<2u;phase++){
        let source=side_payload(payload_i,payload_j,grain_flux[phase]>=0.);let amount=abs(grain_flux[phase]);
        let stress=payload_shear(source,amount);
        if grain_flux[phase]>=0.{out.forward+=stress;}else{out.reverse+=stress;}
    }
    transport[f]=out;
}
