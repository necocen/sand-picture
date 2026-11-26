// Normal pressure is the multiplier of a cell's compression row, not an
// advected constituent. Keep that row's aperture and wet-volume coordinate
// from the last stress solve. Project its retained traction into the new
// row in the positive response metric, without carrying it to other cells.
@compute @workgroup_size(128) fn transfer_contact(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}
    let history=geometry[SOLVE_STATE_ROW*cfg.n+i];
    if history.x<=0.||history.y<=0.{
        cells[i].stress.pressure=0.;cells[i].stress.load=0.;return;
    }
    let p=point(i);
    let old=geometry[SOLVE_APERTURE_ROW*cfg.n+i];
    var overlap=0.;var diagonal=0.;
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=p-select(axis(a),vec2<i32>(0),side==1u);
        if !valid(base)||!valid(base+axis(a)){continue;}
        let f=2u*at(base)+a;let face=faces[f];let grains=face.composition.y+face.composition.z;
        let solid=face.composition.w*grains;let opening=face.coarse_momentum.z;
        if solid<1e-30||opening<=0.{continue;}
        let contrast=pressure_common_response(f);
        let mobility=face.grain_response.w+grains*face.response.w*contrast*contrast;
        if mobility<=0.{continue;}
        let factor=mobility/solid;
        diagonal+=factor*opening*opening;
        overlap+=factor*opening*min(opening,old[2u*a+side]);
    }}
    let wet=free_wet(cells[i]);var pressure=0.;
    if diagonal>0.&&overlap>0.&&history.y>0.&&wet>0.&&history.x>0.{
        // Restrict the old bulk load to the retained wet material. Newly
        // arrived water has no normal-pressure history. For nested subcell
        // domains the common volume is min(W_old,W_new); this is an upper
        // bound on retained volume for a general moving interface. Extending
        // old intrinsic traction over all W_new invents a large load when
        // an almost empty cell refills. This is only the constraint guess;
        // the ordinary granular solve still determines the new pressure.
        let multiplier=frexp(history.x);let ratio=frexp(clamp(overlap/diagonal,0.,1.));
        let before=frexp(history.y);let after=frexp(min(wet,history.y));
        if before.fract!=0.{
            let exp=clamp(multiplier.exp+ratio.exp+after.exp-before.exp,-126,126);
            pressure=ldexp(multiplier.fract*ratio.fract*after.fract/before.fract,exp);
        }
    }
    cells[i].stress.pressure=pressure;cells[i].stress.load=pressure;
}
