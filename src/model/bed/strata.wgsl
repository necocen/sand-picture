// The record of the bed's layers as they were laid. The bed stores one
// composition per cell, but a deposit is laid in layers a few grains thick:
// each column keeps STRATA_RES slices per cell height, with the coarse share
// of the slice's solid and the time it was laid. The record is drawn and the
// surface flow reads it (peel); it is written at the start of an update.

// Per column: [0] = (recorded height in cells, time in s), [1] = unused,
// [2] = (time a large drop began, 0), then one (coarse share, time) per slice
// from the floor up.
@group(0) @binding(12) var<storage,read_write> strata:array<vec2<f32>>;
const STRATA_RES:u32=8u;
const STRATA_HEADER:u32=3u;
// Seconds a record outlasts a fall of its bed by more than a cell: a column
// released for lost floor contact is usually captured again in place.
const STRATA_HOLD:f32=1.;
fn strata_base(x:u32)->u32{return x*(STRATA_HEADER+cfg.ny*STRATA_RES);}
// The solid (cell volumes) one slice of cell c's bed holds at its packing.
fn slice_solid(c:Cell)->f32{
    let solid=max(c.bed.x,0.)+max(c.bed.y,0.);if solid<=0.||c.bed.z<=0.{return 0.;}
    return solid/c.bed.z/f32(STRATA_RES);
}
// The grains (fine, coarse) that the slices of column x's row y hold below
// the record's height `recorded` (in slices), at the cell's packing and at
// most the cell's grains. A slice the height cuts holds its part.
fn recorded_grains(x:u32,y:u32,recorded:f32)->vec2<f32>{
    let c=cells[y*cfg.nx+x];let slice=slice_solid(c);if slice<=0.{return vec2<f32>(0.);}
    let base=strata_base(x)+STRATA_HEADER;var held=vec2<f32>(0.);
    for(var k=y*STRATA_RES;k<(y+1u)*STRATA_RES&&f32(k)<recorded;k++){
        let part=min(1.,recorded-f32(k));let coarse=strata[base+k].x;held+=part*slice*vec2<f32>(1.-coarse,coarse);
    }
    let amount=held.x+held.y;let solid=max(c.bed.x,0.)+max(c.bed.y,0.);
    if amount>solid{held*=solid/amount;}
    return held;
}
// The grains (fine, coarse) of the top `amount` (cell volumes of solid) of
// column x's bed, last laid first: what the cells hold beyond the record
// (laid since it was written), then the record's slices from the top down.
//
// The sand stays as it was laid: a flow neither mixes nor sorts it. A
// surface flow is a shear layer 1-2 mm deep that grains cross by shear
// diffusion alone, D = 0.046 gamma_dot d^2 (Duan et al. 2021): sqrt(2Dt) =
// 0.25-0.4 mm, two or three slices, in the 10-20 s down a flank at 3-5 per
// second. The black, 1.68 times larger and 1.92 times denser than the white
// (2.55 submerged), rises by its size and sinks by its density, which
// nearly cancel (ibid.). So the flow's top, which runs fastest, carries the
// last-laid sand first.
fn peel(x:u32,amount:f32)->vec2<f32>{
    var left=amount;var taken=vec2<f32>(0.);if left<=0.{return taken;}
    let recorded=strata[strata_base(x)].x*f32(STRATA_RES);let base=strata_base(x)+STRATA_HEADER;
    let top=min(u32(terrain[x].y)+1u,cfg.ny-1u);
    for(var row=0u;row<=top;row++){
        let y=top-row;let c=cells[y*cfg.nx+x];let slice=slice_solid(c);if slice<=0.{continue;}
        let held=recorded_grains(x,y,recorded);
        let fresh=max(max(c.bed.xy,vec2<f32>(0.))-held,vec2<f32>(0.));let laid=fresh.x+fresh.y;
        if laid>0.{let t=min(left,laid);taken+=fresh*(t/laid);left-=t;if left<=0.{return taken;}}
        let all=held.x+held.y;if all<=0.{continue;}
        // The cell's slices from its top down, scaled as recorded_grains
        // scaled them to the cell's grains.
        var nominal=0.;
        for(var k=y*STRATA_RES;k<(y+1u)*STRATA_RES&&f32(k)<recorded;k++){nominal+=min(1.,recorded-f32(k))*slice;}
        let scale=all/max(nominal,1e-30);
        for(var j=0u;j<STRATA_RES;j++){
            let k=(y+1u)*STRATA_RES-1u-j;if f32(k)>=recorded{continue;}
            let part=min(1.,recorded-f32(k))*slice*scale;let coarse=strata[base+k].x;
            let t=min(left,part);taken+=t*vec2<f32>(1.-coarse,coarse);left-=t;if left<=0.{return taken;}
        }
    }
    return taken;
}
fn coarse_share(grains:vec2<f32>,otherwise:f32)->f32{
    if grains.x+grains.y<=0.{return otherwise;}return grains.y/(grains.x+grains.y);
}
// A column whose bed grew records the new slices with the composition of the
// sand laid since the record was written (each cell's grains beyond its
// recorded slices); one whose bed fell keeps the slices below its new
// height. A slice laid in two parts keeps their mean composition and time.
@compute @workgroup_size(128) fn record_strata(@builtin(global_invocation_id) id:vec3<u32>){
    let x=id.x;if x>=cfg.nx{return;}let base=strata_base(x);
    let header=strata[base];let hold=strata[base+2u].x;
    let height=clamp(terrain[x].x,0.,f32(cfg.ny));let time=header.y+cfg.dt;
    if height<header.x-1.{
        // A large drop: the record is kept for STRATA_HOLD.
        if hold==0.{strata[base+2u]=vec2<f32>(time,0.);strata[base]=vec2<f32>(header.x,time);return;}
        if time-hold<STRATA_HOLD{strata[base]=vec2<f32>(header.x,time);return;}
    }
    if height>header.x{
        let lo=header.x*f32(STRATA_RES);let hi=min(height*f32(STRATA_RES),f32(cfg.ny*STRATA_RES));
        var row=cfg.ny;var share=.5;
        for(var k=u32(floor(lo));f32(k)<hi;k++){
            let start=max(lo,f32(k));let fill=min(hi,f32(k+1u))-start;if fill<=0.{continue;}
            let y=k/STRATA_RES;
            if y!=row{
                row=y;let c=cells[y*cfg.nx+x];let grains=max(c.bed.xy,vec2<f32>(0.));
                share=coarse_share(max(grains-recorded_grains(x,y,lo),vec2<f32>(0.)),coarse_share(grains,.5));
            }
            var slice=vec2<f32>(share,time);
            let kept=start-f32(k);
            if kept>0.{slice=(kept*strata[base+STRATA_HEADER+k]+fill*slice)/(kept+fill);}
            strata[base+STRATA_HEADER+k]=slice;
        }
    }
    strata[base]=vec2<f32>(height,time);strata[base+1u]=vec2<f32>(0.);strata[base+2u]=vec2<f32>(0.);
}
// A turn lifts the bed off the floor and the next update releases it: the
// record starts again from the floor and its clock from zero (only the order
// of the times within one drainage matters, and a clock counting for days
// would stop advancing in f32). Slices above the height are never read.
@compute @workgroup_size(128) fn clear_strata(@builtin(global_invocation_id) id:vec3<u32>){
    let x=id.x;if x>=cfg.nx{return;}let base=strata_base(x);
    strata[base]=vec2<f32>(0.);strata[base+1u]=vec2<f32>(0.);strata[base+2u]=vec2<f32>(0.);
}
