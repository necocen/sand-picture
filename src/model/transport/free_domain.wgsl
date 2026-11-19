// The free domain: the mobile mixture of a cell, outside the deposit. Its
// volume and apertures, the velocities of its sand and its gas on a face,
// the carry's low flux with what a face keeps back, and the deposit's shear
// and support. Read by the mechanics, the bed and the carry.

// Only the mixture outside the deposit, an impermeable skeleton, moves.
fn free_wet(c:Cell)->f32{
    return max(0.,c.volume.water+c.volume.fine+c.volume.coarse);
}
// The deposit's share of the cell's wet part, through the smaller of the
// two: b/(b+w) can round below one where b+w rounds to b (relaxed
// reciprocals), a water film over a full bed that no stored water makes.
fn occupied_wet_fraction(c:Cell)->f32{
    if c.bed.z<=0.{return 0.;}let free=free_wet(c);let total=c.bed.z+free;
    if c.bed.z>=free{return 1.-free/total;}
    return c.bed.z/total;
}
fn free_area(f:u32)->f32{return geometry[BED_PASSAGE_ROW*cfg.n+f].x;}
// The share of face f open to the free liquid: its wet span less the
// deposit's trace, and none where the deposit fills a cell's wet part (the
// difference of the two reconstructions would leave a residue there).
fn free_wet_aperture(f:u32)->f32{
    let i=f/2u;let q=point(i)+axis(f%2u);
    if !valid(q){return 0.;}
    if geometry[BED_OCCUPANCY_ROW*cfg.n+i].x>=1.||geometry[BED_OCCUPANCY_ROW*cfg.n+at(q)].x>=1.{return 0.;}
    return max(0.,label_liquid_aperture(f)-geometry[BED_PASSAGE_ROW*cfg.n+f].z);
}
fn face_gas(f:u32)->f32{
    let q=point(f/2u)+axis(f%2u);if !valid(q){return 0.;}
    return .5*(cells[f/2u].volume.gas+cells[at(q)].volume.gas);
}
fn stream_velocity(f:u32)->f32{
    let area=free_area(f);if area<=0.{return 0.;}
    return volume_current(f)/area;
}
// Whether the gas glides on face f. A liquid film lubricates a bubble
// against wet sand. Under the slab a raft cell holds the bubble's top below
// the slab's last layer of sand, and a vertical face between two such
// cells is cut into a strip of gas and a strip of sand: the sand stands
// with the slab and the gas carries the face's flux (with one velocity for
// both, the plates would hold the raft through that sand). The test asks
// for a strip of each, not a sliver, and an interface that cuts the face.
fn glides(f:u32)->bool{
    let i=f/2u;let a=f%2u;if a!=0u{return false;}let p=point(i);let q=p+axis(a);if !valid(q){return false;}
    let l=cells[i];let r=cells[at(q)];
    if min(l.volume.gas,r.volume.gas)<.25||.5*(l.volume.gas+r.volume.gas)>.9{return false;}
    let gl=grains(l);let gr=grains(r);let sand=gl.x+gl.y+gr.x+gr.y;
    let wet=max(l.volume.water,0.)+max(r.volume.water,0.)+sand;
    if wet<=0.||sand<.25*wet{return false;}
    let o=axis(1u);var along=0.;var count=0.;
    for(var k=0u;k<2u;k++){
        let c=select(p,q,k==1u);
        if valid(c+o)&&valid(c-o){along+=.5*(cells[at(c+o)].volume.gas-cells[at(c-o)].volume.gas);count+=1.;}
    }
    return count>0.&&abs(along/count)>abs(r.volume.gas-l.volume.gas);
}
// The gas strip's velocity over the face's mean: one over the strip's
// share of the face by the gas planes, the share the carry's sweep sees
// (by the cells' gas the sweep would fall short of the face's flux).
fn glide_ratio(f:u32)->f32{return 1./max(1.-faces[f].coarse_momentum.z,.25);}
// The sand's velocity on a face: the flow's, none on a gliding face. The
// gas's: its strip's on a gliding face; elsewhere the flow's with the slip
// of a bubble smaller than the gap that crosses the face (bubble_slip),
// the lower or left cell's forward, else the upper or right cell's back.
fn wet_velocity(f:u32)->f32{return select(stream_velocity(f),0.,glides(f));}
fn gas_velocity(f:u32)->f32{
    let a=f%2u;let stream=stream_velocity(f);let q=point(f/2u)+axis(a);
    if glides(f){return stream*glide_ratio(f);}
    if !valid(q){return stream;}
    let forward=stream+bubble_slip(f/2u)[a];if forward>0.{return forward;}
    return min(stream+bubble_slip(at(q))[a],0.);
}
// The low flux (water, fine, coarse, gas; cells) of the carrier `amount`
// through face f: the donor group's composition, the water taking the
// place of the gas and the grains that the face keeps back.
fn common_low(f:u32,donor:u32,amount:f32)->vec4<f32>{
    let free=group_payload(donor).volume;
    let total=free.x+free.y+free.z+free.w;
    if amount==0.||total<=0.{return vec4<f32>(0.);}
    var share=free/total;
    if free.w>0.&&gas_sieved(f,donor){share=vec4<f32>(share.x+share.w,share.y,share.z,0.);}
    if sieve_holds(f,donor,amount,transport[f].volume.w){
        if phase_locked(f)||glides(f)||grains_refused(f,donor)||standing_face(f){share=vec4<f32>(share.x+share.y+share.z,0.,0.,share.w);}
        else if coarse_locked(f){share=vec4<f32>(share.x+share.z,share.y,0.,share.w);}
        else if layer_far_face(f,donor){
            let supply=layer_supply(f,donor,cfg.dt/(carry_divisions()*cfg.h));
            let want=abs(amount)*share.yz;let excess=max(want-supply,vec2<f32>(0.))/abs(amount);
            share=vec4<f32>(share.x+excess.x+excess.y,share.y-excess.x,share.z-excess.y,share.w);
        }
    }
    return amount*share;
}
// A cell whose wet part is at the close packing takes no more grains: they
// stop at its face and the water passes, as at a sieve. The capacities
// bound only the high flux; the low one would carry sand into cells with
// little water (a bubble's cell, a wall's corner).
fn grains_refused(f:u32,donor:u32)->bool{
    let i=f/2u;let j=at(point(i)+axis(f%2u));let receiver=select(i,j,donor==i);
    let c=cells[receiver];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    return solid>0.&&solid>=CLOSE_PACKING*wet;
}
// A gap narrower than a grain passes the water and keeps the grain back, as
// a sieve, while it stands still: most of the face's current is liquid. An
// interface that sweeps across the face pushes the grains ahead of it.
fn sieves(f:u32,amount:f32,gas:f32)->bool{return abs(gas)<.5*abs(amount);}
// The sieve holds only while the donor's water can carry the wet part's
// share of all its group sends out this subdivision: a cell without free
// water pushes its grains along with the gas. (A share taken from water
// that is not there is cut at the donor and still reaches the receiver.)
fn sieve_holds(f:u32,donor:u32,amount:f32,gas:f32)->bool{
    if !sieves(f,amount,gas){return false;}
    let free=group_payload(donor).volume;
    let total=free.x+free.y+free.z+free.w;
    return free.x*total>=group_outgoing(donor)*(free.x+free.y+free.z);
}
// The common volume the donor's group sends out this subdivision.
fn group_outgoing(donor:u32)->f32{
    let owner=group_owner(donor);var outgoing=0.;
    for(var member=0u;member<group_count(owner);member++){
        let p=point(group_member(owner,member));
        for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
            let base=p-select(axis(a),vec2<i32>(0),side==1u);if !valid(base)||!valid(base+axis(a)){continue;}
            let g=2u*at(base)+a;if internal_face(g){continue;}
            outgoing+=max(select(-1.,1.,side==1u)*common_carrier(g),0.);
        }}
    }
    return outgoing;
}
// Gas stops at the face of wet sand that the plates hold still: entering
// its pores takes the capillary entry pressure, kilopascals against the
// tens of pascals a bubble presses with, and held grains do not move away.
// The flow solve closes only the faces between a bubble's inside and a
// held packing (bubble_on_packing); through the others the low flux would
// leave gas in the slab's base at every jostle of the raft. A body's last
// layer keeps the free water beside its packing open to the gas.
fn gas_refused(f:u32,donor:u32)->bool{
    let i=f/2u;let j=at(point(i)+axis(f%2u));let receiver=select(i,j,donor==i);
    let c=cells[receiver];if !packed_wet(c)||!held_still(receiver){return false;}
    let entry=geometry[BODY_ROW*cfg.n+receiver];
    return entry.y<0.||entry.x>=1.;
}
// The donor's film water crosses in the refused gas's place, while its
// group has the water for all it sends out.
fn gas_sieved(f:u32,donor:u32)->bool{
    return gas_refused(f,donor)&&group_payload(donor).volume.x>=group_outgoing(donor);
}
// The drag coefficient (Pa s/m^2) of the free liquid on the deposit's
// surface in cell i: linear shear to the still bed across the free layer,
// as deep as its volume V over the surface's wetted width w:
// eta w^2/(V h^2). A trace of free water is no layer to shear.
fn bed_shear(i:u32)->f32{
    let c=cells[i];let bed=geometry[BED_OCCUPANCY_ROW*cfg.n+i];let wet=free_wet(c);
    if bed.x<=0.||bed.x>=1.||wet<=1e-6{return 0.;}
    let p=wet_polygon(i);let y=bed.x-.5;var lo=-.5;var hi=.5;
    let gas=outer_gas_shapes(i);
    for(var k=0u;k<2u;k++){
        if gas.present[k]==0u{continue;}
        let s=gas.shapes[k];let rhs=s.z-s.y*y;
        if s.x>0.{lo=max(lo,rhs/s.x);}
        else if s.x<0.{hi=min(hi,rhs/s.x);}
        else if rhs>0.{return 0.;}
    }
    let width=max(0.,hi-lo);
    return .00094*(width/wet)*width/(cfg.h*cfg.h);
}
fn boundary_shear(f:u32)->f32{
    let q=point(f/2u)+axis(f%2u);if !valid(q){return 0.;}
    return .5*(bed_shear(f/2u)+bed_shear(at(q)));
}
// The support (N per m^3 of grains; x, and y upward) that the deposit of
// cell i needs under the apparent gravity (`lateral` along x, m/s^2): its
// saturated weight, less the pressure gradient on the faces of the column's
// open cell, less the flow's drag on its surface (bed_shear).
fn required_support(i:u32,share:vec2<f32>,lateral:f32)->vec2<f32>{
    let p=point(i);let x=u32(p.x);let row=max(u32(p.y),bed_open(x));
    let sample=vec2<i32>(p.x,i32(min(row,cfg.ny-1u)));
    var pressure=vec2<f32>(-RHO.x*lateral,RHO.x*cfg.gravity);var weight=vec2<f32>(0.);
    var sum=vec2<f32>(0.);var drag=vec2<f32>(0.);
    for(var a=0u;a<2u;a++){for(var side=0u;side<2u;side++){
        let base=sample-select(axis(a),vec2<i32>(0),side==1u);
        if valid(base)&&valid(base+axis(a)){
            let f=faces[2u*at(base)+a];
            if f.response.w>0.{sum[a]+=f.composition.w*f.support.z;weight[a]+=f.composition.w;}
        }
        let here=p-select(axis(a),vec2<i32>(0),side==1u);
        if valid(here)&&valid(here+axis(a)){
            // The flow's velocity on the face, not the gas's, which a
            // bubble smaller than the gap outruns.
            let f=2u*at(here)+a;let area=free_area(f);
            if area>0.{drag[a]+=.5*bed_shear(i)*faces[f].volume.w/area;}
        }
    }}
    for(var a=0u;a<2u;a++){if weight[a]>0.{pressure[a]=sum[a]/weight[a];}}
    let c=cells[i];let solid=c.bed.x+c.bed.y;var packing=BED_PACKING;
    if c.bed.z>0.{packing=solid/c.bed.z;}
    // The skeleton's volume includes its pore water: under a hydrostatic
    // gradient the load is exactly the grains' submerged weight.
    let density=dot(share,RHO.yz)+(1./packing-1.)*RHO.x;
    var load=vec2<f32>(-density*lateral,density*cfg.gravity)-pressure/packing;
    // The shear is a stress on the bed's surface, carried by its top layer
    // of grains: a bed thinner than one grain is grains lying on that
    // surface, each taking the drag of a grain of a full layer (Shields),
    // not the whole stress on a sliver of volume.
    if solid>0.{load-=drag/max(solid,BED_PACKING*GRAIN_LAYER/cfg.h);}
    return load;
}
