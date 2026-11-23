// Breaching: a packed face under water gives way only as fast as water
// enters the layer behind it, which must dilate before it flows, whether or
// not the plates hold the body (they hold the body, not the face's grains),
// and only where the grains it releases leave it: a side face's must fall
// away (falls_away), a roof's drop from it into any water (roof_share). Also
// the counterflow that carries sand down a slot and the dilatancy suction.
// No kernel: body_faces, the transport's face fluxes and the mechanics call
// these.

// Water a released grain can fall through: no deposit, no bubble a grain
// layer thick, and uncaged sand (below FREEZING_PACKING of its wet part;
// from FREEZING_PACKING up grains are caged in contact and block falling grains).
fn open_water(c:Cell)->bool{
    if plug_deposit(c)||c.volume.gas*cfg.h>=GRAIN_LAYER{return false;}
    let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    return wet>0.&&solid<FREEZING_PACKING*wet;
}
// Sand that can shed grains from a face: sandy (teeth_sandy), no deposit.
fn breach_donor(c:Cell)->bool{return !plug_deposit(c)&&teeth_sandy(c);}
// Whether a grain shed into the open water at `start` leaves its face for
// good: it goes straight down through open water, else on in the direction
// `away` from the face (+1 or -1 along x) over the sand beneath it, until
// it reaches the deposit or the tank's floor. A breaching face retreats
// while a turbidity current carries its grains off; grains that come to
// rest on the upper sand (a funnel's floor, a pocket) bury its foot.
fn falls_away(start:vec2<i32>,away:i32)->bool{
    var p=start;
    for(var step=0;step<4*i32(cfg.ny);step++){
        let below=p-axis(1u);
        if !valid(below){return true;}
        let cb=cells[at(below)];
        if plug_deposit(cb){return true;}
        if open_water(cb){p=below;continue;}
        let ahead=p+vec2<i32>(away,0);
        if !valid(ahead)||!open_water(cells[at(ahead)]){return false;}
        p=ahead;
    }
    return false;
}
// The grains (cells per substep, + toward +x or +y) a packed face sheds
// across `f` into open water they fall away through (by the receiver's
// BODY_ROW), a lower bound of the transport's grain flux there: its
// breaching rate for its inclination times its packing and the wet
// aperture the mechanics recorded on `f`. A face looking down is
// roof_breach's; an aperture under the jamming width arches; the receiver
// takes no more than brings it to LOOSE_PACKING.
fn free_face_breach(f:u32,interval:f32)->f32{
    let i=f/2u;let q=point(i)+axis(f%2u);if !valid(q){return 0.;}
    let j=at(q);
    let aperture=faces[f].coarse_momentum.z;
    if aperture*cfg.h<JAMMING_WIDTH*GRAIN_DIAMETER.y{return 0.;}
    if f%2u==1u{return -roof_breach(i,j,aperture,interval);}
    for(var k=0u;k<2u;k++){
        let donor=select(j,i,k==0u);let receiver=select(i,j,k==0u);
        let record=scratch[BODY_ROW*cfg.n+receiver];
        if select(record.w,record.z,k==0u)<=0.||record.y>=0.||column_cell(receiver){continue;}
        let c=cells[donor];if plug_deposit(c)||column_cell(donor){continue;}
        let g=grains(c);let solid=g.x+g.y;let dense=dense_volume(donor);
        if solid<=0.||solid<CRITICAL_PACKING*dense{continue;}
        let phi=min(solid/dense,CLOSE_PACKING);
        let r=cells[receiver];let gr=grains(r);let wr=max(r.volume.water,0.)+gr.x+gr.y;
        let room=max(LOOSE_PACKING*wr-gr.x-gr.y,0.);
        let up=side_face_up(point(donor),point(receiver));
        let amount=min(min(breaching_rate(g,white_powder(c),phi)*inclination_factor(up,grain_friction(g))*phi*aperture*interval,solid),room);
        return select(-amount,amount,k==0u);
    }
    return 0.;
}
// The grains (cells per substep, downward) that the packed sand in `above`,
// a full cell or a body's last layer hanging in the top of one, sheds into
// the open water in `below`: a bare underside, an overhang, the roof of a
// pocket. Nothing holds a roof's grains: the gap between the plates, 5 to 9
// grains, is wider than any arch across it. They leave at the breaching rate
// for a roof, no faster than the counterflow of water up through them
// carries (batch_flux_rate), in the measure roof_share gives; a column's
// faces are joint_mouth's. A layer in `below` with open water under it hangs
// there: it is the roof's face, which recedes cell by cell, and sheds in its
// turn. A layer that rests on sand leaves the water over it under the roof.
fn roof_breach(below:u32,above:u32,aperture:f32,interval:f32)->f32{
    if column_cell(below)||column_cell(above){return 0.;}
    let c=cells[above];if plug_deposit(c){return 0.;}
    let g=grains(c);let solid=g.x+g.y;let dense=dense_volume(above);
    if solid<=0.||solid<CRITICAL_PACKING*dense{return 0.;}
    let r=cells[below];if !open_water(r){return 0.;}
    let under=point(below)-axis(1u);
    if scratch[BODY_ROW*cfg.n+below].y>=0.&&valid(under)&&open_water(cells[at(under)]){return 0.;}
    let share=roof_share(point(below),above);if share<=0.{return 0.;}
    let phi=min(solid/dense,CLOSE_PACKING);
    let gr=grains(r);let wr=max(r.volume.water,0.)+gr.x+gr.y;
    let room=max(LOOSE_PACKING*wr-gr.x-gr.y,0.);
    let powder=white_powder(c);
    let supply=min(breaching_rate(g,powder,phi)*inclination_factor(-1.,grain_friction(g))*phi,batch_flux_rate(g,powder));
    return min(min(share*supply*aperture*interval,solid),room);
}
// The share of its breaching rate at which the roof in cell `roof` sheds
// into the open water at `start`, by where its grains end, falling straight
// down through open water. On the deposit or the tank's floor they are gone
// and the roof recedes: 1. On a bubble the raft carries them and the
// underside rests on it: 0. On sand the water is a pocket of the upper
// sand, which keeps its water whatever its roof sheds and is filled only as
// the sand over the roof follows it down: the roof sheds in the measure of
// its body's supply (SET_IN_ROW, the share of a draw on it that reaches the
// top, where the surface can sink), and the walk hands it what it shed.
// Without supply it stands. A roof that receded there would carry the
// pocket up through the body as a hole; in real sand that water is in the
// pores of the dome that has given way over it (its dilation, which the
// set-in only counts).
fn roof_share(start:vec2<i32>,roof:u32)->f32{
    var p=start;
    for(var step=0;step<i32(cfg.ny);step++){
        let below=p-axis(1u);
        if !valid(below){return 1.;}
        let cb=cells[at(below)];
        if plug_deposit(cb){return 1.;}
        if !open_water(cb){
            if cb.volume.gas*cfg.h>=GRAIN_LAYER{return 0.;}
            return geometry[SET_IN_ROW*cfg.n+roof_body(roof)].y;
        }
        p=below;
    }
    return 1.;
}
// The cell whose packing the roof in cell i belongs to: the neighbour that a
// last layer lies against (BODY_ROW), or the cell itself.
fn roof_body(i:u32)->u32{
    let entry=scratch[BODY_ROW*cfg.n+i];
    if entry.y<0.||entry.x>=1.{return i;}
    let side=u32(entry.y);let step=select(-1,1,side%2u==1u);
    return at(point(i)+step*axis(side/2u));
}
// Kozeny-Carman's hydraulic conductivity (m/s, at the in-plane gravity) of
// the mixture `g` packed to the solid fraction `phi`, from the species'
// Stokes speeds. The species combine by specific surface (the Sauter mean
// diameter), so the fine one, whose pores are the smallest, sets most of
// the resistance: a mean of the conductivities would let the coarse pores
// carry the water past the fines. The white's whole powder fills its pores.
fn packing_conductivity(g:vec2<f32>,powder:PowderShare,phi:f32)->f32{
    let solid=g.x+g.y;if solid<=0.||phi<=0.{return 0.;}
    let porosity=1.-min(phi,CLOSE_PACKING);var inverse_root=0.;
    let speeds=packing_speeds(powder)*(cfg.gravity/STANDARD_GRAVITY);
    for(var phase=0u;phase<2u;phase++){
        let weight=g[phase]/solid;if weight<=0.{continue;}
        let delta=(RHO[phase+1u]-RHO.x)/RHO.x;
        let k=speeds[phase]*(1./delta)*porosity*porosity*porosity/(10.*phi*phi);
        inverse_root+=weight/sqrt(k);
    }
    return 1./(inverse_root*inverse_root);
}
// The speed (m/s) at which a face of the mixture `g` packed to `phi`
// retreats by breaching (Van Rhee and Bezuijen 1998): its grains leave only
// as fast as water enters the layer behind them to let it dilate to the
// critical packing, K_l Delta phi / (phi - phi_c), with K_l the
// conductivity of the dilated packing and Delta the grains' relative
// density. About 1.4 mm/s at the deposit's margin of 0.025; at the critical
// packing the sand flows freely and the margin's floor of 0.01 stands in.
fn breaching_rate(g:vec2<f32>,powder:PowderShare,phi:f32)->f32{
    let solid=g.x+g.y;if solid<=0.{return 0.;}
    var relative_density=0.;
    for(var phase=0u;phase<2u;phase++){relative_density+=g[phase]/solid*(RHO[phase+1u]-RHO.x)/RHO.x;}
    return packing_conductivity(g,powder,CRITICAL_PACKING)*relative_density*phi/max(phi-CRITICAL_PACKING,.01);
}
// The factor of a face's inclination on breaching_rate. Van Rhee's breach
// recedes normal to its face at sin(alpha - phi)/sin(phi) (1 - n0)/dn
// k_l Delta (Alhaddad, Labeur and Uijttewaal 2020), alpha the face's
// inclination from the horizontal (180 for a roof) and phi the sand's
// friction angle: a slope at the angle of repose does not breach. Beside
// breaching_rate's k_l Delta phi0/(phi0 - phi_l) that leaves
// phi_l (sqrt(1 - up^2)/mu - up), `up` the upward part of the face's
// outward normal and `mu` = tan phi: 0.56 for a roof, 0.93 for a vertical
// face of the black sand (0.6).
fn inclination_factor(up:f32,mu:f32)->f32{
    let n=clamp(up,-1.,1.);
    return CRITICAL_PACKING*max(sqrt(max(1.-n*n,0.))/mu-n,0.);
}
// How packed a cell's wet part is, 0 for water and 1 from CRITICAL_PACKING
// up; a cell outside the tank reads as `inside`.
fn packedness(p:vec2<i32>,inside:f32)->f32{
    if !valid(p){return inside;}
    let c=cells[at(p)];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if wet<=0.{return 0.;}
    return clamp(solid/(CRITICAL_PACKING*wet),0.,1.);
}
// The upward part of the outward normal of the side face between the packed
// cell `donor` and the water cell `receiver` beside it, from the packedness
// of the two columns over three rows.
fn side_face_up(donor:vec2<i32>,receiver:vec2<i32>)->f32{
    let toward=f32(receiver.x-donor.x);var across=0.;var up=0.;
    for(var k=-1;k<=1;k++){
        let row=vec2<i32>(0,k);let weight=select(.25,.5,k==0);
        across+=weight*(packedness(receiver+row,packedness(receiver,0.))-packedness(donor+row,packedness(donor,0.)));
    }
    for(var column=0;column<2;column++){
        let p=select(donor,receiver,column==1);let here=packedness(p,0.);
        up+=.25*(packedness(p+axis(1u),here)-packedness(p-axis(1u),here));
    }
    let gradient=vec2<f32>(toward*across,up);let size=length(gradient);
    if size<=0.{return 0.;}
    return -gradient.y/size;
}
// The sand volume flux (m/s per unit slot width) that a column of the
// mixture `g` at the wet packing `phi` carries down a slot against the
// water it displaces: Kynch's batch flux (1952), grain-weighted over the
// species. The slab is a chamber sealed by the ceiling, the walls and the
// raft, so water rising through the same slot replaces the sand that
// leaves. A grain slips through the liquid at v_t n^HINDRANCE (Wen and Yu,
// as the material table's drag) and the liquid rises to fill its place, so
// the flux is phi v_t n^(HINDRANCE+1): small when dilute or dense, largest
// at phi = 1/(HINDRANCE+2), about a fifth.
fn batch_flux_at(g:vec2<f32>,powder:PowderShare,phi:f32)->f32{
    let solid=g.x+g.y;if solid<=0.||phi<=0.{return 0.;}
    let porosity=max(1.-phi,0.);var flux=0.;let speeds=settling_speeds(powder);
    for(var phase=0u;phase<2u;phase++){flux+=g[phase]/solid*speeds[phase];}
    return flux*phi*pow(porosity,HINDRANCE+1.);
}
// The most a counterflow column of the mixture `g` carries: the flux's peak.
fn batch_flux_rate(g:vec2<f32>,powder:PowderShare)->f32{return batch_flux_at(g,powder,1./(HINDRANCE+2.));}
// The batch flux across a face between sand at the packing `above` and sand
// at `below` (Kynch 1952; the Godunov flux of the scalar law): draining
// into a looser packing, sand crosses at the flux curve's highest value
// between the two; into a denser one, at the lower of the two fluxes (a
// shock). A dense column over open water passes the peak flux, three times
// its own at LOOSE_PACKING.
fn godunov_flux(g:vec2<f32>,powder:PowderShare,above:f32,below:f32)->f32{
    if above>=below{return batch_flux_at(g,powder,max(below,min(above,1./(HINDRANCE+2.))));}
    return min(batch_flux_at(g,powder,above),batch_flux_at(g,powder,below));
}
// The speed (m/s) at which cell i's packing retreats by breaching.
// friction_limit drops a layer's dilatancy suction while voids reach it
// slower than this: faster, the grains below it are in free fall, not an
// orifice's slow flow, and the layer keeps its dilatancy strength.
fn breaching_speed(i:u32)->f32{
    let c=cells[i];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if solid<=0.||wet<=0.{return 0.;}
    return breaching_rate(g,white_powder(c),min(solid/wet,CLOSE_PACKING));
}
// The pore pressure deficit (Pa) of cell i's skeleton dilating at the rate
// `theta` (1/s): it draws water in through its own pores. A cell of a sealed
// body has water on one side: the inflow theta h enters through that face
// under the head gradient theta h/K and crosses the cell, a deficit
// rho g theta h^2/K, with K the volume mean of the species' Kozeny-Carman
// conductivities at the cell's packing. This is the breaching law's own
// picture (the water enters through the face that sheds), so a layer
// dilates in place no faster than its face would breach.
// The deficit is effective stress: it holds the skeleton (friction, yield)
// and pushes nothing, and it keeps a dense face from unravelling at the
// settling speed. Below FREEZING_PACKING no grain is caged: no suction.
fn dilatancy_suction(i:u32,theta:f32)->f32{
    let c=cells[i];let g=grains(c);let solid=g.x+g.y;let wet=max(c.volume.water,0.)+solid;
    if theta<=0.||solid<=0.||wet<=0.{return 0.;}
    let phi=solid/wet;if phi<FREEZING_PACKING{return 0.;}
    var conductivity=0.;
    for(var phase=0u;phase<2u;phase++){
        let weight=g[phase]/solid;let delta=(RHO[phase+1u]-RHO.x)/RHO.x;
        conductivity+=weight*packing_speeds(white_powder(c))[phase]*(1./delta)*pow(1.-min(phi,CLOSE_PACKING),3.)/(10.*phi*phi);
    }
    // rho_w g / K is mu / k whatever the gravity: both at the standard one.
    return RHO.x*STANDARD_GRAVITY*theta*cfg.h*cfg.h/max(conductivity,1e-9);
}
