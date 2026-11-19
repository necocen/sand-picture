// Where the bed lies inside its cells and what it leaves open on their
// faces. The gas planes cut a convex wet polygon out of a cell and the bed
// fills its lowest part (prepare_bed_occupancy); a face is closed over the
// bed's traces on its two sides (prepare_bed_passage). Both are written at
// the start of an update and again after the capture and the surface flow.

struct WetPolygon {points:array<vec2<f32>,8>,count:u32}
// The part of polygon p where dot(n, point) >= alpha.
fn clip_wet(p:WetPolygon,n:vec2<f32>,alpha:f32)->WetPolygon{
    var out:WetPolygon;
    for(var k=0u;k<p.count;k++){
        let a=p.points[k];let b=p.points[(k+1u)%p.count];
        let da=dot(n,a)-alpha;let db=dot(n,b)-alpha;
        if da>=0.{out.points[out.count]=a;out.count++;}
        if (da>=0.)!=(db>=0.){out.points[out.count]=a+(b-a)*(da/(da-db));out.count++;}
    }return out;
}
fn wet_polygon(i:u32)->WetPolygon{
    var p:WetPolygon;p.count=4u;
    p.points[0]=vec2<f32>(-.5,-.5);p.points[1]=vec2<f32>(.5,-.5);
    p.points[2]=vec2<f32>(.5,.5);p.points[3]=vec2<f32>(-.5,.5);
    let gas=outer_gas_shapes(i);
    /*loop: slower written out*/for(var k=0u;k<2u;k++){
        if gas.present[k]>0u{p=clip_wet(p,gas.shapes[k].xy,gas.shapes[k].z);}
    }return p;
}
fn polygon_area(p:WetPolygon)->f32{
    if p.count<3u{return 0.;}var result=0.;let origin=p.points[0];
    for(var k=1u;k+1u<p.count;k++){
        let a=p.points[k]-origin;let b=p.points[k+1u]-origin;result+=a.x*b.y-a.y*b.x;
    }return .5*abs(result);
}
// Per cell, BED_OCCUPANCY_ROW: x the height of the bed's surface in the cell
// (0 to 1), y the bed's packing, z the wet polygon's area where gas cuts the
// cell, w the bed's share of the wet part. The surface is the level under
// which the polygon holds that share. Polygon and stored volumes differ by
// rounding: only the volumes' ratio is used; no inventory is edited to match.
@compute @workgroup_size(128) fn prepare_bed_occupancy(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let c=cells[i];var out=vec4<f32>(0.);
    let wet=liquid(c);
    if c.bed.z>0.&&wet>0.{
        let fraction=occupied_wet_fraction(c);out.w=fraction;
        if fraction>=1.{out.x=1.;}
        else if c.volume.gas<=0.{out.x=fraction;}
        else{
            let p=wet_polygon(i);let area=polygon_area(p);let bed_area=fraction*area;
            out.z=area;var lower=0.;var upper=1.;
            if area>0.{
                /*loop: slower written out*/for(var iteration=0u;iteration<24u;iteration++){
                    let middle=.5*(lower+upper);
                    let amount=polygon_area(clip_wet(p,vec2<f32>(0.,-1.),.5-middle));
                    if amount<bed_area{lower=middle;}else{upper=middle;}
                }
                out.x=.5*(lower+upper);
            }
        }
        out.y=(c.bed.x+c.bed.y)/c.bed.z;
    }
    scratch[BED_OCCUPANCY_ROW*cfg.n+i]=out;
}
// The interval that cell i's bed covers of the cell's edge on `side` (-0.5
// or 0.5) of axis a, in cell coordinates along the edge, or zero: a side
// edge up to the bed's surface, the bottom edge whole, the top edge only
// under a full cell, each less what the gas planes cut off.
fn bed_edge_interval(i:u32,a:u32,side:f32)->vec2<f32>{
    let bed=scratch[BED_OCCUPANCY_ROW*cfg.n+i];if bed.x<=0.{return vec2<f32>(0.);}
    var span=vec2<f32>(-.5,.5);
    if a==0u{span.y=min(span.y,bed.x-.5);}
    else if side>0.&&bed.x<1.{return vec2<f32>(0.);}
    let gas=outer_gas_shapes(i);
    /*loop: slower written out*/for(var k=0u;k<2u;k++){
        if gas.present[k]==0u{continue;}
        let shape=gas.shapes[k];let rhs=shape.z-shape[a]*side;let tangent=shape[1u-a];
        if tangent>0.{span.x=max(span.x,rhs/tangent);}
        else if tangent<0.{span.y=min(span.y,rhs/tangent);}
        else if rhs>0.{return vec2<f32>(0.);}
    }
    if span.y<=span.x{return vec2<f32>(0.);}return span;
}
// Per face, BED_PASSAGE_ROW: x the share of the face free of bed, y the
// share of the bed's pores (no way for the flow), z the share the bed
// covers, w the share of its solid.
@compute @workgroup_size(128) fn prepare_bed_passage(@builtin(global_invocation_id) id:vec3<u32>){
    let f=id.x;if f>=2u*cfg.n{return;}let i=f/2u;let a=f%2u;let q=point(i)+axis(a);
    var out=vec4<f32>(1.,0.,0.,0.);
    if valid(q){
        let j=at(q);let l=bed_edge_interval(i,a,.5);let r=bed_edge_interval(j,a,-.5);
        let left=max(0.,l.y-l.x);let right=max(0.,r.y-r.x);
        let overlap=max(0.,min(l.y,r.y)-max(l.x,r.x));
        let occupied=clamp(left+right-overlap,0.,1.);
        let lp=scratch[BED_OCCUPANCY_ROW*cfg.n+i].y;let rp=scratch[BED_OCCUPANCY_ROW*cfg.n+j].y;
        // Where the two cells' traces overlap the denser one sets the solid;
        // apart, their solids add.
        let solid=lp*left+rp*right-min(lp,rp)*overlap;
        out=vec4<f32>(1.-occupied,max(0.,occupied-solid),occupied,solid);
    }
    scratch[BED_PASSAGE_ROW*cfg.n+f]=out;
}
