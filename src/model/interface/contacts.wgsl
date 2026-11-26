// Gas obstacle shapes and curved/planar face passage clearance, reading the
// cell-local gas planes rasterized by `foam_raster` (`SHAPE_ROW`, `CONTACT_ROW`).

// Whether the half-plane n.x x + n.y y <= alpha covers some area of the unit
// cell: a line or a point is no obstacle (its gas stays in the inventory).
fn represented_gas_area(shape:vec4<f32>)->bool {
    let width=abs(shape.x)+abs(shape.y);
    return width>0.&&shape.z>-.5*width;
}

// On each side the half-plane of the largest intercept contains the others,
// so the cell's gas, as an obstacle, is at most two clipped polygons.
struct OuterGas { shapes:array<vec4<f32>,2>, present:vec2<u32> }
fn outer_gas_shapes(i:u32)->OuterGas {
    var result:OuterGas;
    for(var l=0u;l<GAS_SLOTS;l++){
        if geometry[layer_index(i,l)].x<=0.{continue;}
        let shape=geometry[shape_index(i,l)];if !represented_gas_area(shape){continue;}var side=0u;
        if result.present.x>0u&&dot(result.shapes[0].xy,shape.xy)<0.{side=1u;}
        if result.present[side]==0u||shape.z>result.shapes[side].z{
            result.shapes[side]=shape;result.present[side]=1u;
        }
    }return result;
}

// A cell's two contact shapes, a plane (n, alpha, 0) or a disk (centre,
// radius, 1), written by `foam_raster` at the end of an update for the next one.
fn contact_index(i:u32,l:u32)->u32{return CONTACT_ROW*cfg.n+2u*i+l;}

fn vertex_span(v:vec2<f32>,a:u32,side:f32,row:f32,r:f32)->vec2<f32> {
    let distance=side-v[a];if abs(distance)>r{return vec2<f32>(1.e10,-1.e10);}
    let extent=sqrt(max(0.,r*r-distance*distance));return vec2<f32>(v[1u-a]+row-extent,v[1u-a]+row+extent);
}
fn span_union(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return vec2<f32>(min(a.x,b.x),max(a.y,b.y));}
// The span of a face closed to a grain's centre by (disk d within its cell)
// expanded by the grain's radius r: offset arc points and expanded vertices.
fn disk_exclusion(d:vec4<f32>,a:u32,side:f32,row:f32,r:f32)->vec2<f32> {
    var span=vec2<f32>(1.e10,-1.e10);let radius=d.z;
    let corners=array<vec2<f32>,4>(vec2<f32>(-.5,-.5),vec2<f32>(.5,-.5),vec2<f32>(.5,.5),vec2<f32>(-.5,.5));
    for(var k=0u;k<4u;k++){
        let v=corners[k];if dot(v-d.xy,v-d.xy)<=radius*radius{span=span_union(span,vertex_span(v,a,side,row,r));}
    }
    for(var b=0u;b<2u;b++){for(var sign=-1;sign<=1;sign+=2){
        let edge=.5*f32(sign);let distance=edge-d[b];if abs(distance)>radius{continue;}
        let extent=sqrt(max(0.,radius*radius-distance*distance));
        for(var t=-1;t<=1;t+=2){
            let tangent=d[1u-b]+f32(t)*extent;if tangent<-.5||tangent>.5{continue;}
            var v=vec2<f32>(0.);v[b]=edge;v[1u-b]=tangent;
            span=span_union(span,vertex_span(v,a,side,row,r));
        }
    }}
    let expanded=radius+r;let distance=side-d[a];
    if abs(distance)<=expanded{
        let extent=sqrt(max(0.,expanded*expanded-distance*distance));
        for(var t=-1;t<=1;t+=2){
            var p=vec2<f32>(0.);p[a]=side;p[1u-a]=d[1u-a]+f32(t)*extent;
            let nearest=p-(r/expanded)*(p-d.xy);
            if all(nearest>=vec2<f32>(-.5))&&all(nearest<=vec2<f32>(.5)){
                span=span_union(span,vec2<f32>(p[1u-a]+row));
            }
        }
    }
    return span;
}
struct Passage { largest:vec2<f32>, total:f32 }
// The part of face f free for the centre of a grain of radius r (cells; 0
// for the liquid), beside the contact shapes of the cells around it and the
// tank's wall. Curved gas can cut a face's middle: every free piece counts.
fn curved_passage(f:u32,r:f32)->Passage {
    let i=f/2u;let a=f%2u;let p=point(i);let q=p+axis(a);
    if !valid(q){return Passage(vec2<f32>(.5,-.5),0.);}
    var spans:array<vec2<f32>,12>;var count=0u;
    for(var row=-1;row<=1;row++){
        if r==0.&&row!=0{continue;}
        for(var column=0;column<2;column++){
            let cell=p+column*axis(a)+row*axis(1u-a);let side=.5-f32(column);
            if !valid(cell){
                if row<0{spans[count]=vec2<f32>(-.5,-.5+r);}else{spans[count]=vec2<f32>(.5-r,.5);}count++;continue;
            }
            let j=at(cell);if cells[j].volume.gas<=0.{continue;}
            for(var s=0u;s<2u;s++){
                let shape=geometry[contact_index(j,s)];var span=vec2<f32>(1.e10,-1.e10);
                if shape.w>0.{span=disk_exclusion(shape,a,side,f32(row),r);}
                else if represented_gas_area(shape){span=polygon_exclusion(shape,a,side,f32(row),r);}
                span=vec2<f32>(max(-.5,span.x),min(.5,span.y));
                if span.x<span.y{spans[count]=span;count++;}
            }
        }
    }
    for(var j=1u;j<count;j++){
        let v=spans[j];var k=j;
        while k>0u{if spans[k-1u].x<=v.x{break;}spans[k]=spans[k-1u];k--;}
        spans[k]=v;
    }
    var out=Passage(vec2<f32>(-.5,-.5),0.);var cursor=-.5;
    for(var j=0u;j<count;j++){
        let span=spans[j];let width=max(0.,span.x-cursor);out.total+=width;
        if width>out.largest.y-out.largest.x{out.largest=vec2<f32>(cursor,span.x);}
        cursor=max(cursor,span.y);
    }
    let width=max(0.,.5-cursor);out.total+=width;
    if width>out.largest.y-out.largest.x{out.largest=vec2<f32>(cursor,.5);}
    return out;
}
