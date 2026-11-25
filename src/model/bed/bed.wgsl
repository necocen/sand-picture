// The deposit as a bed connected to the floor: the cells of each column that
// belong to it, the release of bed that lost its support, and the grains'
// friction by species. A cell's `bed` is (fine, coarse, the volume the bed
// occupies, its pore water), in cell volumes. The columns are found at the
// start of an update and again before the capture.

// Per column. [x]: the bed's height (cells), its top row and the reaction
// that stopped the captured sand (Pa; x, y). [nx+x]: the surface flow of the
// pair (x, x+1): its depth (m) and the flux it moved (m2/s, toward x+1).
// [2nx+x]: what the capture saw: y the solid taken, w its arriving speed.
@group(0) @binding(10) var<storage,read_write> terrain:array<vec4<f32>>;
// The grains' internal friction at slow shear (fine, coarse): the white's
// glass beads and flour slide as Boyer's spheres, the black's angular
// chromite as natural sand (tan 31 degrees).
const SPECIES_FRICTION=vec2<f32>(.32,.6);
// A mixture shears at its species' volume mean: in a dense flow the grains
// of both are in contact and move together.
fn grain_friction(g:vec2<f32>)->f32{
    let solid=g.x+g.y;if solid<=0.{return SPECIES_FRICTION.y;}
    return dot(max(g,vec2<f32>(0.)),SPECIES_FRICTION)/solid;
}
const ROUND_GEOMETRY:f32=8.*1.1920929e-7;
// One coarse grain (m): a phase too little to make a layer this thick across
// a cell is a trace among the others and does not act as a layer.
const GRAIN_LAYER:f32=GRAIN_DIAMETER.y;
// The lowest row of column x's bed, or ny for a lifted column. Gas on the
// floor under grains in contact is trapped, however tall the pocket (the
// pores' capillary entry pressure keeps it out of the packing): it carries
// the sand above as the floor would, and the column starts on what it
// carries (tested first: started under a buried bubble, the column would
// end in its gas). Else the column starts on the floor where the floor
// cell's wet part reaches the floor line, a bubble under water or a
// suspension only sitting in its cell. Gas covering that line lifts it.
fn column_base(x:u32)->u32{
    let buried=pocket_cover(x,0u,false);if buried<cfg.ny{return buried;}
    if reaches_floor(x){return 0u;}
    return cfg.ny;
}
// From row `start` up: the row where the sand that a pocket carries begins,
// or ny when there is no buried pocket. `open`: the pocket has begun under
// `start` (a bubble in the free part of a part-filled bed cell); else it
// must begin at `start`. A bubble trapped on the deposit carries the grains
// in contact above it as one trapped on the floor does.
fn pocket_cover(x:u32,start:u32,open:bool)->u32{
    var pocket=open;
    for(var y=start;y<cfg.ny;y++){
        let c=cells[y*cfg.nx+x];
        // A pocket's row is the pocket itself, with the sand around the
        // bubble: what the pocket holds up lies above it. Only its top row
        // may hold a cap the bubble carries, sand caged in a wet layer over
        // the gas, and the column starts in that row.
        let caged=c.volume.fine+c.volume.coarse+c.bed.x+c.bed.y>=FREEZING_PACKING*liquid(c);
        if pocket_row(c){
            if caged&&liquid(c)*cfg.h>=GRAIN_LAYER&&(y+1u>=cfg.ny||!pocket_row(cells[(y+1u)*cfg.nx+x])){return y;}
            pocket=true;continue;
        }
        if !pocket{return cfg.ny;}
        if touching(c){return y;}
        // The sand lining the bubble's top is a packing even loosened below
        // contact: caged, it rests on the pocket. A suspension raining onto
        // a bubble is not caged, and the bubble is not buried yet.
        if caged{return y;}
        return cfg.ny;
    }
    return cfg.ny;
}
// A row of a pocket: gas of a grain layer or more, or no wet layer at all.
fn pocket_row(c:Cell)->bool{
    return c.volume.gas*cfg.h>=GRAIN_LAYER||liquid(c)*cfg.h<GRAIN_LAYER;
}
// Grains in contact, mobile or stored, in the cell's wet part.
fn touching(c:Cell)->bool{
    return c.volume.fine+c.volume.coarse+c.bed.x+c.bed.y>=LOOSE_PACKING*liquid(c);
}
// Whether the wet part of column x's floor cell reaches the floor line: a
// trace of gas leaves it wet; else the gas planes must leave a span of it.
fn reaches_floor(x:u32)->bool{
    if liquid(cells[x])<=0.{return false;}
    if cells[x].volume.gas*cfg.h<GRAIN_LAYER{return true;}
    let gas=outer_gas_shapes(x);var span=vec2<f32>(-.5,.5);
    for(var k=0u;k<2u;k++){
        if gas.present[k]==0u{continue;}
        let shape=gas.shapes[k];let rhs=shape.z+.5*shape.y;
        if shape.x>0.{span.x=max(span.x,rhs/shape.x);}
        else if shape.x<0.{span.y=min(span.y,rhs/shape.x);}
        else if rhs>0.{return false;}
    }
    return span.y>span.x;
}
fn bed_height(x:u32)->f32{return bed_column(x).height;}
fn bed_top(x:u32)->u32{return bed_column(x).top;}
fn bed_open(x:u32)->u32{return bed_column(x).open;}
@compute @workgroup_size(128) fn bed_geometry(@builtin(global_invocation_id) id:vec3<u32>){
    let x=id.x;if x>=cfg.nx{return;}let column=bed_column(x);
    terrain[x]=vec4<f32>(column.height,f32(column.top),0.,0.);
}
fn connected_height(x:i32)->f32{return terrain[u32(clamp(x,0,i32(cfg.nx)-1))].x;}
// The unit normal of the bed's surface at column x, out of the bed.
fn surface_normal(x:i32)->vec2<f32>{
    let l=connected_height(max(0,x-1));let r=connected_height(min(i32(cfg.nx)-1,x+1));
    let width=select(2.,1.,x==0||x==i32(cfg.nx)-1);
    return normalize(vec2<f32>(-(r-l)/width,1.));
}
// Bed without connection to the floor returns to the mobile inventory: a
// lifted column's, what lies above a column's top row, and any steep vertical
// cliff face (> 3 cells above an adjacent column's bed) directly exposed to
// open water (e.g. right after a 90-degree rotation).
fn cliff_exposed(x:i32,y:i32)->bool{
    for(var dx=-1;dx<=1;dx+=2){
        let nx_i=x+dx;
        if nx_i<0||nx_i>=i32(cfg.nx){continue;}
        if f32(y)<=terrain[u32(nx_i)].x+3.{continue;}
        let c=cells[u32(y)*cfg.nx+u32(nx_i)];
        let solid=max(c.volume.fine,0.)+max(c.volume.coarse,0.)+max(c.bed.x,0.)+max(c.bed.y,0.);
        if c.volume.gas<.25&&solid<.25{return true;}
    }
    return false;
}
@compute @workgroup_size(128) fn bed_release(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=cfg.n{return;}let p=point(i);
    if terrain[u32(p.x)].x<=0.||f32(p.y)>terrain[u32(p.x)].y||cliff_exposed(p.x,p.y){release_bed(i);}
}
// `height`: the bed's surface in cells from the floor; `top`: its highest
// row; `open`: the row where it ends, whose free part takes the next sand
// (the top row while that is part-filled, else the row above).
struct BedColumn {height:f32,top:u32,open:u32,connected:bool}
// Whether the sand of cell (x, y) rests on a pocket: the row under it is a
// pocket's, or it lies around a bubble in a floor cell. The pocket or the
// floor carries it, so the bubble's buoyancy in its cell is no lift.
fn on_pocket(x:u32,y:u32)->bool{
    if y==0u{return cells[x].volume.gas*cfg.h>=GRAIN_LAYER;}
    return pocket_row(cells[(y-1u)*cfg.nx+x]);
}
// Free wet volume too little for a layer one grain thick, under a layer of
// bed, is what rounding leaves (the capture fills a cell before the one
// above it): it does not end the column.
fn filled(c:Cell,above:Cell)->bool{
    return free_wet(c)*cfg.h<GRAIN_LAYER&&above.bed.z*cfg.h>=GRAIN_LAYER;
}
// A bed cell above whose free wet part makes no layer one grain thick shuts
// this cell's free water and gas in: they cannot leave through the packed
// sand, and the bed above rests on them as on the floor, so a pocket buried
// in the deposit does not cut its column. A trace of bed in a cell of gas
// shuts nothing in.
fn enclosed(x:u32,y:u32)->bool{
    if y+1u>=cfg.ny{return false;}
    let above=cells[(y+1u)*cfg.nx+x];
    return above.bed.z>0.&&liquid(above)*cfg.h>=GRAIN_LAYER&&free_wet(above)*cfg.h<GRAIN_LAYER;
}
fn bed_column(x:u32)->BedColumn{
    var result:BedColumn;let base=column_base(x);
    // A lifted column holds no bed: what slides in from its neighbours comes
    // to rest on the bubble's top and buries it there (surface_path).
    if base>=cfg.ny{return result;}
    result.connected=true;result.open=base;result.height=f32(base);
    for(var y=base;y<cfg.ny;y++){
        let c=cells[y*cfg.nx+x];let wet=liquid(c);
        result.open=y;let closed=enclosed(x,y);
        // Where the column ends, a pocket that grains in contact bury carries
        // them: the column goes on from the sand it carries. The pocket
        // starts in this row, or over a part-filled bed cell in its free
        // part or the row above; water under it stays in the deposit.
        if (c.bed.z<=0.||wet<=0.)&&!closed{
            let cover=pocket_cover(x,y,false);
            if cover>y&&cover<cfg.ny{y=cover-1u;continue;}
            result.height=f32(y);return result;
        }
        result.top=y;let fraction=occupied_wet_fraction(c);
        result.height=f32(y)+fraction;
        if fraction<1.-ROUND_GEOMETRY&&!closed&&(y+1u>=cfg.ny||!filled(c,cells[(y+1u)*cfg.nx+x])){
            let cover=pocket_cover(x,y+1u,pocket_row(c));
            if cover>y&&cover<cfg.ny{y=cover-1u;continue;}
            return result;
        }
        result.height=f32(y+1u);
        if y+1u>=cfg.ny{return result;}
        // A full cell carries the column on, whatever its gas: the connection
        // is one of support, not a way a grain's centre could travel.
    }return result;
}

// Returns cell i's bed, grains and pore water, to the mobile inventory with
// no momentum added. The white's powder stays bound to its grit.
fn release_bed(i:u32){
    let c=cells[i];put_mobile_volumes(i,mobile_volumes(c)+vec4<f32>(c.bed.w,c.bed.xy,0.));
    cells[i].bed=vec4<f32>(0.);cells[i].powder=CellPowder(c.powder.mobile+c.powder.bed,0.,c.powder.free);
}
