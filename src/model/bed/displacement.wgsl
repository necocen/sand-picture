// The record of the moves the bed's remaps have already made (the capture
// and the surface flow), per face in volume per face area and time (m/s):
// xy the bed's grains (fine, coarse), zw the mobile grains. Water carries
// minus their sum; a remap never moves gas. No rule reads the record, and a
// later transport must not add it to its fluxes.

// Twice an update, before the bed's release and before the capture: the
// record so far becomes the previous one and a new one begins.
@compute @workgroup_size(128) fn begin_displacement(@builtin(global_invocation_id) id:vec3<u32>){
    let f=id.x;if f>=2u*cfg.n{return;}
    faces[f].previous_displacement=faces[f].displacement;
    faces[f].displacement=vec4<f32>(0.);
}
// Records a surface packet (its bed grains `bed` and the mobile grains
// `mobile` that go with them, negative as they return; cell volumes) on the
// faces of its way: along column xa from row ya to row `gate`, across to
// column xb there, and along xb to row yb (surface_path).
fn record_surface_path(xa:u32,ya:u32,xb:u32,yb:u32,gate:u32,bed:vec2<f32>,mobile:vec2<f32>){
    let q=vec4<f32>(bed,mobile)*cfg.h/cfg.dt;
    faces[2u*(gate*cfg.nx+min(xa,xb))].displacement+=select(-q,q,xb>xa);
    for(var row=min(ya,gate);row<max(ya,gate);row++){
        faces[2u*(row*cfg.nx+xa)+1u].displacement+=select(-q,q,gate>ya);
    }
    for(var row=min(yb,gate);row<max(yb,gate);row++){
        faces[2u*(row*cfg.nx+xb)+1u].displacement+=select(-q,q,yb>gate);
    }
}
