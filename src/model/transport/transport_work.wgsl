// The transport groups' capacities: each owner's outgoing ratio per phase
// (OUTGOING_CAPACITY_ROW) and its receiving ratio (`compression`, volume.x,
// in the mechanics' scratch once it has run), from material_capacity to
// limit_layer_flux.
fn outgoing_capacity(i:u32)->vec4<f32> {return scratch[OUTGOING_CAPACITY_ROW*cfg.n+i];}
fn receiving_capacity(i:u32)->f32 {return compression[i].volume.x;}
fn put_outgoing_capacity(i:u32,ratios:vec4<f32>) {scratch[OUTGOING_CAPACITY_ROW*cfg.n+i]=ratios;}
fn put_receiving_capacity(i:u32,ratio:f32) {compression[i].volume.x=ratio;}
