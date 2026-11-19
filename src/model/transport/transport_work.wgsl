// The transport groups' capacities: each owner's outgoing ratio per phase
// and its receiving ratio, from material_capacity to limit_layer_flux.
fn outgoing_capacity(i:u32)->vec4<f32> {return scratch[OUTGOING_CAPACITY_ROW*cfg.n+i];}
fn receiving_capacity(i:u32)->f32 {return scratch[RECEIVING_CAPACITY_ROW*cfg.n+i].x;}
fn put_outgoing_capacity(i:u32,ratios:vec4<f32>) {scratch[OUTGOING_CAPACITY_ROW*cfg.n+i]=ratios;}
fn put_receiving_capacity(i:u32,ratio:f32) {scratch[RECEIVING_CAPACITY_ROW*cfg.n+i].x=ratio;}
