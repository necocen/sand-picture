// The transport groups' capacities, in the mechanics' scratch once it has
// run: each owner's outgoing ratio per phase (`next`) and its receiving
// ratio (`compression`, volume.x), from material_capacity to
// limit_layer_flux.
fn outgoing_capacity(i:u32)->vec4<f32> {return next[i];}
fn receiving_capacity(i:u32)->f32 {return compression[i].volume.x;}
fn put_outgoing_capacity(i:u32,ratios:vec4<f32>) {next[i]=ratios;}
fn put_receiving_capacity(i:u32,ratio:f32) {compression[i].volume.x=ratio;}
