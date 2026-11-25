// The carry's common terms (stage TransportCarry). A face carries one common
// volume, and each material crosses with a blend of its low flux, the
// upwind group's share of that volume, and its high one (the gas planes'
// sweep, the reconstructed grains): one blend for every material of a face.

// A subdivision carries dt over the steps run, or over the steps the
// Courant number requires where they exceed the budget: every flux then
// shrinks alike, and no donor gives more than it holds.
fn carry_divisions()->f32{return f32(max(carry_plan.steps,carry_plan.required));}
// The common volume (cells) that crosses face f in a subdivision, positive
// towards +x or +y; none inside a group.
fn common_carrier(f:u32)->f32{
    if internal_face(f){return 0.;}return faces[f].volume.w*cfg.dt/(carry_divisions()*cfg.h);
}
// The part of an increase over the low flux that the stock `amount` pays.
// Budgets bound only increases: the low fluxes keep every stock nonnegative
// while a group's outgoing Courant numbers sum to one at most.
fn capacity_ratio(amount:f32,increase:f32)->f32 {
    if increase<=0.{return 1.;}return min(1.,max(amount,0.)/increase);
}

// The subdivision's first kernel: the cells' gas planes, two lanes to a
// cell (paired_reconstruct). The second lane, free after the shared
// normals, gathers the cell's transport group; neither reads what the other
// writes. A slot past the schedule's steps ends at the barrier, as in every
// carry kernel: the dispatch is direct, since wgpu validates each indirect
// one in a pass of its own.
@compute @workgroup_size(128) fn reconstruct_common(@builtin(global_invocation_id) id:vec3<u32>) {
    if carry_plan.current>=carry_plan.steps{return;}
    let i=id.x;
    if i<cfg.n&&group_owner(i)==i{cache_group(i);}
}

// The first words of the schedule's Plan (carry_schedule.wgsl).
struct CarryPlan {maximum:u32,steps:u32,current:u32,required:u32,sum_steps:u32,updates:u32,max_required:u32,padding:u32}
@group(0) @binding(9) var<storage,read_write> carry_plan:CarryPlan;
