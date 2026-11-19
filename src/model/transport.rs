//! Grouping, bounded adaptive substeps and conservative phase/gas/momentum
//! transport, after the second flow solve. The schedule of its subdivisions
//! is chosen on the GPU without readback: before the carry the faces' flow is
//! reduced to the subdivisions this update needs (at most the grid's budget),
//! and every slot of the budget is encoded; the kernels of an unused slot end
//! at once.
use super::buffers::Buffers;
use super::grid::subdivisions;
use super::kernel::Plan;
use super::shader::Program;
use super::update::Plans;

pub struct Transport {
    groups: [usize; 4],
    carry: [usize; 6],
    common_reconstruct: usize,
    trace: usize,
    /// Before the carry: the powder freed where water runs through touching
    /// sand faster than its grains settle (free_powder).
    release: usize,
    /// Before the carry: each cell's grains and body membership (plug_record).
    record: usize,
    /// After the carry: the kinematic descent of packed bodies (plug_descent).
    plug: usize,
    /// The schedule: reset, reduce and finish before the carry, advance after
    /// each slot.
    schedule: [usize; 3],
    advance: usize,
}
impl Transport {
    pub fn new(plans: &mut Plans, buffers: &Buffers) -> Self {
        let n = buffers.n;
        let groups = n.div_ceil(128);
        let edges = (2 * n).div_ceil(128);
        let common = |entry, groups| {
            Plan::new(
                Program::Model,
                entry,
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (2, &buffers.next),
                    (3, &buffers.faces),
                    (4, &buffers.geometry),
                    (6, &buffers.compression),
                    (12, &buffers.members),
                    (8, &buffers.transport),
                    (9, &buffers.schedule),
                ],
                groups,
            )
        };
        let scheduled = |entry, groups| {
            Plan::new(
                Program::Schedule,
                entry,
                &[
                    (0, &buffers.config),
                    (1, &buffers.faces),
                    (2, &buffers.schedule),
                    (3, &buffers.cells),
                    (4, &buffers.geometry),
                    (5, &buffers.members),
                ],
                groups,
            )
        };
        Self {
            groups: [
                plans.add(common("request_groups", groups)),
                plans.add(common("choose_groups", groups)),
                plans.add(common("resolve_groups", groups)),
                plans.add(common("accept_groups", groups)),
            ],
            carry: [
                plans.add(Plan::new(
                    Program::Model,
                    "layer_flux",
                    &[
                        (0, &buffers.config),
                        (1, &buffers.cells),
                        (3, &buffers.faces),
                        (4, &buffers.geometry),
                        (8, &buffers.transport),
                        (9, &buffers.schedule),
                        (16, &buffers.labels),
                    ],
                    edges,
                )),
                plans.add(common("gas_capacity", groups)),
                plans.add(common("compose_layer_flux", edges)),
                plans.add(common("material_capacity", groups)),
                plans.add(common("limit_layer_flux", edges)),
                plans.add(common("advect_groups", groups)),
            ],
            trace: plans.add(Plan::new(
                Program::Model,
                "clear_trace_gas",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (4, &buffers.geometry),
                    (16, &buffers.labels),
                ],
                groups,
            )),
            release: plans.add(Plan::new(
                Program::Model,
                "free_powder",
                &[(0, &buffers.config), (1, &buffers.cells)],
                groups,
            )),
            record: plans.add(Plan::new(
                Program::Model,
                "plug_record",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (4, &buffers.geometry),
                ],
                groups,
            )),
            plug: plans.add(Plan::new(
                Program::Model,
                "plug_descent",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (4, &buffers.geometry),
                ],
                1,
            )),
            common_reconstruct: plans.add(Plan::new(
                Program::Model,
                "reconstruct_common",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (4, &buffers.geometry),
                    (12, &buffers.members),
                    (9, &buffers.schedule),
                ],
                n.div_ceil(64),
            )),
            schedule: [
                plans.add(scheduled("reset", 1)),
                plans.add(scheduled("reduce", n.div_ceil(128))),
                plans.add(scheduled("finish", 1)),
            ],
            advance: plans.add(scheduled("advance", 1)),
        }
    }

    /// Free the powder, record the bodies, fix the transport groups and choose
    /// this update's subdivisions.
    pub fn prepare(&self) -> Vec<usize> {
        [
            vec![self.release, self.record],
            self.groups.to_vec(),
            self.schedule.to_vec(),
        ]
        .concat()
    }

    /// Every subdivision slot of the grid's budget.
    pub fn carry(&self) -> Vec<usize> {
        let mut kernels = Vec::new();
        for _ in 0..subdivisions() {
            kernels.push(self.common_reconstruct);
            kernels.extend(self.carry);
            kernels.push(self.advance);
        }
        kernels
    }

    /// Once per update: return trace gas identities to the liquid, then let
    /// the voids the carry left in packed bodies rise (the plug descent).
    pub fn clear_trace(&self) -> Vec<usize> {
        vec![self.trace, self.plug]
    }
}
