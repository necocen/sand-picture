//! Gas shape, contact fit and wet aperture reconstruction: the last stages of
//! an update, read by the next update's mechanics and transport.
use super::buffers::Buffers;
use super::kernel::Plan;
use super::mechanics::common;
use super::shader::Program;
use super::update::Plans;

pub struct Interface {
    pub planes: usize,
    /// Reset the fit list; write planar cells and list disk candidates; write
    /// the indirect arguments; fit each listed cell with one workgroup.
    contacts: [usize; 4],
    pub apertures: usize,
}
impl Interface {
    pub fn new(plans: &mut Plans, buffers: &Buffers) -> Self {
        let n = buffers.n;
        let groups = n.div_ceil(128);
        let edges = (2 * n).div_ceil(128);
        let listed = [
            (0, &buffers.config),
            (4, &buffers.geometry),
            (13, &buffers.contact_list),
        ];
        Self {
            planes: plans.add(common(buffers, "reconstruct_interface", n.div_ceil(64))),
            contacts: [
                plans.add(Plan::new(
                    Program::Model,
                    "reset_contact_work",
                    &[(13, &buffers.contact_list)],
                    1,
                )),
                plans.add(Plan::new(Program::Model, "plan_contacts", &listed, groups)),
                plans.add(Plan::new(
                    Program::Model,
                    "seal_contact_work",
                    &[(13, &buffers.contact_list), (14, &buffers.contact_groups)],
                    1,
                )),
                plans.add(
                    Plan::new(Program::Model, "fit_contacts", &listed, 0)
                        .indirect(&buffers.contact_groups),
                ),
            ],
            apertures: plans.add(common(buffers, "wet_apertures", edges)),
        }
    }

    pub fn contacts(&self) -> Vec<usize> {
        self.contacts.to_vec()
    }

    /// The whole reconstruction: at the start and after a turn.
    pub fn kernels(&self) -> Vec<usize> {
        [vec![self.planes], self.contacts(), vec![self.apertures]].concat()
    }
}
