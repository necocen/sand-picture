//! Predicted flow and grain/contact response, before and after each of an
//! update's two flow solves. The flow solve itself is `stream`.
use super::grid::{CELLS, pressure_pairs};
use super::kernel::Plan;
use super::shader::Program;
use super::update::Plans;

pub struct Mechanics {
    pub momentum: usize,
    pub predict: usize,
    pub grain_stress: usize,
    pub grain_velocity: usize,
    pub grain: usize,
    /// Red and black sweeps of the grain-pressure relaxation, and the last pair.
    pressure: [usize; 2],
    pressure_last: [usize; 2],
    pub transfer_contact: usize,
    pub cache_contact: usize,
    contact_velocity: usize,
    contact_grain: usize,
    pub copy_stress: usize,
    pub project_faces: usize,
    pub prepare_stream: usize,
    pub hold_friction: usize,
}
impl Mechanics {
    pub fn new(plans: &mut Plans) -> Self {
        let [nx, ny] = CELLS.map(|v| v as u32);
        let n = nx * ny;
        let groups = n.div_ceil(128);
        // A red-black sweep serves only its colour's cells, half of each row.
        let colour = (ny * nx.div_ceil(2)).div_ceil(128);
        let edges = (2 * n).div_ceil(128);
        let mut build = |entry, groups| plans.add(Plan::new(Program::Model, entry, groups));
        Self {
            momentum: build("apply_momentum", groups),
            predict: build("predict", edges),
            grain_stress: build("prepare_grain_stress", groups),
            grain_velocity: build("prepare_grain_velocity", groups),
            grain: build("grain_constraint", groups),
            pressure: [
                build("pressure_red", colour),
                build("pressure_black", colour),
            ],
            pressure_last: [
                build("pressure_red_last", colour),
                build("pressure_black_last", colour),
            ],
            transfer_contact: build("transfer_contact", groups),
            cache_contact: build("cache_contact", groups),
            contact_velocity: build("prepare_contact_velocity", groups),
            contact_grain: build("contact_constraint", groups),
            copy_stress: build("copy_stress", groups),
            project_faces: build("project_faces", edges),
            prepare_stream: build("prepare_stream", edges),
            hold_friction: build("hold_friction", groups),
        }
    }

    /// Red-black sweeps of the grain pressure after the first return, the
    /// common flow held fixed; the last pair writes the returned stress.
    pub fn relax_pressure(&self) -> Vec<usize> {
        let mut kernels = Vec::new();
        for _ in 1..pressure_pairs() {
            kernels.extend(self.pressure);
        }
        kernels.extend(self.pressure_last);
        kernels
    }

    /// The second cycle's contact response, after its prediction.
    pub fn resolve_contact(&self) -> Vec<usize> {
        vec![
            self.transfer_contact,
            self.prepare_stream,
            self.contact_velocity,
            self.contact_grain,
            self.copy_stress,
            self.grain_stress,
            self.prepare_stream,
        ]
    }
}
