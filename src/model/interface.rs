//! Wet aperture reconstruction from the foam bubbles' exact edge planes.
use super::grid::CELLS;
use super::kernel::Plan;
use super::shader::Program;
use super::update::Plans;

pub struct Interface {
    pub apertures: usize,
}
impl Interface {
    pub fn new(plans: &mut Plans) -> Self {
        let n = (CELLS[0] * CELLS[1]) as u32;
        let edges = (2 * n).div_ceil(128);
        Self {
            apertures: plans.add(Plan::new(Program::Model, "wet_apertures", edges)),
        }
    }
}
