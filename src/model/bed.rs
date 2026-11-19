//! The floor-connected bed: release, capture, surface remap and water
//! displacement, before the first flow cycle and between the two.
use super::grid::CELLS;
use super::kernel::Plan;
use super::shader::Program;
use super::update::Plans;

pub struct Bed {
    pub geometry: usize,
    pub occupancy: usize,
    pub passage: usize,
    pub release: usize,
    pub exchange: usize,
    pub capture_source: usize,
    pub begin_displacement: usize,
    pub surface_even: usize,
    pub surface_odd: usize,
    pub record_strata: usize,
    pub clear_strata: usize,
}
impl Bed {
    pub fn new(plans: &mut Plans) -> Self {
        let [nx, ny] = CELLS.map(|v| v as u32);
        let n = nx * ny;
        let groups = n.div_ceil(128);
        let edges = (2 * n).div_ceil(128);
        let columns = nx.div_ceil(128);
        let bed = |entry, groups| Plan::new(Program::Model, entry, groups);
        Self {
            geometry: plans.add(bed("bed_geometry", columns)),
            occupancy: plans.add(bed("prepare_bed_occupancy", groups)),
            passage: plans.add(bed("prepare_bed_passage", edges)),
            release: plans.add(bed("bed_release", groups)),
            exchange: plans.add(bed("bed_exchange", columns)),
            capture_source: plans.add(bed("capture_bed_source", groups)),
            begin_displacement: plans.add(bed("begin_displacement", edges)),
            // The surface flow reads the strata: it runs the column's top out
            // last laid first (peel). One thread per column.
            surface_even: plans.add(bed("surface_even", columns)),
            surface_odd: plans.add(bed("surface_odd", columns)),
            // The record, one thread per column.
            record_strata: plans.add(bed("record_strata", columns)),
            clear_strata: plans.add(bed("clear_strata", columns)),
        }
    }
}
