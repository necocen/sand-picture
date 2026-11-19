//! The floor-connected bed: release, capture, surface remap and water
//! displacement, before the first flow cycle and between the two.
use super::buffers::Buffers;
use super::kernel::Plan;
use super::mechanics::common;
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
    pub fn new(plans: &mut Plans, buffers: &Buffers) -> Self {
        let n = buffers.n;
        let groups = n.div_ceil(128);
        let edges = (2 * n).div_ceil(128);
        let columns = buffers.nx.div_ceil(128);
        let bed = |entry, groups| {
            Plan::new(
                Program::Model,
                entry,
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (3, &buffers.faces),
                    (4, &buffers.geometry),
                    (5, &buffers.table),
                    (10, &buffers.terrain),
                    (11, &buffers.arrival),
                ],
                groups,
            )
        };
        // The surface flow also reads the strata: it runs the column's top out
        // last laid first (peel). One thread per column.
        let surface = |entry| {
            Plan::new(
                Program::Model,
                entry,
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (3, &buffers.faces),
                    (4, &buffers.geometry),
                    (5, &buffers.table),
                    (10, &buffers.terrain),
                    (11, &buffers.arrival),
                    (12, &buffers.strata),
                ],
                columns,
            )
        };
        Self {
            geometry: plans.add(bed("bed_geometry", columns)),
            occupancy: plans.add(common(buffers, "prepare_bed_occupancy", groups)),
            passage: plans.add(common(buffers, "prepare_bed_passage", edges)),
            release: plans.add(bed("bed_release", groups)),
            exchange: plans.add(bed("bed_exchange", columns)),
            capture_source: plans.add(bed("capture_bed_source", groups)),
            begin_displacement: plans.add(common(buffers, "begin_displacement", edges)),
            surface_even: plans.add(surface("surface_even")),
            surface_odd: plans.add(surface("surface_odd")),
            // The record, one thread per column.
            record_strata: plans.add(Plan::new(
                Program::Model,
                "record_strata",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (10, &buffers.terrain),
                    (12, &buffers.strata),
                ],
                columns,
            )),
            clear_strata: plans.add(Plan::new(
                Program::Model,
                "clear_strata",
                &[(0, &buffers.config), (12, &buffers.strata)],
                columns,
            )),
        }
    }
}
