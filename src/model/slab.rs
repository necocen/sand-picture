//! The slab's structure before an update's transport: its teeth, the last
//! layers of its packed bodies and the columns sliding through the slots
//! between bubbles (slab/*.wgsl).
use super::buffers::Buffers;
use super::kernel::Plan;
use super::shader::Program;
use super::update::Plans;

pub struct Slab {
    /// Per column, the undersides of the sand and of the firm sand hanging
    /// from the ceiling.
    underside: usize,
    /// Per cell, the last layer of a packed body and the side the body lies on.
    faces: usize,
    /// Per cell, its slot between bubbles (gap_marks), read by the column scan.
    gaps: usize,
    /// Per column, the cells of the packed column that slides through a slot
    /// between bubbles below the body.
    columns: usize,
}
impl Slab {
    pub fn new(plans: &mut Plans, buffers: &Buffers) -> Self {
        let fields = [
            (0, &buffers.config),
            (1, &buffers.cells),
            (4, &buffers.geometry),
        ];
        let columns = buffers.nx.div_ceil(128);
        let cells = buffers.n.div_ceil(128);
        Self {
            underside: plans.add(Plan::new(
                Program::Model,
                "teeth_underside",
                &fields,
                columns,
            )),
            faces: plans.add(Plan::new(Program::Model, "body_faces", &fields, cells)),
            // The marks read the wet apertures the mechanics recorded on the
            // faces (slot_span): the water of two cells is one slot only
            // across a face open to water; and the bubbles' areas from the
            // last relabelling (spacer_in_row).
            gaps: plans.add(Plan::new(
                Program::Model,
                "gap_marks",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (3, &buffers.faces),
                    (4, &buffers.geometry),
                    (15, &buffers.gaps),
                    (16, &buffers.labels),
                ],
                cells,
            )),
            columns: plans.add(Plan::new(
                Program::Model,
                "column_cells",
                &[
                    (0, &buffers.config),
                    (1, &buffers.cells),
                    (4, &buffers.geometry),
                    (15, &buffers.gaps),
                    (16, &buffers.labels),
                ],
                columns,
            )),
        }
    }

    pub fn kernels(&self) -> Vec<usize> {
        vec![self.underside, self.faces, self.gaps, self.columns]
    }
}
