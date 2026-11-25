//! Shader assembly: the four programs of the model as text. A program is its
//! constants and its fragments in a fixed order.
use super::DT;
use super::foam::{FOAM_VERTS, MAX_BUBBLES};
use super::grid::{CELLS, pressure_omega};
use super::labels::{MAX_COMPONENTS, cell_words};
use super::material::{
    BED_PACKING, BUBBLE_LEVELS, CLOSE_PACKING, CRITICAL_PACKING, FREEZING_PACKING, Grains,
    HINDRANCE, LOOSE_PACKING, POWDER_DIAMETER, POWDER_LEVELS, SURFACE_TENSION, bubble_rise_speeds,
    powder_speed,
};
use super::rows;
use super::stream::{COARSEN_ROWS, POINT_THREADS};
use crate::wgsl;

/// A program: the text several kernels' entry points live in, compiled as
/// one shader module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Program {
    /// Mechanics, interface, bed, slab, transport and labels: one text,
    /// since they share the cells' and faces' declarations and helpers.
    Model,
    /// The flow solver.
    Stream,
    /// The transport's subdivision schedule.
    Schedule,
    /// The turn's mirroring of the grid.
    Invert,
}

impl Program {
    pub const ALL: [Self; 4] = [Self::Model, Self::Stream, Self::Schedule, Self::Invert];

    /// The program's WGSL, as it is compiled.
    pub fn source(self, grains: &Grains) -> String {
        wgsl::compiled(self.text(grains))
    }

    /// The program's constants and fragments, its loops as written. Every
    /// program starts with the rows' constants, by which it addresses the
    /// two row buffers.
    fn text(self, grains: &Grains) -> String {
        let cells = (CELLS[0] * CELLS[1]) as u32;
        match self {
            Self::Model => model(grains),
            // The update interval and the surface tension enter the capillary
            // terms as shader constants; DT and SURFACE_TENSION stay their
            // sources, as COARSEN_ROWS does for the coarsening's workgroup
            // memory and the labels' layout for the place of the cells' words.
            Self::Stream => format!(
                "{}const CAPILLARY_DT:f32={DT:?};const SURFACE_TENSION:f32={SURFACE_TENSION:?};const COARSEN_ROWS:u32={COARSEN_ROWS}u;const POINT_THREADS:u32={POINT_THREADS}u;const LABEL_CELL_WORDS:u32={}u;\n{}\n{}\n{}",
                rows::constants(),
                cell_words(cells),
                include_str!("stream/stream.wgsl"),
                include_str!("stream/free_stream.wgsl"),
                include_str!("stream/fcg.wgsl")
            ),
            Self::Schedule => rows::constants() + include_str!("carry_schedule.wgsl"),
            Self::Invert => format!(
                "{}const LABEL_CELL_WORDS:u32={}u;\n{}",
                rows::constants(),
                cell_words(cells),
                include_str!("invert.wgsl")
            ),
        }
    }
}

/// The arrival bookkeeping keeps one value per row of a column, and the
/// grain-pressure relaxation's over-relaxation follows the grid's height, so
/// both are constants of the shader. The grains' diameters and settling
/// speeds, the packings and the liquid's surface tension come from
/// `material`, their single source.
fn model(grains: &Grains) -> String {
    let [columns, rows] = CELLS;
    let max_side = columns.max(rows);
    let omega = pressure_omega();
    let [fine, coarse] = grains.diameters.map(|d| d as f32);
    let [slow, fast] = grains.terminal_speeds().map(|v| v as f32);
    let powder = powder_speed() as f32;
    let [freezing, loose, critical, bed, close] = [
        FREEZING_PACKING,
        LOOSE_PACKING,
        CRITICAL_PACKING,
        BED_PACKING,
        CLOSE_PACKING,
    ]
    .map(|p| p as f32);
    let hindrance = HINDRANCE as f32;
    let powder_diameter = POWDER_DIAMETER as f32;
    // The capillary length of the liquid, sqrt(sigma / (rho_w g)).
    let capillary = (SURFACE_TENSION / (997. * 9.80665)).sqrt();
    let rise = bubble_rise_speeds()
        .map(|v| format!("{:?}", v as f32))
        .join(",");
    [
        format!(
            "const MAX_SIDE:u32={max_side}u;const OMEGA:f32={omega:?};\
             const GRAIN_DIAMETER=vec2<f32>({fine:?},{coarse:?});\
             const TERMINAL_SPEED=vec2<f32>({slow:?},{fast:?});\
             const POWDER_SPEED:f32={powder:?};\
             const FREEZING_PACKING:f32={freezing:?};\
             const LOOSE_PACKING:f32={loose:?};\
             const CRITICAL_PACKING:f32={critical:?};\
             const BED_PACKING:f32={bed:?};\
             const CLOSE_PACKING:f32={close:?};\
             const HINDRANCE:f32={hindrance:?};\
             const SURFACE_TENSION:f32={SURFACE_TENSION:?};\
             const POWDER_DIAMETER:f32={powder_diameter:?};\
             const POWDER_LEVELS:u32={POWDER_LEVELS}u;\
             const MAX_COMPONENTS:u32={MAX_COMPONENTS}u;\
             const MAX_BUBBLES:u32={MAX_BUBBLES}u;\
             const FOAM_VERTS:u32={FOAM_VERTS}u;\
             const CAPILLARY_LENGTH:f32={capillary:?};\
             const BUBBLE_LEVELS:u32={BUBBLE_LEVELS}u;\
             const BUBBLE_RISE=array<f32,{}>({rise});",
            BUBBLE_LEVELS + 1
        )
        .as_str(),
        rows::constants().as_str(),
        include_str!("shared/shared.wgsl"),
        include_str!("mechanics/material.wgsl"),
        include_str!("mechanics/mechanics.wgsl"),
        include_str!("mechanics/pressure.wgsl"),
        include_str!("shared/passage.wgsl"),
        include_str!("mechanics/momentum.wgsl"),
        include_str!("interface/interface.wgsl"),
        include_str!("transport/transport.wgsl"),
        include_str!("interface/contacts.wgsl"),
        include_str!("mechanics/contact_law.wgsl"),
        include_str!("mechanics/mechanics_work.wgsl"),
        include_str!("transport/transport_work.wgsl"),
        include_str!("transport/free_domain.wgsl"),
        include_str!("bed/bed.wgsl"),
        include_str!("transport/phase_transport.wgsl"),
        include_str!("transport/powder.wgsl"),
        include_str!("transport/gas_transport.wgsl"),
        include_str!("bed/surface.wgsl"),
        include_str!("bed/arrival.wgsl"),
        include_str!("bed/displacement.wgsl"),
        include_str!("bed/corrector.wgsl"),
        include_str!("bed/bed_geometry.wgsl"),
        include_str!("transport/transport_groups.wgsl"),
        include_str!("transport/group_capacity.wgsl"),
        include_str!("transport/advect_groups.wgsl"),
        include_str!("mechanics/contact_transfer.wgsl"),
        include_str!("bed/strata.wgsl"),
        include_str!("slab/teeth.wgsl"),
        include_str!("slab/body.wgsl"),
        include_str!("slab/breaching.wgsl"),
        include_str!("slab/joints.wgsl"),
        include_str!("transport/plug.wgsl"),
        include_str!("labels.wgsl"),
        include_str!("foam.wgsl"),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::{Grains, Program, wgsl};

    #[test]
    fn every_program_is_valid_wgsl() {
        let grains = Grains::default();
        for program in Program::ALL {
            wgsl::assert_valid(&format!("{program:?}"), &program.text(&grains));
        }
    }
}
