//! The rows of the model's two row buffers. Each holds one vec4 an entry, in
//! rows of the cell count: a use takes whole rows, with one entry a cell
//! unless it says otherwise.
//!
//! `geometry` is what an update leaves for the next beside the cells: it is
//! part of the picture's state. `scratch` is what one update's kernels write
//! before they read it: nothing in it lasts into the next update, so it is
//! neither saved nor compared. (The preparation of the first update runs no
//! slab stage and reads the slab's rows as the zeros a new buffer holds.)
//!
//! A use's first row follows from the uses before it, so no two share a row.
//! The shaders call that row `NAME_ROW`.
use super::GAS_SLOTS;

/// A use of a row buffer: its name in the shaders and the rows it takes.
type Use = (&'static str, u64);

const SLOTS: u64 = GAS_SLOTS as u64;

pub const GEOMETRY: &[Use] = &[
    // Each identity's gas volume and label.
    ("LAYER", SLOTS),
    // Each identity's interface plane.
    ("SHAPE", SLOTS),
    // The two contact fits of a cell.
    ("CONTACT", 2),
    // The four faces' apertures at the last stress solve, and the pressure
    // and the wet volume then.
    ("SOLVE_APERTURE", 1),
    ("SOLVE_STATE", 1),
    // The traction the plates hold, sliding or not, and the creep.
    ("HOLD", 1),
    // The bed's share of a cell and its surface; per face, the free area
    // beside the bed. An update writes them before it reads them, but the
    // interface rebuilt when a turn mirrors the grid reads the ones the
    // update before left.
    ("BED_OCCUPANCY", 1),
    ("BED_PASSAGE", 2),
    // The plug descent's set-in (plug.wgsl): how far the sand of a cell has
    // given way to a draw from below (1 when it flows with it), and from the
    // last walk the cell's supply and whether it was a link.
    ("SET_IN", 1),
];

pub const SCRATCH: &[Use] = &[
    // The slab (teeth.wgsl): per column its undersides, a packed body's last
    // layer and where its faces shed, and the column sliding through a slot
    // between bubbles.
    ("TEETH", 1),
    ("BODY", 1),
    ("COLUMN", 1),
    // A body's grains before the transport, and the grains the transport
    // brings in through each face (plug.wgsl).
    ("PLUG", 1),
    ("PLUG_INFLOW", 1),
    // The pressure sweeps: a cell's local problem, its material and whether
    // it is solved, the stored pressure with its scale and the sweeps'
    // iterate, the trial shear stress; per face, the coupling of the two
    // cells' pressures.
    ("PRESSURE_PROBLEM", 1),
    ("PRESSURE_MATERIAL", 1),
    ("PRESSURE_ITERATE", 1),
    ("PRESSURE_TRIAL", 1),
    ("PRESSURE_COUPLING", 2),
    // The stress the grain constraint returns (mechanics_work.wgsl).
    ("RETURNED_STRESS", 1),
    // Per face, the first projection's velocity.
    ("FACE_PREDICTION", 2),
    // The transport's groups: what a cell asks of a group, the group's
    // union-find parent, a cell's group (owner, size, free inventory), and
    // the group's contents.
    ("GROUP_REQUEST", 1),
    ("GROUP_PARENT", 1),
    ("GROUP_MAP", 1),
    ("GROUP_PAYLOAD", 4),
    // Each group's outgoing ratio per phase, and its receiving ratio
    // (transport_work.wgsl).
    ("OUTGOING_CAPACITY", 1),
    ("RECEIVING_CAPACITY", 1),
];

/// The rows the uses take together.
pub fn count(uses: &[Use]) -> u64 {
    uses.iter().map(|(_, rows)| rows).sum()
}

/// The first row of the use called `name`.
pub fn first(uses: &[Use], name: &str) -> u64 {
    let at = uses
        .iter()
        .position(|(known, _)| *known == name)
        .unwrap_or_else(|| panic!("no rows are called {name}"));
    count(&uses[..at])
}

/// The constants by which a shader addresses the two buffers: the gas
/// identities a cell can hold, and every use's first row.
pub fn constants() -> String {
    let mut text = format!("const GAS_SLOTS:u32={GAS_SLOTS}u;");
    for uses in [GEOMETRY, SCRATCH] {
        for (name, _) in uses {
            text.push_str(&format!("const {name}_ROW:u32={}u;", first(uses, name)));
        }
    }
    text.push('\n');
    text
}
