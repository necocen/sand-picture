// The gas identities a cell can hold.
const GAS_SLOTS:u32=9u;
// The rows of `geometry`, in units of the cell count cfg.n, one vec4 per
// entry and one entry per cell unless the row says otherwise: what an update
// leaves for the next beside the cells. Row 0 is each cell's gas plane
// (contacts.wgsl).
const LAYER_ROW:u32=1u;               // GAS_SLOTS rows: each identity's gas volume and label
const SHAPE_ROW:u32=10u;              // GAS_SLOTS rows: each identity's interface plane
const CONTACT_ROW:u32=19u;            // 2 rows: the contact fits of a cell
const CONTACT_AUDIT_ROW:u32=21u;      // 2 rows: the fits' residuals
const SOLVE_APERTURE_ROW:u32=23u;     // the four faces' apertures at the last stress solve
const SOLVE_STATE_ROW:u32=24u;        // the pressure and the wet volume at the last stress solve
const HOLD_ROW:u32=25u;               // the traction the plates hold, sliding or not, and the creep
// The rows of `scratch`, in the same units: what one update's kernels write
// before they read it. Nothing in them lasts into the next update.
const FLUX_ROW:u32=0u;                // 2 GAS_SLOTS rows: the gas flux through each face, by identity
const PRESSURE_PROBLEM_ROW:u32=18u;   // the pressure sweeps' local problem
const PRESSURE_MATERIAL_ROW:u32=19u;  // its material and whether it is solved
const PRESSURE_ITERATE_ROW:u32=20u;   // the stored pressure, its scale and the sweeps' iterate
const PRESSURE_TRIAL_ROW:u32=21u;     // the trial shear stress
const PRESSURE_COUPLING_ROW:u32=22u;  // 2 rows, per face: the coupling of the two cells' pressures
const BED_OCCUPANCY_ROW:u32=24u;      // the bed's share of a cell and its surface
const BED_PASSAGE_ROW:u32=25u;        // 2 rows, per face: the free area beside the bed
const GROUP_REQUEST_ROW:u32=27u;      // what a cell asks of a transport group
const FACE_PREDICTION_ROW:u32=28u;    // 2 rows, per face: the first projection's velocity
const GROUP_PARENT_ROW:u32=30u;       // the group's union-find parent
const GROUP_MAP_ROW:u32=31u;          // a cell's group: owner, size, free inventory
const GROUP_PAYLOAD_ROW:u32=32u;      // 4 rows: the group's contents
const PLUG_INFLOW_ROW:u32=36u;        // the grains the transport brings in through each face (plug.wgsl)
const GROUP_LABEL_ROW:u32=37u;        // GAS_SLOTS rows: the group's gas by identity
const TEETH_ROW:u32=46u;              // per column: the slab's undersides (teeth.wgsl)
const BODY_ROW:u32=47u;               // a packed body's last layer and where its faces shed (teeth.wgsl)
const PLUG_ROW:u32=48u;               // a body's grains before the transport (plug.wgsl)
const COLUMN_ROW:u32=49u;             // the column sliding through a slot between bubbles (teeth.wgsl)
const RETURNED_STRESS_ROW:u32=50u;    // the stress the grain constraint returns (mechanics_work.wgsl)
const OUTGOING_CAPACITY_ROW:u32=51u;  // each transport group's outgoing ratio per phase (transport_work.wgsl)
