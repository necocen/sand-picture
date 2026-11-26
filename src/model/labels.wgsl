// Bubble identity and area lookup table (`labels`), populated each update by
// the Lagrangian foam network (`foam_commit` and `foam_raster`) and read by
// the slab joint detector (`label_area`) and the flow solver (`stream.wgsl`).

const NO_LABEL:u32=0xffffffffu;
// The buffer in words: parent[GAS_SLOTS n], each (cell, slot) node's owning
// bubble index (or NO_LABEL); compact[GAS_SLOTS n], mapping bubble index to
// record index; the record counter; MAX_COMPONENTS records (field 2 stores
// bubble area in 1/LABEL_AREA_UNIT cells); then each cell's pressed/held
// counts, anchored/held bitmask word (`label_cell_anchor`), and per-slot
// record indices.
@group(0) @binding(16) var<storage,read_write> labels:array<atomic<u32>>;
// Areas are stored as fixed-point integers in 1/LABEL_AREA_UNIT of a cell.
const LABEL_AREA_UNIT:f32=1024.;
const LABEL_RECORD:u32=17u+4u*GAS_SLOTS;

fn label_cell_anchor(i:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*MAX_COMPONENTS+2u*GAS_SLOTS*cfg.n+i;}
// Per-cell bitmask stored at `label_cell_anchor(i)`: bits 0..GAS_SLOTS mark
// anchored slots, bits GAS_SLOTS..2*GAS_SLOTS (`CELL_HELD`) mark occupied gas
// slots read by `stream.wgsl`, and bit 31 (`CELL_KEPT`) marks the word valid.
const CELL_HELD:u32=GAS_SLOTS;
const CELL_KEPT:u32=1u<<31u;

fn label_node(i:u32,l:u32)->u32{return i*GAS_SLOTS+l;}
fn label_compact(node:u32)->u32{return GAS_SLOTS*cfg.n+node;}
fn label_field(k:u32,field:u32)->u32{return 2u*GAS_SLOTS*cfg.n+1u+LABEL_RECORD*k+field;}

// The area (cells) of the bubble of slot l that cell i belongs to, or 0.
// The slab reads it at the start of the next update, before anything moves.
fn label_area(i:u32,l:u32)->f32{
    let parent=atomicLoad(&labels[label_node(i,l)]);if parent==NO_LABEL{return 0.;}
    let k=atomicLoad(&labels[label_compact(parent)]);if k>=MAX_COMPONENTS{return 0.;}
    return f32(atomicLoad(&labels[label_field(k,2u)]))/LABEL_AREA_UNIT;
}
