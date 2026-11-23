//! Bubbles that split keep apart: the gas identities relabelled by connected
//! component after each update's transport (labels.wgsl), where bubbles
//! pressed one under another join and bubbles at rest are held.
use super::GAS_SLOTS;

/// Bubbles the relabelling tracks per update (components at least as large as
/// a round bubble a capillary length across); further ones keep their identity.
pub const MAX_COMPONENTS: u32 = 256;
/// A held count beyond any drainage time: the bubble has rested under what
/// holds it.
const RESTED: u32 = u32::MAX / 2;

/// The words of the labels buffer before the cells' words (labels.wgsl,
/// label_cell_anchor): the nodes, their areas, the record counter, the
/// records and the cells' two counts by identity.
pub fn cell_words(n: u32) -> u32 {
    let layers = GAS_SLOTS as u32 * n;
    4 * layers + 1 + (17 + 4 * GAS_SLOTS as u32) * MAX_COMPONENTS
}

/// The parts of the labels buffer (bytes) in which two runs agree: the
/// union-find's nodes, and each cell's pressed and held counts by identity
/// with its word. The rest holds the bubbles' record numbers, which follow
/// the order in which the GPU's threads reach the records.
pub fn agreeing(n: u32) -> [std::ops::Range<usize>; 2] {
    let layers = GAS_SLOTS * n as usize;
    let words = cell_words(n) as usize;
    [
        0..4 * layers,
        4 * (words - 2 * layers)..4 * (words + n as usize),
    ]
}

/// The labels buffer of a picture at rest.
pub fn initial(n: u32) -> Vec<u8> {
    let layers = GAS_SLOTS as u64 * u64::from(n);
    let records = 2 * layers + 1 + (17 + 4 * GAS_SLOTS as u64) * u64::from(MAX_COMPONENTS);
    let words = u64::from(cell_words(n)) + u64::from(n) + layers;
    // Every word starts as NO_LABEL (all bits set): no bubble is known yet.
    let mut contents = vec![0xff_u8; 4 * words as usize];
    // But the picture starts at rest, and the films over a resting bubble
    // drained long ago: each cell's held count (the second block after the
    // records, label_cell_settle) starts beyond any drainage time, so a
    // bubble that lies against the frame's top is anchored from the first
    // relabelling and not set free by the shake that begins the turn.
    let held = 4 * (records + layers) as usize..4 * (records + 2 * layers) as usize;
    for count in contents[held].chunks_exact_mut(4) {
        count.copy_from_slice(&RESTED.to_le_bytes());
    }
    // A cell's word starts with every identity anchored, as all bits set
    // would say, and without the bit that says its upper bits are kept
    // (label_cell_word): the first relabelling sets them.
    let first = 4 * cell_words(n) as usize;
    let anchored = (1_u32 << GAS_SLOTS) - 1;
    for word in contents[first..first + 4 * n as usize].chunks_exact_mut(4) {
        word.copy_from_slice(&anchored.to_le_bytes());
    }
    contents
}
