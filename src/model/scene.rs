//! The deterministic, volume-conservative scene every run starts from: the
//! picture as the model itself left it at the end of a drainage, at rest, the
//! deposit on the floor (water and the two grain species per cell) and the
//! bubbles under the frame's top, each with its identity. A drainage starts
//! when the picture is turned over (`Model::start_turn`).
use super::GAS_SLOTS;
use super::grid::CELLS;
use super::material::{BED_PACKING, POWDER_SHARE};

pub struct Scene {
    /// Water, fine grains, coarse grains and gas of each cell, from the floor up.
    pub cells: Vec<[f32; 4]>,
    /// Each cell's gas by identity: slot l holds identity l + 1 (labels.wgsl).
    pub gas_layers: Vec<[f32; GAS_SLOTS]>,
}

impl Scene {
    /// The product's scene: the stored deposit, settled, and the stored
    /// bubbles in its water.
    pub fn product() -> Self {
        let settled = settle(&stored_scene());
        let (cells, gas_layers) = with_air(&settled, &stored_air());
        Self { cells, gas_layers }
    }

    /// The initial `cells` buffer, 24 values a cell: the scene's phases and
    /// powder, with only the sand already connected to the floor stored as
    /// deposit, at the packing of its column's bottom cell.
    pub fn cell_records(&self) -> Vec<f32> {
        let [nx, ny] = CELLS;
        let mut values = vec![[0_f32; 24]; self.cells.len()];
        for x in 0..nx {
            let bottom = self.cells[x];
            let packing = ((bottom[1] + bottom[2]) / (1. - bottom[3])).max(0.585);
            let mut connected = true;
            for y in 0..ny {
                let i = y * nx + x;
                let c = self.cells[i];
                let solid = c[1] + c[2];
                let wet = c[0] + solid;
                let domain = (solid / packing).min(wet);
                values[i][..4].copy_from_slice(&c);
                values[i][17] = POWDER_SHARE * c[1];
                if connected && solid > 0. {
                    values[i][17] = 0.;
                    values[i][18] = POWDER_SHARE * c[1];
                    values[i][20] = c[1];
                    values[i][21] = c[2];
                    values[i][22] = domain;
                    let pore = ((f64::from(domain) - f64::from(c[1]) - f64::from(c[2]))
                        .max(0.)
                        .min(f64::from(c[0]))) as f32;
                    values[i][23] = pore;
                    values[i][0] = (f64::from(c[0]) - f64::from(pore)) as f32;
                    values[i][1] = 0.;
                    values[i][2] = 0.;
                }
                // Allow only float rounding at a fully occupied material cell.
                connected &= domain >= 1. - 8. * f32::EPSILON;
            }
        }
        values.into_iter().flatten().collect()
    }
}

/// The scene's sand is the deposit of the last drainage, settled in water at
/// the deposit's packing, denser than the critical state of slow shear
/// (CRITICAL_PACKING): it cannot slide without dilating first, and the water
/// it has to draw in holds it while the row of bubbles forms. The margin over
/// the critical state sets how fast its faces give way and so how fast it
/// feeds the joints.
const SETTLED_PACKING: f32 = BED_PACKING as f32;

/// Each column's sand packed up from the floor to SETTLED_PACKING in the
/// order of its layers, with the same grains; water fills the rows it leaves.
/// The deposit is packed to its top row, as the model's own deposits are: a
/// looser top slides in the turn's shake and frees the white's powder.
fn settle(stored: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let [nx, ny] = CELLS;
    let mut cells = stored.to_vec();
    for x in 0..nx {
        // Grains of the column from the floor up: a queue of layers.
        let mut layers: Vec<[f64; 2]> = Vec::new();
        let mut last = None;
        for y in 0..ny {
            let c = stored[y * nx + x];
            if c[1] + c[2] > 0. {
                layers.push([f64::from(c[1]), f64::from(c[2])]);
                last = Some(y);
            }
        }
        let Some(top) = last else { continue };
        for y in 0..=top {
            let c = &mut cells[y * nx + x];
            let wet = c[0] + c[1] + c[2];
            *c = [wet, 0., 0.];
        }
        let wet_rows: Vec<f64> = (0..ny).map(|y| f64::from(cells[y * nx + x][0])).collect();
        // Pack the column's layers from the floor up; the grains per row and
        // the highest row that holds grains.
        let settled = f64::from(SETTLED_PACKING);
        let (rows, highest) = {
            let mut rows = vec![[0_f64; 2]; ny];
            let mut highest = 0;
            let mut layers = layers.into_iter();
            let mut piece = [0_f64; 2];
            for y in 0..ny {
                let mut room = settled * wet_rows[y];
                let mut grains = [0_f64; 2];
                while room > 0. {
                    if piece[0] + piece[1] <= 0. {
                        match layers.next() {
                            Some(next) => piece = next,
                            None => break,
                        }
                    }
                    let amount = piece[0] + piece[1];
                    let take = room.min(amount);
                    let share = take / amount;
                    grains[0] += piece[0] * share;
                    grains[1] += piece[1] * share;
                    piece[0] -= piece[0] * share;
                    piece[1] -= piece[1] * share;
                    room -= take;
                }
                if grains[0] + grains[1] <= 0. {
                    break;
                }
                rows[y] = grains;
                highest = y;
            }
            (rows, highest)
        };
        for y in 0..ny {
            let grains = rows[y];
            if grains[0] + grains[1] <= 0. && y > highest {
                continue;
            }
            cells[y * nx + x] = [
                (wet_rows[y] - grains[0] - grains[1]) as f32,
                grains[0] as f32,
                grains[1] as f32,
            ];
        }
    }
    cells
}

/// Water, fine and coarse grains of each cell, from the floor up, with gas
/// counted as water: the end of a drainage of the model, written out.
fn stored_scene() -> Vec<[f32; 3]> {
    include_bytes!("../../assets/scene.bin")
        .chunks_exact(12)
        .map(|row| {
            std::array::from_fn(|i| f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect()
}

/// The gas of each cell by identity, from records of cell, slot and volume
/// fraction.
///
/// The air is the bubbles the last drainage left under the frame's top: about
/// a hundred of 1 to 20 mm2, 650 mm2 in all. The turn puts them on the floor
/// under the water, and they rise to the slab's underside spread over its
/// width as the raft's bubbles: the model decides how many there are, how
/// large and where (labels.wgsl). The amount of air is the input, as the
/// maker's instructions put it: more air drains slower. Pressed into the
/// slab's underside the air is a puddle two capillary lengths thick,
/// 2 sqrt(sigma / (rho g)) = 3.4-3.8 mm, so a row of bubbles closes the
/// tank's width near 1010 mm2 of air; held bubbles creep only up their
/// support, so a raft of less air keeps the gaps it formed with, its joints.
/// The sand leaves through those joints once the hanging mounds are eaten
/// level (plug.wgsl), so the amount sets a life's length: with a fifth more
/// the raft closes most of them and the picture all but stops, with less the
/// life shortens steadily. The air starts as these bubbles, not
/// as one layer: a layer reaches the underside within a second as a few large
/// bubbles that gather in its recesses and leave the hanging mounds bare, and
/// a bare underside sheds as a curtain.
fn stored_air() -> Vec<[f32; GAS_SLOTS]> {
    let [nx, ny] = CELLS;
    let mut air = vec![[0_f32; GAS_SLOTS]; nx * ny];
    for record in include_bytes!("../../assets/air.bin").chunks_exact(12) {
        let word = |k: usize| u32::from_le_bytes(record[4 * k..4 * k + 4].try_into().unwrap());
        air[word(0) as usize][word(1) as usize] = f32::from_bits(word(2));
    }
    air
}

/// The bubbles take their gas from the water of the cells they lie in.
fn with_air(
    phases: &[[f32; 3]],
    air: &[[f32; GAS_SLOTS]],
) -> (Vec<[f32; 4]>, Vec<[f32; GAS_SLOTS]>) {
    phases
        .iter()
        .zip(air)
        .map(|(&[water, fine, coarse], layers)| {
            let stored: f32 = layers.iter().sum();
            let gas = stored.min(water);
            let kept = if stored > 0. { gas / stored } else { 0. };
            ([water - gas, fine, coarse, gas], layers.map(|v| v * kept))
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::{CELLS, SETTLED_PACKING, Scene, settle, stored_air, stored_scene};

    #[test]
    fn settling_keeps_each_columns_grains_and_packs_them_on_the_floor() {
        let stored = stored_scene();
        let [nx, ny] = CELLS;
        let settled = settle(&stored);
        for x in 0..nx {
            for phase in 0..3 {
                let before: f64 = (0..ny).map(|y| f64::from(stored[y * nx + x][phase])).sum();
                let after: f64 = (0..ny).map(|y| f64::from(settled[y * nx + x][phase])).sum();
                assert!(
                    (after - before).abs() <= 1e-4,
                    "column {x} phase {phase}: {before} vs {after}"
                );
            }
            let floor = settled[x];
            assert!(
                (floor[1] + floor[2] - SETTLED_PACKING).abs() < 1e-4,
                "{floor:?}"
            );
        }
    }

    #[test]
    fn the_stored_bubbles_keep_their_gas() {
        let stored: f64 = stored_air().iter().flatten().map(|&v| f64::from(v)).sum();
        assert!(stored > 350., "{stored} cells of air");
        let scene = Scene::product();
        let mut total = 0_f64;
        for (c, layers) in scene.cells.iter().zip(&scene.gas_layers) {
            let gas: f32 = layers.iter().sum();
            assert!((gas - c[3]).abs() <= 1e-6, "{c:?} {layers:?}");
            assert!(c.iter().all(|&v| v >= 0.), "{c:?}");
            assert!(c.iter().sum::<f32>() <= 1. + 1e-5, "{c:?}");
            total += f64::from(c[3]);
        }
        assert!((total / stored - 1.).abs() < 1e-5, "{total} of {stored}");
    }
}
