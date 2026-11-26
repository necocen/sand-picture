//! The tank and its grid, with the numerical settings that follow the grid:
//! transport subdivisions, grain-pressure relaxation and flow iterations.

/// Inner width and height of the tank in metres. The plates are 1 mm apart.
pub const TANK: [f64; 2] = [0.24, 0.16];

/// Square cells across and up the tank, 1.11 mm wide: wider than the plate
/// gap, over which the depth-averaged model varies.
pub const CELLS: [usize; 2] = [216, 144];

/// Conjugate-gradient iterations of an update's first and second flow
/// solves. The rims of the bubbles converge slowest. The first solve starts
/// from the previous update's streamfunction, the second from zero. The
/// counts are part of the model's behaviour: with other counts the sand
/// drains at another rate.
pub const FLOW_ITERATIONS: [u32; 2] = [2, 3];

/// Cell width in metres, rounded once to the simulation's f32.
pub fn cell_width() -> f32 {
    (TANK[0] / CELLS[0] as f64) as f32
}

/// Transport subdivision budget per update: 8 for 180 columns and
/// proportionally more for finer cells, so the fastest flow carried without
/// slowing (one cell per subdivision) is the same physical speed, 1.67 m/s.
/// The flow asks one subdivision nearly always and up to 5 or 6 during the
/// shake and the turn; the budget caps the rare spikes of trace phases. Each
/// slot is encoded every update whether it runs or not.
pub fn subdivisions() -> u32 {
    (8 * CELLS[0] / 180) as u32
}

/// Over-relaxation of the grain-pressure relaxation. The slowest error spans
/// the sand's height, about 0.4 of the tank or N cells; for that mode
/// successive over-relaxation is fastest at 2/(1 + sin(pi/N)), where every
/// mode decays by omega - 1 per red-black pair (thinner sand, with a smaller
/// optimum, decays as fast).
pub fn pressure_omega() -> f32 {
    let height = (0.4 * CELLS[1] as f64).round();
    (2. / (1. + (std::f64::consts::PI / height).sin())) as f32
}

/// Red-black pairs per update: enough to remove 95% of any error at
/// `pressure_omega`.
pub fn pressure_pairs() -> u32 {
    let decay = f64::from(pressure_omega()) - 1.;
    (0.05_f64.ln() / decay.ln()).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::{pressure_omega, pressure_pairs, subdivisions};

    #[test]
    fn the_settings_of_the_fine_grid() {
        assert_eq!(subdivisions(), 9);
        assert_eq!(pressure_pairs(), 28);
        let decay = (f64::from(pressure_omega()) - 1.).powi(pressure_pairs() as i32);
        assert!(decay <= 0.05 && decay > 0.04, "{decay}");
    }
}
