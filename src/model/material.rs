//! The materials, the single source of their constants for the shaders: the
//! liquid, the grains, the white's powder, the packings and the small bubbles.
//! Its table, built once before the first update, condenses the gap: positive
//! factors of the mean resistance, with analytic absent-phase limits.
pub const SIDE: usize = 129;
pub const MAX_SOLID: f64 = 0.8;

/// The liquid's viscosity (Pa s).
pub fn liquid_viscosity() -> f64 {
    0.00094
}

/// Surface tension of the liquid against the gas (N/m): water with soap, as
/// sand pictures are filled.
pub const SURFACE_TENSION: f32 = 0.035;

/// The diameters (m) of the two grain species: the white's grit and the black.
pub struct Grains {
    pub diameters: [f64; 2],
}
impl Default for Grains {
    fn default() -> Self {
        Self {
            diameters: [0.000110, 0.000185],
        }
    }
}
impl Grains {
    pub fn terminal_speeds(&self) -> [f64; 2] {
        std::array::from_fn(|i| terminal_speed(self.diameters[i], [2450., 4700.][i]))
    }
}

/// The terminal settling speed (m/s) of one sphere of `diameter` (m) and
/// `density` (kg/m3) in the liquid: the Schiller--Naumann isolated-sphere
/// force balance. The plates' resistance is no part of it: it belongs only to
/// the depth condensation.
fn terminal_speed(diameter: f64, density: f64) -> f64 {
    drag_limited_speed(diameter, density - 997., 1.)
}

/// The speed (m/s) at which the Schiller--Naumann drag of a rigid sphere of
/// `diameter` (m), multiplied by `wall`, balances its weight relative to the
/// liquid's, `buoyant_density` (kg/m3) times g.
fn drag_limited_speed(diameter: f64, buoyant_density: f64, wall: f64) -> f64 {
    let viscosity = liquid_viscosity();
    let stokes = buoyant_density * 9.80665 * diameter * diameter / (18. * viscosity);
    let mut lo = 0.;
    let mut hi = stokes;
    for _ in 0..64 {
        let speed = (lo + hi) * 0.5;
        let reynolds = 997. * diameter * speed / viscosity;
        if wall * speed * (1. + 0.15 * reynolds.powf(0.687)) < stokes {
            lo = speed;
        } else {
            hi = speed;
        }
    }
    (lo + hi) * 0.5
}

pub const BUBBLE_LEVELS: usize = 16;
/// The rise speeds (m/s) of bubbles of diameter k / BUBBLE_LEVELS of the gap,
/// k = 0..=BUBBLE_LEVELS: 0, 24 mm/s at 0.31 of the gap, 44 mm/s at 0.63.
///
/// A bubble smaller than the 1 mm gap between the plates is a sphere between
/// them, touching neither, below the grid: the cells' flow moves the water
/// around it, and the bubble rises through that water at its own terminal
/// speed. Its surface is a rigid sphere's (the soap immobilises a small
/// bubble's interface; Clift, Grace and Weber 1978), its weight relative to
/// the water is the water's it displaces (air's is a thousandth of it), and
/// the plates multiply its drag by Faxen's factor for a sphere midway between
/// two plane walls, 1/(1 - 1.004 l + 0.418 l^3 + 0.21 l^4 - 0.169 l^5) in its
/// diameter over the gap l (Happel and Brenner 1965). The reflection series
/// holds to about l = 0.6; nearer contact the plates' drag is larger
/// (Ganatos, Pfeffer and Weinbaum 1980) and the speeds are upper bounds.
pub fn bubble_rise_speeds() -> [f64; BUBBLE_LEVELS + 1] {
    std::array::from_fn(|k| {
        let ratio = k as f64 / BUBBLE_LEVELS as f64;
        let wall = 1.
            / (1. - 1.004 * ratio + 0.418 * ratio.powi(3) + 0.21 * ratio.powi(4)
                - 0.169 * ratio.powi(5));
        drag_limited_speed(ratio * 0.001, 997., wall)
    })
}

/// The powder's diameter (m): the patent's 200-mesh silica is a flour of this
/// size, and the white haze a real picture's eroding masses shed settles under
/// a level top that descends at 0.2-0.5 mm/s, the speed of single silica
/// grains of 15-23 um at the haze's few percent.
pub const POWDER_DIAMETER: f64 = 20e-6;
/// The powder's settling speed (m/s): silica (2650 kg/m3) of POWDER_DIAMETER,
/// about 0.38 mm/s.
pub fn powder_speed() -> f64 {
    terminal_speed(POWDER_DIAMETER, 2650.)
}

/// Exponent of the voidage in the grains' drag per unit relative velocity,
/// Wen and Yu's hindrance: a grain among others slips through the liquid at
/// its terminal speed times the voidage to this power.
pub const HINDRANCE: f64 = 2.77;
/// The white sand of the patent (US4885192) is six parts 200-mesh silica
/// powder, two parts 120-mesh silica and three parts fine glass beads. The
/// fine species is its grit (110 um) and carries the powder as a share of its
/// volume (the cells' and the deposit's powder volumes). With one velocity per
/// species the white moves at the volume mean of its grit's and its powder's,
/// each slipping through the liquid at its own speed under the same body force
/// per volume, so its settling is the mean of their terminal speeds and the
/// transport splits its flux between them by the same ratios. The drag of its
/// packing is Kozeny--Carman's with the white's Sauter mean, whose inverse is
/// the volume mean of the grit's and the powder's inverses: at the patent's
/// share about twelve times the grit's own. The material table runs over the
/// share in POWDER_LEVELS steps.
pub const POWDER_LEVELS: usize = 9;
/// The white's settling speed (m/s) when the share `share` of its volume is
/// powder.
pub fn white_speed(grit: f64, share: f64) -> f64 {
    (1. - share) * grit + share * powder_speed()
}
/// The drag of the white's packing relative to its grit's own when the share
/// `share` of its volume is powder: Kozeny--Carman's with the Sauter mean.
pub fn white_packed_drag(grit_diameter: f64, share: f64) -> f64 {
    ((1. - share) + share * grit_diameter / POWDER_DIAMETER).powi(2)
}
/// The white's packed drag relative to its dilute drag: the table's drag
/// coefficient for the white runs from the mean speed's, dilute, to the
/// Sauter mean's Kozeny--Carman one, packed.
fn white_packed_ratio(grit_speed: f64, grit_diameter: f64, share: f64) -> f64 {
    white_packed_drag(grit_diameter, share) * white_speed(grit_speed, share) / grit_speed
}
/// The powder's share of the white species by volume, six of the patent's
/// eleven parts.
pub const POWDER_SHARE: f32 = 6. / 11.;

/// The material table: the two rows of `coefficients` for every powder level,
/// SIDE fine shares of the solid and SIDE total solid fractions (quadratic in
/// the index, up to MAX_SOLID).
pub fn table(grains: &Grains) -> Vec<[f32; 4]> {
    let [grit, heavy] = grains.terminal_speeds();
    let viscosity = liquid_viscosity();
    let mut result = Vec::with_capacity(POWDER_LEVELS * SIDE * SIDE * 2);
    for level in 0..POWDER_LEVELS {
        let powder = level as f64 / (POWDER_LEVELS - 1) as f64;
        let speeds = [white_speed(grit, powder), heavy];
        let packed = white_packed_ratio(grit, grains.diameters[0], powder);
        for share in 0..SIDE {
            for total in 0..SIDE {
                let solid = MAX_SOLID * (total as f64 / (SIDE - 1) as f64).powi(2);
                let fine = solid * share as f64 / (SIDE - 1) as f64;
                result.extend(
                    coefficients(
                        fine,
                        solid - fine,
                        grains.diameters,
                        speeds,
                        packed,
                        viscosity,
                    )
                    .map(|row| row.map(|v| v as f32)),
                );
            }
        }
    }
    result
}

/// The share of a sphere's volume below the plane `x` radii above its centre.
fn cdf(x: f64) -> f64 {
    if x <= -1. {
        0.
    } else if x >= 1. {
        1.
    } else {
        0.5 + 0.75 * x - 0.25 * x * x * x
    }
}
/// Depth nodes of the gap problem. A packing's wall layer changes over a tenth
/// of a grain (11 um for the fines), which 7.8 um nodes resolve.
const DEPTH_NODES: usize = 128;

// The packings (solid fractions of a cell's wet part) the model tells apart;
// the shaders take them from here (shader.rs).

/// Hard spheres freeze: above it grains are caged and begin to order against a wall.
pub const FREEZING_PACKING: f64 = 0.494;
/// Frictional grains touch at rest from the random loose packing up (Onoda and
/// Liniger 1990, Jerkins et al. 2008); a looser mixture is a suspension that
/// still settles.
pub const LOOSE_PACKING: f64 = 0.55;
/// The packing of slow steady shear (the critical state) of rounded natural sand
/// at the picture's stresses, tens of Pa (spheres: 0.585). Packed sand must
/// dilate to it before it flows, so the margin above it sets how fast a free
/// face of a packing gives way (breaching).
pub const CRITICAL_PACKING: f64 = 0.56;
/// The packing the deposit settles to and the scene's sand starts at.
pub const BED_PACKING: f64 = 0.585;
/// Random close packing: no wet part holds more grains.
pub const CLOSE_PACKING: f64 = 0.64;
/// Voidage of the closest sphere packing, 1 - pi/(3 sqrt 2).
const DENSEST_VOIDAGE: f64 = 0.2595;
/// Suspended grains: centres spread evenly between a radius from each plate.
fn suspended(z: f64, d: f64) -> f64 {
    let r = d / 2.;
    cdf((z - r) / r) - cdf((z - (0.001 - r)) / r)
}
/// A packing in contact: its first layer rests on the plates. The measured
/// wall voidage of random sphere packings (de Klerk 2003), its shape at a
/// bulk voidage of 0.36, where the two pieces meet.
fn packed(z: f64, d: f64) -> f64 {
    let x = z.min(0.001 - z) / d;
    let voidage = if x <= 0.637 {
        2.14 * x * x - 2.53 * x + 1.
    } else {
        0.36 + 0.29 * (-0.6 * x).exp() * (2.3 * std::f64::consts::PI * (x - 0.16)).cos()
            + 0.15 * (-0.9 * x).exp()
    };
    1. - voidage
}
/// One species' solid across the gap at the given total solid fraction, mean
/// one over the nodes: suspended below freezing, packed from contact, and a
/// smooth blend of the two shapes between.
fn depth_profile(solid: f64, d: f64) -> [f64; DEPTH_NODES] {
    let t = ((solid - FREEZING_PACKING) / (LOOSE_PACKING - FREEZING_PACKING)).clamp(0., 1.);
    let packing = t * t * (3. - 2. * t);
    let z = |i: usize| (i as f64 + 0.5) * 0.001 / DEPTH_NODES as f64;
    let mean = |shape: fn(f64, f64) -> f64| {
        (0..DEPTH_NODES).map(|i| shape(z(i), d)).sum::<f64>() / DEPTH_NODES as f64
    };
    let (loose, dense) = (mean(suspended), mean(packed));
    std::array::from_fn(|i| {
        (1. - packing) * suspended(z(i), d) / loose + packing * packed(z(i), d) / dense
    })
}
/// The condensed factors `[k, a_f, a_h, x]` and the plates' `[p_f, p_h]` at the
/// solid fractions `fine` (f) and `coarse` (h): the flow across the gap, of
/// mean voidage n, dissipates per volume `n k (u - f a_f u_f - h a_h u_h)^2`
/// `+ f p_f u_f^2 + h p_h u_h^2 + f h x (u_f - u_h)^2` for the liquid's mean
/// velocity u and the species' u_f, u_h (the test below). `packed` is the
/// white's packed drag relative to its dilute drag at the speed `speeds[0]`
/// (white_packed_ratio), reached from the freezing packing to contact.
fn coefficients(
    fine: f64,
    coarse: f64,
    diameters: [f64; 2],
    speeds: [f64; 2],
    packed: f64,
    viscosity: f64,
) -> [[f64; 4]; 2] {
    const N: usize = DEPTH_NODES;
    let profiles = diameters.map(|d| depth_profile(fine + coarse, d));
    let t = ((fine + coarse - FREEZING_PACKING) / (LOOSE_PACKING - FREEZING_PACKING)).clamp(0., 1.);
    let packed_drag = 1. + (packed - 1.) * t * t * (3. - 2. * t);
    let cs: [[f64; 2]; N] = std::array::from_fn(|i| [profiles[0][i], profiles[1][i]]);
    let mut mu = [0.; N];
    let mut b = [[0.; 3]; N];
    let mut mean = 0.;
    for i in 0..N {
        // No layer of spheres is denser than their closest packing; the table
        // runs past close packing only to keep lookups finite.
        let n = (1. - fine * cs[i][0] - coarse * cs[i][1]).max(DENSEST_VOIDAGE);
        // Bulk liquid shear contribution of Baumgarten--Kamrin (2019), eq.21.
        // This already includes the fluid fraction: multiplying by n again
        // would omit the particle-induced fluid stress. Relative drag stays
        // separate, as does the contact stress evolved on the GPU.
        mu[i] = viscosity * (1. + 2.5 * (1. - n));
        mean += n / N as f64;
        let beta = [
            cs[i][0] * 1453. * 9.80665 / speeds[0] * n.powf(-HINDRANCE) * packed_drag,
            cs[i][1] * 3703. * 9.80665 / speeds[1] * n.powf(-HINDRANCE),
        ];
        b[i] = [n, beta[0], beta[1]];
    }
    let scale = (N as f64 / 0.001).powi(2);
    let mut d = [0.; N];
    let mut lo = [0.; N];
    let mut x = b;
    for i in 0..N {
        let left = if i == 0 {
            2. * mu[i]
        } else {
            0.5 * (mu[i] + mu[i - 1])
        } * scale;
        let right = if i + 1 == N {
            2. * mu[i]
        } else {
            0.5 * (mu[i] + mu[i + 1])
        } * scale;
        d[i] = left + right + fine * b[i][1] + coarse * b[i][2];
        lo[i] = -left;
        if i > 0 {
            let factor = lo[i] / d[i - 1];
            d[i] -= factor * lo[i];
            let above = x[i - 1];
            for (value, above) in x[i].iter_mut().zip(above) {
                *value -= factor * above;
            }
        }
    }
    for value in &mut x[N - 1] {
        *value /= d[N - 1];
    }
    for i in (0..N - 1).rev() {
        let below = x[i + 1];
        for (value, below) in x[i].iter_mut().zip(below) {
            *value = (*value - lo[i + 1] * below) / d[i];
        }
    }
    let s: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|c| (0..N).map(|i| b[i][a] * x[i][c] / N as f64).sum())
    });
    // b/x retain the per-grain-volume forcing even when that species is absent.
    // The no-slip liquid solve has 0 <= fine*x_f + coarse*x_h < 1, so these
    // plate factors and the cross Green function are positive. Interpolating
    // them preserves a positive quadratic dissipation at arbitrary fractions.
    let plate: [f64; 2] = std::array::from_fn(|a| {
        (0..N)
            .map(|i| b[i][a + 1] * (1. - fine * x[i][1] - coarse * x[i][2]) / N as f64)
            .sum()
    });
    [
        [
            mean / s[0][0],
            s[0][1] / mean,
            s[0][2] / mean,
            (s[1][2] + s[2][1]) * 0.5,
        ],
        [plate[0], plate[1], 0., 0.],
    ]
}

#[test]
fn condensed_energy_matches_resolved_gap() {
    use faer::linalg::solvers::DenseSolveCore;

    const N: usize = DEPTH_NODES;
    let grains = Grains::default();
    let speeds = grains.terminal_speeds();
    let viscosity = liquid_viscosity();
    assert!(
        table(&grains)
            .iter()
            .flatten()
            .all(|v| v.is_finite() && *v >= 0.)
    );
    for (fine, heavy) in [
        (0., 0.),
        (0.0017538655, 9.1374e-37),
        (0.2, 0.),
        (0., 0.2),
        (0.1, 0.3),
        (0.28, 0.28),
        (0.4, 0.4),
        (0., 0.8),
    ] {
        let share = f64::from(POWDER_SHARE);
        let packed = white_packed_ratio(speeds[0], grains.diameters[0], share);
        let speeds = [white_speed(speeds[0], share), speeds[1]];
        let factors = coefficients(fine, heavy, grains.diameters, speeds, packed, viscosity);
        let profile = grains.diameters.map(|d| depth_profile(fine + heavy, d));
        let water: [f64; N] = std::array::from_fn(|i| {
            (1. - fine * profile[0][i] - heavy * profile[1][i]).max(DENSEST_VOIDAGE)
        });
        let porosity = water.iter().sum::<f64>() / N as f64;
        let mu: [f64; N] = water.map(|n| viscosity * (1. + 2.5 * (1. - n)));
        let t =
            ((fine + heavy - FREEZING_PACKING) / (LOOSE_PACKING - FREEZING_PACKING)).clamp(0., 1.);
        let powder = [1. + (packed - 1.) * t * t * (3. - 2. * t), 1.];
        let beta: [[f64; 2]; N] = std::array::from_fn(|i| {
            std::array::from_fn(|a| {
                [fine, heavy][a] * profile[a][i] * [1453., 3703.][a] * 9.80665 / speeds[a]
                    * water[i].powf(-HINDRANCE)
                    * powder[a]
            })
        });
        let scale = (N as f64 / 0.001).powi(2);
        let mut matrix = vec![0.; N * N];
        for i in 0..N {
            matrix[i * N + i] += beta[i][0] + beta[i][1];
            if i == 0 {
                matrix[i * N + i] += 2. * mu[i] * scale;
            }
            if i + 1 == N {
                matrix[i * N + i] += 2. * mu[i] * scale;
            }
            if i + 1 < N {
                let edge = 0.5 * (mu[i] + mu[i + 1]) * scale;
                matrix[i * N + i] += edge;
                matrix[(i + 1) * N + i + 1] += edge;
                matrix[i * N + i + 1] -= edge;
                matrix[(i + 1) * N + i] -= edge;
            }
        }
        let inverse = faer::Mat::from_fn(N, N, |i, j| matrix[i * N + j])
            .llt(faer::Side::Lower)
            .expect("resolved liquid resistance is positive definite")
            .inverse();
        let solve = |rhs: [f64; N]| {
            std::array::from_fn::<_, N, _>(|i| {
                (0..N).map(|j| inverse[(i, j)] * rhs[j]).sum::<f64>()
            })
        };
        let pressure = solve(water);
        let capacity = (0..N)
            .map(|i| water[i] * pressure[i] / N as f64)
            .sum::<f64>();
        for u in [
            [0.02, -0.03, 0.015],
            [1., 1., 1.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ] {
            let free = solve(std::array::from_fn(|i| {
                beta[i][0] * u[1] + beta[i][1] * u[2]
            }));
            let constraint = (porosity * u[0]
                - (0..N).map(|i| water[i] * free[i] / N as f64).sum::<f64>())
                / capacity;
            let liquid: [f64; N] = std::array::from_fn(|i| free[i] + constraint * pressure[i]);
            let mut resolved =
                2. * scale * (mu[0] * liquid[0].powi(2) + mu[N - 1] * liquid[N - 1].powi(2))
                    / N as f64;
            for i in 0..N {
                resolved += (beta[i][0] * (liquid[i] - u[1]).powi(2)
                    + beta[i][1] * (liquid[i] - u[2]).powi(2))
                    / N as f64;
                if i + 1 < N {
                    resolved +=
                        0.5 * (mu[i] + mu[i + 1]) * scale * (liquid[i] - liquid[i + 1]).powi(2)
                            / N as f64;
                }
            }
            let [k, af, ah, exchange] = factors[0];
            let condensed = porosity * k * (u[0] - fine * af * u[1] - heavy * ah * u[2]).powi(2)
                + fine * factors[1][0] * u[1].powi(2)
                + heavy * factors[1][1] * u[2].powi(2)
                + fine * heavy * exchange * (u[1] - u[2]).powi(2);
            assert!(
                (resolved - condensed).abs() <= 1e-9 * resolved.abs().max(1e-30),
                "{fine} {heavy} {u:?}: resolved {resolved}, condensed {condensed}"
            );
        }
    }
}
