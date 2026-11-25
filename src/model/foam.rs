//! Prototype A: 2D Lagrangian closed-curve deformable foam bubble network.
//!
//! Each bubble is represented as a closed CCW polygon of `FOAM_VERTS` vertices
//! in cell coordinates `[0, NX] x [0, NY]`, carrying its exact conserved area,
//! graph-colored gas slot (`0..GAS_SLOTS`), velocity, bounding box, and
//! vertical-stack film drainage timer for 2-second coalescence.
use super::GAS_SLOTS;
use super::grid::CELLS;
use super::kernel::Plan;
use super::scene::Scene;
use super::shader::Program;
use super::update::Plans;
use std::collections::VecDeque;

/// Maximum simultaneous bubbles tracked in the foam buffer.
pub const MAX_BUBBLES: usize = 128;
/// Vertices per closed bubble curve.
pub const FOAM_VERTS: usize = 28;
/// Byte size of one `FoamBubble` record in WGSL (16-byte aligned: 64 + 28*8 = 288).
pub const FOAM_BUBBLE_BYTES: usize = 288;

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Default target total air area (cells^2) matching the polydisperse single-layer foam
/// (108% of the earlier 435 cells^2 baseline): richer bubble barrier while keeping
/// worst-case horizontal span safely below wall-to-wall sealing across flips.
pub const DEFAULT_AIR_AREA: f32 = 470.0;
/// Bounds for dynamic user air adjustment (cells^2).
pub const MIN_AIR_AREA: f32 = 150.0;
pub const MAX_AIR_AREA: f32 = 720.0;
/// Steady-state Rayleigh-Taylor breakup area threshold (cells^2), matching `foam.wgsl`.
const STEADY_SPLIT_AREA: f32 = 16.5;

struct RawBubble {
    area: f32,
    cx: f32,
    cy: f32,
    bbox_min: [f32; 2],
    bbox_max: [f32; 2],
    verts: [[f32; 2]; FOAM_VERTS],
    slot: u32,
}

/// Extract the initial ceiling bubbles from `scene.gas_layers` and match them to
/// the natural single-layer steady-state population (~105-110 bubbles, varied sizes
/// up to the Rayleigh-Taylor splitting scale, exact conserved total air).
pub fn initial(scene: &Scene, target_air_area: f32) -> Vec<u8> {
    let [nx, ny] = CELLS;
    let mut visited = vec![false; nx * ny * GAS_SLOTS];
    let mut seeds: Vec<(f32, f32)> = Vec::new(); // (cx, area)

    for slot in 0..GAS_SLOTS {
        for y in 0..ny {
            for x in 0..nx {
                let start = (y * nx + x) * GAS_SLOTS + slot;
                if visited[start] || scene.gas_layers[y * nx + x][slot] < 0.02 {
                    continue;
                }
                let mut queue = VecDeque::new();
                queue.push_back((x, y));
                visited[start] = true;
                let mut area = 0.0_f64;
                let mut sx = 0.0_f64;
                while let Some((cx, cy)) = queue.pop_front() {
                    let g = f64::from(scene.gas_layers[cy * nx + cx][slot]);
                    area += g;
                    sx += g * (cx as f64 + 0.5);
                    for (dx, dy) in [(-1_isize, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx_i = cx as isize + dx;
                        let ny_i = cy as isize + dy;
                        if nx_i < 0 || ny_i < 0 || nx_i >= nx as isize || ny_i >= ny as isize {
                            continue;
                        }
                        let (qx, qy) = (nx_i as usize, ny_i as usize);
                        let key = (qy * nx + qx) * GAS_SLOTS + slot;
                        if !visited[key] && scene.gas_layers[qy * nx + qx][slot] >= 0.005 {
                            visited[key] = true;
                            queue.push_back((qx, qy));
                        }
                    }
                }
                // Keep small capillary pearls (>= 0.70 cells^2) alongside medium domes
                // and wide flat ribbons so the foam starts truly polydisperse.
                if area >= 0.70 {
                    seeds.push(((sx / area) as f32, area as f32));
                }
            }
        }
    }

    seeds.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Scale extracted air cleanly to `target_air_area` and split any oversized blobs
    // asymmetrically (matching foam_coalesce's Rayleigh-Taylor breakup) so the
    // initial population starts in the natural steady-state size distribution.
    let extracted_gas: f64 = seeds.iter().map(|&(_, a)| f64::from(a)).sum();
    let scale = if extracted_gas > 1.0 {
        (f64::from(target_air_area.clamp(MIN_AIR_AREA, MAX_AIR_AREA)) / extracted_gas) as f32
    } else {
        1.0
    };

    const SPLIT_FRACS: [f32; 4] = [0.26, 0.36, 0.45, 0.62];
    let mut split_idx = 0usize;
    let mut steady: Vec<(f32, f32)> = Vec::new();
    for (cx, raw_area) in seeds {
        let mut stack = vec![(cx, raw_area * scale)];
        while let Some((x, area)) = stack.pop() {
            if area > STEADY_SPLIT_AREA && steady.len() + stack.len() + 2 <= MAX_BUBBLES - 12 {
                let frac = SPLIT_FRACS[split_idx % SPLIT_FRACS.len()];
                split_idx += 1;
                let a_left = area * frac;
                let a_right = area - a_left;
                let sep = 1.25 * area.sqrt();
                stack.push((x + frac * sep, a_right));
                stack.push((x - (1.0 - frac) * sep, a_left));
            } else if steady.len() < MAX_BUBBLES - 8 {
                steady.push((x, area));
            }
        }
    }
    steady.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Shape each ceiling bubble at its natural Young-Laplace aspect ratio in
    // surfactant water: tiny pearls (< 3 cells^2) stay round while large pockets
    // (12-22 cells^2) flatten horizontally at capillary half-height ~1.35 cells.
    // Preserve `assets/air.bin`'s organic horizontal spacing without carving
    // artificial gaps or squishing into uniform teeth.
    let pad = 0.06_f32;
    let usable_w = nx as f32 - 2.0 * pad;
    let mut geom: Vec<(f32, f32, f32, f32)> = steady
        .iter()
        .map(|&(cx, area)| {
            let aspect = 1.0 + 2.4 * ((area - 2.5) / 17.5).clamp(0.0, 1.0);
            let ry = (area / (std::f32::consts::PI * aspect))
                .sqrt()
                .clamp(0.45, 1.42);
            let rx = (area / (std::f32::consts::PI * ry)).max(0.45);
            (cx, area, rx, ry)
        })
        .collect();

    let raw_span: f32 = geom.iter().map(|g| 2.0 * g.2).sum();
    if raw_span > 0.94 * usable_w && raw_span > 1.0 {
        let sx = (0.94 * usable_w) / raw_span;
        for g in &mut geom {
            g.2 = (g.2 * sx).max(0.42);
            g.3 = g.1 / (std::f32::consts::PI * g.2);
        }
    }

    let gap = 0.04_f32;
    for _ in 0..24 {
        for g in &mut geom {
            g.0 = g.0.clamp(pad + g.2, nx as f32 - pad - g.2);
        }
        for i in 0..geom.len().saturating_sub(1) {
            let min_dist = geom[i].2 + geom[i + 1].2 + gap;
            let dist = geom[i + 1].0 - geom[i].0;
            if dist < min_dist {
                let push = 0.5 * (min_dist - dist);
                geom[i].0 -= push;
                geom[i + 1].0 += push;
            }
        }
    }
    for g in &mut geom {
        g.0 = g.0.clamp(pad + g.2, nx as f32 - pad - g.2);
    }

    let mut list: Vec<RawBubble> = Vec::with_capacity(geom.len());
    let top_y = ny as f32 - pad;
    for (idx, &(cx, area_f, rx, ry)) in geom.iter().enumerate() {
        // Flat-topped ceiling dome: elliptical belly pressed smoothly against top_y.
        let cy = (top_y - 0.78 * ry).clamp(pad + ry, top_y - 0.35);
        let mut verts = [[0.0_f32; 2]; FOAM_VERTS];
        for (k, v) in verts.iter_mut().enumerate() {
            let theta = 2.0 * std::f32::consts::PI * (k as f32) / (FOAM_VERTS as f32);
            let s = theta.sin();
            let dy = if s > 0.0 {
                0.78 * ry * s.powf(0.65)
            } else {
                ry * s
            };
            v[0] = (cx + rx * theta.cos()).clamp(pad, nx as f32 - pad);
            v[1] = (cy + dy).clamp(pad, top_y);
        }
        // Rescale belly depth slightly so the initial polygon matches exact `area_f`.
        let mut a2 = 0.0_f32;
        for k in 0..FOAM_VERTS {
            let p0 = verts[k];
            let p1 = verts[(k + 1) % FOAM_VERTS];
            a2 += p0[0] * p1[1] - p1[0] * p0[1];
        }
        let cur_a = (0.5 * a2).max(1e-4);
        let sy = (area_f / cur_a).clamp(0.75, 1.35);
        let mut bmin = [nx as f32, ny as f32];
        let mut bmax = [0.0_f32, 0.0_f32];
        for v in &mut verts {
            v[1] = (top_y - (top_y - v[1]) * sy).clamp(pad, top_y);
            bmin[0] = bmin[0].min(v[0]);
            bmin[1] = bmin[1].min(v[1]);
            bmax[0] = bmax[0].max(v[0]);
            bmax[1] = bmax[1].max(v[1]);
        }
        list.push(RawBubble {
            area: area_f,
            cx,
            cy: 0.5 * (bmin[1] + bmax[1]),
            bbox_min: bmin,
            bbox_max: bmax,
            verts,
            slot: (idx % GAS_SLOTS) as u32,
        });
    }

    // Greedy graph coloring so nearby bubbles never share a GAS_SLOTS index.
    let margin = 2.0_f32;
    for i in 0..list.len() {
        let mut used = 0_u32;
        for j in 0..i {
            let overlap_x = list[i].bbox_min[0] <= list[j].bbox_max[0] + margin
                && list[j].bbox_min[0] <= list[i].bbox_max[0] + margin;
            let overlap_y = list[i].bbox_min[1] <= list[j].bbox_max[1] + margin
                && list[j].bbox_min[1] <= list[i].bbox_max[1] + margin;
            if overlap_x && overlap_y {
                used |= 1_u32 << list[j].slot;
            }
        }
        let mut chosen = (i % GAS_SLOTS) as u32;
        for s in 0..GAS_SLOTS as u32 {
            if (used & (1_u32 << s)) == 0 {
                chosen = s;
                break;
            }
        }
        list[i].slot = chosen;
    }

    let mut out = Vec::with_capacity(MAX_BUBBLES * FOAM_BUBBLE_BYTES);
    for b in &list {
        push_f32(&mut out, b.area);
        push_f32(&mut out, 0.0); // drain_time
        push_u32(&mut out, b.slot);
        push_u32(&mut out, 1); // alive
        push_f32(&mut out, b.cx);
        push_f32(&mut out, b.cy);
        push_f32(&mut out, 0.0); // vel.x
        push_f32(&mut out, 0.0); // vel.y
        push_f32(&mut out, b.bbox_min[0]);
        push_f32(&mut out, b.bbox_min[1]);
        push_f32(&mut out, b.bbox_max[0]);
        push_f32(&mut out, b.bbox_max[1]);
        push_u32(&mut out, u32::MAX); // drain_partner
        push_u32(&mut out, 0); // held (resting freely under smooth top glass)
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        for v in &b.verts {
            push_f32(&mut out, v[0]);
            push_f32(&mut out, v[1]);
        }
    }
    while out.len() < MAX_BUBBLES * FOAM_BUBBLE_BYTES {
        out.push(0);
    }
    out
}

pub struct Foam {
    pub step: usize,
    pub coalesce: usize,
    pub raster: usize,
    pub invert: usize,
    pub adjust_air: usize,
    pub measure: usize,
}

impl Foam {
    pub fn new(plans: &mut Plans) -> Self {
        let cells = ((CELLS[0] * CELLS[1]) as u32).div_ceil(128);
        Self {
            step: plans.add(Plan::new(Program::Model, "foam_step", MAX_BUBBLES as u32)),
            coalesce: plans.add(Plan::new(Program::Model, "foam_coalesce", 1)),
            raster: plans.add(Plan::new(Program::Model, "foam_raster", cells)),
            invert: plans.add(Plan::new(Program::Model, "invert_foam", 1)),
            adjust_air: plans.add(Plan::new(Program::Model, "foam_adjust_air", 1)),
            measure: plans.add(Plan::new(Program::Model, "measure_sand", 1)),
        }
    }

    /// Full foam update: advance closed curves, coalesce stacked films, rasterize onto grid.
    pub fn kernels(&self) -> Vec<usize> {
        vec![self.step, self.coalesce, self.raster]
    }
}
