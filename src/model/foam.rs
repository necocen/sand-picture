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

struct RawBubble {
    area: f32,
    cx: f32,
    cy: f32,
    bbox_min: [f32; 2],
    bbox_max: [f32; 2],
    verts: [[f32; 2]; FOAM_VERTS],
    slot: u32,
}

fn polygon_area(verts: &[[f32; 2]; FOAM_VERTS]) -> f32 {
    let mut sum = 0.0_f32;
    for k in 0..FOAM_VERTS {
        let a = verts[k];
        let b = verts[(k + 1) % FOAM_VERTS];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

/// Extract the initial connected bubbles from `scene.gas_layers` as smooth
/// closed curves with exact area conservation and conflict-free slot coloring.
pub fn initial(scene: &Scene) -> Vec<u8> {
    let [nx, ny] = CELLS;
    let mut visited = vec![false; nx * ny * GAS_SLOTS];
    let mut list: Vec<RawBubble> = Vec::new();

    for slot in 0..GAS_SLOTS {
        for y in 0..ny {
            for x in 0..nx {
                let start = (y * nx + x) * GAS_SLOTS + slot;
                if visited[start] || scene.gas_layers[y * nx + x][slot] < 0.05 {
                    continue;
                }
                let mut queue = VecDeque::new();
                queue.push_back((x, y));
                visited[start] = true;
                let mut area = 0.0_f64;
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                let mut min_x = x;
                let mut max_x = x;
                let mut min_y = y;
                let mut max_y = y;
                while let Some((cx, cy)) = queue.pop_front() {
                    let g = f64::from(scene.gas_layers[cy * nx + cx][slot]);
                    area += g;
                    sx += g * (cx as f64 + 0.5);
                    sy += g * (cy as f64 + 0.5);
                    min_x = min_x.min(cx);
                    max_x = max_x.max(cx);
                    min_y = min_y.min(cy);
                    max_y = max_y.max(cy);
                    for (dx, dy) in [(-1_isize, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx_i = cx as isize + dx;
                        let ny_i = cy as isize + dy;
                        if nx_i < 0 || ny_i < 0 || nx_i >= nx as isize || ny_i >= ny as isize {
                            continue;
                        }
                        let (qx, qy) = (nx_i as usize, ny_i as usize);
                        let key = (qy * nx + qx) * GAS_SLOTS + slot;
                        if !visited[key] && scene.gas_layers[qy * nx + qx][slot] >= 0.05 {
                            visited[key] = true;
                            queue.push_back((qx, qy));
                        }
                    }
                }
                if area < 0.25 || list.len() >= MAX_BUBBLES {
                    continue;
                }
                let area_f = area as f32;
                let cx = (sx / area) as f32;
                let cy = (sy / area) as f32;
                let span_x = (max_x - min_x + 1) as f32;
                let span_y = (max_y - min_y + 1) as f32;
                let aspect = (span_x / span_y).clamp(0.7, 3.0);
                let mut ry = (area_f / (std::f32::consts::PI * aspect)).sqrt().max(0.32);
                let mut rx = area_f / (std::f32::consts::PI * ry);

                // Clamp center so the initial curve fits inside the tank's wetting film.
                let pad = 0.08_f32;
                let cx = cx.clamp(pad + rx.min(2.0), nx as f32 - pad - rx.min(2.0));
                let cy = cy.clamp(pad + ry.min(2.0), ny as f32 - pad - ry.min(2.0));
                rx = rx.min(cx - pad).min(nx as f32 - pad - cx).max(0.25);
                ry = ry.min(cy - pad).min(ny as f32 - pad - cy).max(0.25);

                let mut verts = [[0.0_f32; 2]; FOAM_VERTS];
                for (k, v) in verts.iter_mut().enumerate() {
                    let theta = 2.0 * std::f32::consts::PI * (k as f32) / (FOAM_VERTS as f32);
                    v[0] = (cx + rx * theta.cos()).clamp(pad, nx as f32 - pad);
                    v[1] = (cy + ry * theta.sin()).clamp(pad, ny as f32 - pad);
                }
                // Rescale slightly around centroid so exact polygon area matches `area_f`.
                let cur_area = polygon_area(&verts);
                if cur_area > 1e-4 {
                    let s = (area_f / cur_area).sqrt().clamp(0.7, 1.4);
                    for v in &mut verts {
                        v[0] = (cx + (v[0] - cx) * s).clamp(pad, nx as f32 - pad);
                        v[1] = (cy + (v[1] - cy) * s).clamp(pad, ny as f32 - pad);
                    }
                }
                let mut bmin = [nx as f32, ny as f32];
                let mut bmax = [0.0_f32, 0.0_f32];
                for v in &verts {
                    bmin[0] = bmin[0].min(v[0]);
                    bmin[1] = bmin[1].min(v[1]);
                    bmax[0] = bmax[0].max(v[0]);
                    bmax[1] = bmax[1].max(v[1]);
                }
                list.push(RawBubble {
                    area: area_f,
                    cx,
                    cy,
                    bbox_min: bmin,
                    bbox_max: bmax,
                    verts,
                    slot: slot as u32,
                });
            }
        }
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
        push_u32(&mut out, 1); // held (starts resting at top)
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
}

impl Foam {
    pub fn new(plans: &mut Plans) -> Self {
        let cells = ((CELLS[0] * CELLS[1]) as u32).div_ceil(128);
        Self {
            step: plans.add(Plan::new(Program::Model, "foam_step", MAX_BUBBLES as u32)),
            coalesce: plans.add(Plan::new(Program::Model, "foam_coalesce", 1)),
            raster: plans.add(Plan::new(Program::Model, "foam_raster", cells)),
            invert: plans.add(Plan::new(Program::Model, "invert_foam", 1)),
        }
    }

    /// Full foam update: advance closed curves, coalesce stacked films, rasterize onto grid.
    pub fn kernels(&self) -> Vec<usize> {
        vec![self.step, self.coalesce, self.raster]
    }
}
