// Prototype A: 2D Lagrangian closed-curve deformable foam bubble network.
//
// Replaces 9-slot VOF gas advection, PLIC/contact reconstruction, connected-component
// relabelling, and stiff 5x5 VOF surface tension with explicit 28-vertex closed curves:
// - Young-Laplace hydrostatic balance (sigma * kappa = dp0 + rho_w * g * (y - y_mean))
//   naturally flattens wide bubbles into horizontal-bottom air cushions while keeping
//   small bubbles (< capillary length) round spacers.
// - Semi-implicit curvature smoothing on the 28-vertex ring in workgroup shared memory
//   is unconditionally stable at sigma = 0.03 N/m in microseconds.
// - Exact polygon area conservation via free-vertex-weighted Lagrange multiplier.
// - Snug contact against the wavy sand underside and bed (eliminating the 1-2 cell VOF
//   water gap) and symmetric flat soap-film contact + Plateau borders between bubbles.
// - 2-second vertical-stack film drainage coalescence into wider horizontal cushions.

struct FoamBubble {
    target_area: f32,
    drain_time: f32,
    slot: u32,
    alive: u32,
    centroid: vec2<f32>,
    vel: vec2<f32>,
    bbox_min: vec2<f32>,
    bbox_max: vec2<f32>,
    drain_partner: u32,
    held: u32,
    disp_vel: vec2<f32>,
    verts: array<vec2<f32>, FOAM_VERTS>,
}

@group(0) @binding(17) var<storage, read_write> foam: array<FoamBubble>;
@group(0) @binding(18) var<storage, read_write> foam_next: array<FoamBubble>;
// Fixed-point (1/1024 cell) summary metrics written on demand by `measure_sand`:
// [0] total sand, [1] floor-connected bed sand, [2] mobile sand > 6 cells above bed,
// [3] total alive foam target area.
@group(0) @binding(19) var<storage, read_write> metrics: array<u32, 4>;

const FOAM_SUBSTEPS: u32 = 6u;
const FOAM_JACOBI: u32 = 6u;
const FOAM_GAMMA: f32 = 85.0;        // Meniscus normal mobility drag (Pa s / m)
const FOAM_SIGMA: f32 = 0.0170;      // Surfactant-water interfacial tension (N/m): capillary thickness ~2.1 cells under sand
const FOAM_FILM_GAP: f32 = 0.05;     // Soap film half-gap between touching bubbles (cells)
const FOAM_WALL_PAD: f32 = 0.05;     // Wetting water film at tank walls (cells)
const FOAM_SAND_LEVEL: f32 = 0.25;   // Packed sand outer boundary contour (avoids clipping gas in raster)
const FOAM_MAX_NBRS: u32 = 24u;

var<workgroup> sh_pos: array<vec2<f32>, FOAM_VERTS>;
var<workgroup> sh_pred: array<vec2<f32>, FOAM_VERTS>;
var<workgroup> sh_tmp: array<vec2<f32>, FOAM_VERTS>;
var<workgroup> sh_contact: array<f32, FOAM_VERTS>;
var<workgroup> sh_uf: array<vec3<f32>, FOAM_VERTS>;
var<workgroup> sh_nbrs: array<u32, FOAM_MAX_NBRS>;
var<workgroup> sh_nbr_count: u32;
var<workgroup> sh_centroid: vec2<f32>;
var<workgroup> sh_vel: vec2<f32>;
var<workgroup> sh_area: f32;
var<workgroup> sh_perim: f32;
var<workgroup> sh_free_perim: f32;
var<workgroup> sh_held: atomic<u32>;
var<workgroup> sh_step_alive: u32;
// Shared bounding boxes for fast foam_raster culling (128 threads load 128 bubbles once).
var<workgroup> sh_bmin: array<vec2<f32>, MAX_BUBBLES>;
var<workgroup> sh_bmax: array<vec2<f32>, MAX_BUBBLES>;
var<workgroup> sh_alive: array<u32, MAX_BUBBLES>;

fn foam_solid_cell(cx: i32, cy: i32) -> f32 {
    let p = clamp(vec2<i32>(cx, cy), vec2<i32>(0), vec2<i32>(i32(COLUMNS) - 1, i32(ROWS) - 1));
    let c = cells[at(p)];
    return max(c.volume.fine, 0.) + max(c.volume.coarse, 0.) + max(c.bed.x, 0.) + max(c.bed.y, 0.);
}

fn foam_solid_at(pos: vec2<f32>) -> f32 {
    let g = clamp(pos - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(f32(COLUMNS - 1u), f32(ROWS - 1u)));
    let x0 = i32(floor(g.x));
    let y0 = i32(floor(g.y));
    let x1 = min(x0 + 1, i32(COLUMNS) - 1);
    let y1 = min(y0 + 1, i32(ROWS) - 1);
    let tx = g.x - f32(x0);
    let ty = g.y - f32(y0);
    let s00 = foam_solid_cell(x0, y0);
    let s10 = foam_solid_cell(x1, y0);
    let s01 = foam_solid_cell(x0, y1);
    let s11 = foam_solid_cell(x1, y1);
    return mix(mix(s00, s10, tx), mix(s01, s11, tx), ty);
}

// Sample ambient open-water velocity outside the bubble so the ring average
// recovers U_inf with zero self-induced dipole feedback (2D harmonic mean-value theorem).
fn foam_ambient_flow(pos: vec2<f32>) -> vec3<f32> {
    let cx = clamp(i32(floor(pos.x)), 0, i32(COLUMNS) - 1);
    let cy = clamp(i32(floor(pos.y)), 0, i32(ROWS) - 1);
    let i = at(vec2<i32>(cx, cy));
    let c = cells[i];
    let solid = max(c.volume.fine, 0.) + max(c.volume.coarse, 0.) + max(c.bed.x, 0.) + max(c.bed.y, 0.);
    if c.volume.gas > 0.35 || solid > 0.35 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let vr = select(0., faces[2u * i].volume.w, cx < i32(COLUMNS) - 1);
    let vl = select(0., faces[2u * at(vec2<i32>(cx - 1, cy))].volume.w, cx > 0);
    let vu = select(0., faces[2u * i + 1u].volume.w, cy < i32(ROWS) - 1);
    let vd = select(0., faces[2u * at(vec2<i32>(cx, cy - 1)) + 1u].volume.w, cy > 0);
    return vec3<f32>(0.5 * (vl + vr), 0.5 * (vd + vu), 1.0);
}

// Signed distance (< 0 inside) and exact outward unit normal from point `p` to CCW bubble `bi`.
fn foam_sdf_bubble(p: vec2<f32>, bi: u32) -> vec3<f32> {
    var min_d2 = 1e20;
    var best_edge_n = vec2<f32>(0.0, 1.0);
    var best_diff = vec2<f32>(0.0, 1.0);
    var hit_interior = true;
    var crossings = 0i;
    for (var k = 0u; k < FOAM_VERTS; k++) {
        let a = foam[bi].verts[k];
        let b = foam[bi].verts[(k + 1u) % FOAM_VERTS];
        let e = b - a;
        let len2 = dot(e, e);
        let t = select(0.0, clamp(dot(p - a, e) / len2, 0.0, 1.0), len2 > 1e-10);
        let closest = a + t * e;
        let diff = p - closest;
        let d2 = dot(diff, diff);
        if d2 < min_d2 {
            min_d2 = d2;
            let e_len = sqrt(max(len2, 1e-10));
            best_edge_n = vec2<f32>(e.y, -e.x) / e_len;
            best_diff = diff;
            hit_interior = (t > 0.001 && t < 0.999);
        }
        if (a.y <= p.y && b.y > p.y) || (b.y <= p.y && a.y > p.y) {
            let x_int = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if p.x < x_int {
                crossings += 1;
            }
        }
    }
    let inside = (crossings & 1) != 0;
    let dist = sqrt(min_d2);
    let sd = select(dist, -dist, inside);
    var out_n = best_edge_n;
    if !hit_interior && dist > 1e-5 {
        let dir = best_diff / dist;
        out_n = select(dir, -dir, inside);
    }
    return vec3<f32>(sd, out_n.x, out_n.y);
}

fn foam_point_inside(p: vec2<f32>, bi: u32) -> bool {
    var crossings = 0i;
    for (var k = 0u; k < FOAM_VERTS; k++) {
        let a = foam[bi].verts[k];
        let b = foam[bi].verts[(k + 1u) % FOAM_VERTS];
        if (a.y <= p.y && b.y > p.y) || (b.y <= p.y && a.y > p.y) {
            let x_int = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if p.x < x_int {
                crossings += 1;
            }
        }
    }
    return (crossings & 1) != 0;
}

// 1. Advance every active closed-curve bubble by 8 x 1 ms substeps in shared memory.
@compute @workgroup_size(32)
fn foam_step(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bi = wg.x;
    let k = lid.x;
    if bi >= MAX_BUBBLES { return; }

    if k == 0u {
        sh_step_alive = foam[bi].alive;
    }
    if workgroupUniformLoad(&sh_step_alive) == 0u {
        if k == 0u {
            foam_next[bi] = foam[bi];
        }
        return;
    }

    let target_area = foam[bi].target_area;
    if k < FOAM_VERTS {
        sh_pos[k] = foam[bi].verts[k];
        sh_contact[k] = 0.0;
    }
    if k == 0u {
        sh_vel = foam[bi].vel;
        atomicStore(&sh_held, select(0u, 3u, foam[bi].held != 0u));
        let bmin = foam[bi].bbox_min - vec2<f32>(2.0);
        let bmax = foam[bi].bbox_max + vec2<f32>(2.0);
        var ncnt = 0u;
        for (var j = 0u; j < MAX_BUBBLES; j++) {
            if j == bi || foam[j].alive == 0u { continue; }
            let omin = foam[j].bbox_min;
            let omax = foam[j].bbox_max;
            if bmin.x <= omax.x && omin.x <= bmax.x && bmin.y <= omax.y && omin.y <= bmax.y {
                if ncnt < FOAM_MAX_NBRS {
                    sh_nbrs[ncnt] = j;
                    ncnt += 1u;
                }
            }
        }
        sh_nbr_count = ncnt;
    }
    workgroupBarrier();

    let dt_sub = cfg.dt / f32(FOAM_SUBSTEPS);
    let g_vec = vec2<f32>(cfg.lateral, -cfg.gravity);
    let g_mag = length(g_vec);
    let up_dir = select(vec2<f32>(0.0, 1.0), -g_vec / g_mag, g_mag > 1e-4);
    let r_eq = sqrt(max(target_area, 0.1) / 3.14159265);
    let v_rise = min(0.115, 0.042 * r_eq + 0.032) * (g_mag / STANDARD_GRAVITY);

    for (var sub = 0u; sub < FOAM_SUBSTEPS; sub++) {
        if k < FOAM_VERTS {
            let p_prev = sh_pos[(k + FOAM_VERTS - 1u) % FOAM_VERTS];
            let p_next = sh_pos[(k + 1u) % FOAM_VERTS];
            let tang = p_next - p_prev;
            let t_len = max(length(tang), 1e-6);
            let n_out = vec2<f32>(tang.y, -tang.x) / t_len;
            sh_uf[k] = foam_ambient_flow(sh_pos[k] + 1.2 * n_out);
        }
        workgroupBarrier();

        if k == 0u {
            var a2 = 0.0;
            var cx = 0.0;
            var cy = 0.0;
            var per = 0.0;
            var min_y = f32(ROWS);
            var max_y = 0.0;
            var uf_sum = vec2<f32>(0.0);
            var uf_w = 0.0;
            for (var m = 0u; m < FOAM_VERTS; m++) {
                let p0 = sh_pos[m];
                let p1 = sh_pos[(m + 1u) % FOAM_VERTS];
                min_y = min(min_y, p0.y);
                max_y = max(max_y, p0.y);
                let cr = p0.x * p1.y - p1.x * p0.y;
                a2 += cr;
                cx += (p0.x + p1.x) * cr;
                cy += (p0.y + p1.y) * cr;
                per += length(p1 - p0);
                uf_sum += sh_uf[m].xy;
                uf_w += sh_uf[m].z;
            }
            let area = max(0.5 * a2, 1e-4);
            sh_area = area;
            sh_perim = max(per, 1e-3);
            if abs(a2) > 1e-5 {
                sh_centroid = vec2<f32>(cx, cy) / (3.0 * a2);
            } else {
                var avg = vec2<f32>(0.0);
                for (var m = 0u; m < FOAM_VERTS; m++) { avg += sh_pos[m]; }
                sh_centroid = avg / f32(FOAM_VERTS);
            }
            let uf_mean = select(vec2<f32>(0.0), uf_sum / uf_w, uf_w > 0.5);
            let was_held = atomicLoad(&sh_held) >= 3u;
            let at_ceiling = (up_dir.y > 0.5 && max_y >= f32(ROWS) - 0.35)
                || (up_dir.y < -0.5 && min_y <= 0.35);
            var v_tgt = uf_mean + up_dir * select(0.55 * v_rise, 0.16 * v_rise, was_held);
            if was_held {
                // Capillary meniscus friction damps lateral sweep but still allows
                // held bubbles to creep up sloping sand roofs and shift in return
                // eddies so throats migrate, pinch shut, and reopen organically.
                v_tgt.x = 0.28 * uf_mean.x;
                let v_up = dot(v_tgt, up_dir);
                if v_up > 0.0 {
                    v_tgt -= up_dir * (0.35 * v_up);
                }
            } else if at_ceiling {
                // Impermeable top glass stops normal rise so ceiling bubbles keep 100% of
                // their true Young-Laplace volume instead of being squashed into the glass,
                // while retaining full lateral mobility along the smooth ceiling.
                let v_up = dot(v_tgt, up_dir);
                if v_up > 0.0 {
                    v_tgt -= up_dir * v_up;
                }
            }
            let relax = 1.0 - exp(-dt_sub / 0.012);
            sh_vel = clamp(mix(sh_vel, v_tgt, relax), vec2<f32>(-0.22), vec2<f32>(0.22));
            atomicStore(&sh_held, 0u);
        }
        workgroupBarrier();

        // Hydrostatic Young-Laplace normal displacement + bulk advection.
        if k < FOAM_VERTS {
            let p_prev = sh_pos[(k + FOAM_VERTS - 1u) % FOAM_VERTS];
            let p_curr = sh_pos[k];
            let p_next = sh_pos[(k + 1u) % FOAM_VERTS];
            let tang = p_next - p_prev;
            let t_len = max(length(tang), 1e-6);
            let n_out = vec2<f32>(tang.y, -tang.x) / t_len;

            // Exact area-preserving Young-Laplace curvature balance in surfactant water:
            // small bubbles (< l_c) stay crisp round capillary pearls while larger
            // bubbles (> 2*l_c) naturally flatten into wide horizontal-bottom ribbons.
            let perim_m = max(sh_perim * cfg.h, 1e-4);
            let p_laplace = FOAM_SIGMA * (2.0 * 3.14159265 / perim_m);
            let p_hydro = -RHO.x * dot(g_vec, p_curr - sh_centroid) * cfg.h;
            let dn_total = clamp((dt_sub / FOAM_GAMMA) * (p_laplace + p_hydro) / cfg.h, -0.10, 0.10);

            // Blend local ambient fluid velocity softly so sand-plume return currents sway the raft.
            let uf_loc = select(sh_vel, sh_uf[k].xy, sh_uf[k].z > 0.5);
            let v_vert = mix(sh_vel, uf_loc, 0.16);
            let pred = p_curr + (v_vert / cfg.h) * dt_sub + dn_total * n_out;
            sh_pred[k] = pred;
            sh_tmp[k] = pred;
        }
        workgroupBarrier();

        // Semi-implicit curvature surface tension + arclength equalization (Jacobi on ring).
        let ds_m = (sh_perim / f32(FOAM_VERTS)) * cfg.h;
        let lambda = clamp((FOAM_SIGMA * dt_sub) / (FOAM_GAMMA * max(ds_m * ds_m, 1e-8)), 0.02, 8.0);
        let inv_denom = 1.0 / (1.0 + 2.0 * lambda);
        for (var it = 0u; it < FOAM_JACOBI; it++) {
            var next_p = vec2<f32>(0.0);
            if k < FOAM_VERTS {
                let l = sh_tmp[(k + FOAM_VERTS - 1u) % FOAM_VERTS];
                let r = sh_tmp[(k + 1u) % FOAM_VERTS];
                next_p = (sh_pred[k] + lambda * (l + r)) * inv_denom;
            }
            workgroupBarrier();
            if k < FOAM_VERTS {
                sh_tmp[k] = next_p;
            }
            workgroupBarrier();
        }

        // Contact projection: sand underside/bed, tank walls, and neighbor soap films.
        var held_k = 0u;
        if k < FOAM_VERTS {
            var p = sh_tmp[k];
            var contact = 0.0;

            // Sand & bed contact (snug conformation to the wavy sand underside).
            let phi_s = foam_solid_at(p);
            if phi_s > FOAM_SAND_LEVEL {
                let gx = foam_solid_at(p + vec2<f32>(0.45, 0.0)) - foam_solid_at(p - vec2<f32>(0.45, 0.0));
                let gy = foam_solid_at(p + vec2<f32>(0.0, 0.45)) - foam_solid_at(p - vec2<f32>(0.0, 0.45));
                let gp = vec2<f32>(gx, gy);
                let gplen = length(gp);
                // A bubble's buoyancy (~30 Pa) exceeds the submerged weight of a thin
                // <= 3-cell sand crust (~15 Pa) separating it from open water above
                // (a cavity or the final top water layer): punch straight through it
                // instead of corking the cavity from below.
                var thin_crust = false;
                if gplen > 0.03 && dot(gp / gplen, up_dir) > 0.30 {
                    for (var step_u = 1; step_u <= 4; step_u++) {
                        let probe = p + up_dir * f32(step_u);
                        if probe.y >= f32(ROWS) - 1.5 || probe.y <= 1.5 { break; }
                        if foam_solid_at(probe) < 0.15 {
                            thin_crust = true;
                            break;
                        }
                    }
                }
                if !thin_crust {
                    let push = min(0.28, (phi_s - FOAM_SAND_LEVEL) * 1.45);
                    if gplen > 0.03 {
                        let dir = -gp / gplen;
                        p += dir * push;
                        // Only count dense sand genuinely overhead (not a vertical crater wall
                        // or dilute falling veil) as anchoring the bubble against buoyancy.
                        if phi_s > 0.35 && dot(gp / gplen, up_dir) > 0.45 { held_k += 1u; }
                    } else {
                        // Deep inside uniform sand (e.g. mid-drainage flip): escape toward open water.
                        let esc = select(up_dir, -up_dir, p.y > 0.5 * f32(ROWS));
                        p += esc * 0.35;
                        if p.y > 0.5 * f32(ROWS) { held_k += 1u; }
                    }
                    contact = 1.0;
                }
            }

            // Neighbor bubble soap-film contact (flat shared walls + Plateau borders).
            let ncnt = sh_nbr_count;
            for (var m = 0u; m < ncnt; m++) {
                let nb = sh_nbrs[m];
                let omin = foam[nb].bbox_min - vec2<f32>(0.3);
                let omax = foam[nb].bbox_max + vec2<f32>(0.3);
                if p.x < omin.x || p.x > omax.x || p.y < omin.y || p.y > omax.y {
                    continue;
                }
                let res = foam_sdf_bubble(p, nb);
                if res.x < FOAM_FILM_GAP {
                    let pen = FOAM_FILM_GAP - res.x;
                    p += vec2<f32>(res.y, res.z) * (0.52 * pen);
                    contact = max(contact, 0.85);
                    if foam[nb].centroid.y > sh_centroid.y && foam[nb].held != 0u {
                        held_k += 3u;
                    }
                }
            }

            // Tank glass frame with thin wetting water film.
            // Smooth top glass stops upward rise without pinning bubbles laterally,
            // so post-flip clusters spread evenly along the ceiling instead of bunching.
            let min_p = vec2<f32>(FOAM_WALL_PAD, FOAM_WALL_PAD);
            let max_p = vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD);
            if p.x < min_p.x || p.x > max_p.x || p.y < min_p.y || p.y > max_p.y {
                p = clamp(p, min_p, max_p);
                contact = 1.0;
            }

            sh_pos[k] = p;
            sh_contact[k] = contact;
        }
        workgroupBarrier();

        // Exact area conservation via free-vertex-weighted Lagrange multiplier.
        if k == 0u {
            var a2 = 0.0;
            var free_p = 0.0;
            for (var m = 0u; m < FOAM_VERTS; m++) {
                let p0 = sh_pos[m];
                let p1 = sh_pos[(m + 1u) % FOAM_VERTS];
                a2 += p0.x * p1.y - p1.x * p0.y;
                let w = mix(1.0, 0.20, sh_contact[m]);
                free_p += w * length(p1 - p0);
            }
            sh_area = 0.5 * a2;
            sh_free_perim = max(free_p, 1e-3);
        }
        if held_k != 0u {
            atomicAdd(&sh_held, held_k);
        }
        workgroupBarrier();

        if k < FOAM_VERTS {
            var p = sh_pos[k];
            if sh_area < 0.22 * target_area {
                // Degenerate/pinched polygon guard: re-inflate cleanly around centroid
                // at exact target_area so pinched corner bubbles never invert or vanish.
                let theta = 2.0 * 3.14159265 * f32(k) / f32(FOAM_VERTS);
                let rx_safe = max(r_eq, 0.35);
                let ry_safe = max(target_area / (3.14159265 * rx_safe), 0.35);
                p = sh_centroid + vec2<f32>(rx_safe * cos(theta), ry_safe * sin(theta));
            } else {
                let p_prev = sh_pos[(k + FOAM_VERTS - 1u) % FOAM_VERTS];
                let p_next = sh_pos[(k + 1u) % FOAM_VERTS];
                let tang = p_next - p_prev;
                let n_out = vec2<f32>(tang.y, -tang.x) / max(length(tang), 1e-6);
                let w = mix(1.0, 0.20, sh_contact[k]);
                let delta = clamp((target_area - sh_area) / sh_free_perim, -0.25, 0.25);
                p = p + (delta * w) * n_out;
            }
            p = clamp(p, vec2<f32>(FOAM_WALL_PAD), vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD));
            sh_pos[k] = p;
        }
        workgroupBarrier();
    }

    // Write updated bubble to foam_next.
    if k < FOAM_VERTS {
        foam_next[bi].verts[k] = sh_pos[k];
    }
    if k == 0u {
        var bmin = vec2<f32>(f32(COLUMNS), f32(ROWS));
        var bmax = vec2<f32>(0.0, 0.0);
        var a2 = 0.0;
        var cx = 0.0;
        var cy = 0.0;
        for (var m = 0u; m < FOAM_VERTS; m++) {
            let p0 = sh_pos[m];
            let p1 = sh_pos[(m + 1u) % FOAM_VERTS];
            bmin = min(bmin, p0);
            bmax = max(bmax, p0);
            let cr = p0.x * p1.y - p1.x * p0.y;
            a2 += cr;
            cx += (p0.x + p1.x) * cr;
            cy += (p0.y + p1.y) * cr;
        }
        let cent = select(0.5 * (bmin + bmax), vec2<f32>(cx, cy) / (3.0 * a2), abs(a2) > 1e-5);
        foam_next[bi].target_area = target_area;
        foam_next[bi].drain_time = foam[bi].drain_time;
        foam_next[bi].slot = foam[bi].slot;
        foam_next[bi].alive = 1u;
        foam_next[bi].centroid = cent;
        foam_next[bi].vel = sh_vel;
        foam_next[bi].bbox_min = bmin;
        foam_next[bi].bbox_max = bmax;
        let is_held = select(0u, 1u, atomicLoad(&sh_held) >= 3u);
        foam_next[bi].drain_partner = foam[bi].drain_partner;
        foam_next[bi].held = is_held;
        let raw_disp = clamp(
            (cent - foam[bi].centroid) * cfg.h / max(cfg.dt, 1e-6),
            vec2<f32>(-0.15),
            vec2<f32>(0.15),
        );
        // Anchored bubbles are stationary impermeable obstacles (disp_vel = 0) so
        // sub-cell contact projection jitter never rings into the streamfunction solve.
        // Freely rising bubbles use smoothed displacement velocity for their wake.
        foam_next[bi].disp_vel = select(
            mix(foam[bi].disp_vel, raw_disp, 0.2),
            vec2<f32>(0.0),
            is_held != 0u,
        );
    }
}

// 2. Vertical-stack film drainage coalescence (2.0 s) and conflict-free slot coloring,
// then commit foam_next -> foam.
@compute @workgroup_size(128)
fn foam_coalesce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Load all bubble bboxes into workgroup shared memory once for fast pair checks & coloring.
    if tid < MAX_BUBBLES {
        sh_alive[tid] = foam_next[tid].alive;
        sh_bmin[tid] = foam_next[tid].bbox_min;
        sh_bmax[tid] = foam_next[tid].bbox_max;
    }
    workgroupBarrier();

    // Each thread detects whether bubble `tid` is pressed tightly against a coalescence
    // partner: either vertically underneath an upper partner (fast buoyancy film drainage)
    // or strongly compressed laterally against a right-hand neighbor (slower lateral fusion).
    if tid < MAX_BUBBLES && sh_alive[tid] != 0u {
        let a_min = sh_bmin[tid];
        let a_max = sh_bmax[tid];
        let a_cent = foam_next[tid].centroid;
        let wa = max(a_max.x - a_min.x, 0.5);
        let ha = max(a_max.y - a_min.y, 0.5);
        var best_j = 0xffffffffu;
        var best_score = 0.0;
        var drain_rate = 1.0;
        for (var j = 0u; j < MAX_BUBBLES; j++) {
            if j == tid || sh_alive[j] == 0u { continue; }
            let b_min = sh_bmin[j];
            let b_max = sh_bmax[j];
            let b_cent = foam_next[j].centroid;
            let wb = max(b_max.x - b_min.x, 0.5);
            let hb = max(b_max.y - b_min.y, 0.5);
            // 1. Vertical stack (buoyancy presses lower bubble into upper bubble).
            if b_cent.y > a_cent.y + 0.45 {
                let vgap = b_min.y - a_max.y;
                if vgap <= 0.35 && vgap >= -0.8 {
                    let ov = min(a_max.x, b_max.x) - max(a_min.x, b_min.x);
                    if ov > 0.35 * min(wa, wb) && ov > best_score {
                        best_score = ov;
                        best_j = j;
                        drain_rate = 1.0;
                    }
                }
            }
            // 2. Strongly compressed lateral pair under sand (forms wide flat air ribbons).
            if b_cent.x > a_cent.x && abs(b_cent.y - a_cent.y) < 0.65 {
                let hgap = b_min.x - a_max.x;
                let vov = min(a_max.y, b_max.y) - max(a_min.y, b_min.y);
                if hgap <= -0.04 && hgap >= -0.6 && vov > 0.65 * min(ha, hb) {
                    let score = 0.35 * vov;
                    if score > best_score {
                        best_score = score;
                        best_j = j;
                        drain_rate = 0.16; // ~15 s under sustained lateral compression
                    }
                }
            }
        }
        if best_j != 0xffffffffu && best_j == foam_next[tid].drain_partner {
            foam_next[tid].drain_time = foam_next[tid].drain_time + drain_rate * cfg.dt;
        } else {
            foam_next[tid].drain_partner = best_j;
            foam_next[tid].drain_time = 0.0;
        }
    }
    workgroupBarrier();

    // Thread 0 performs size-dependent film-drainage coalescence, Rayleigh-Taylor
    // breakup of over-wide rising ribbons after a flip, and slot coloring.
    if tid == 0u {
        // A. Film-drainage coalescence (Reynolds/Stefan law): deformable bubbles
        // (area >= 2.0 cells^2) pressed together for >= 2.5 s merge into wider flat
        // horizontal air pockets (up to 26.0 cells^2 vertically, 18.0 laterally so
        // repeated flips cannot drift into a wall-to-wall ribbon seal), while small
        // rigid capillary pearls (< 2.0 cells^2) resist rupture and remain spacers.
        for (var i = 0u; i < MAX_BUBBLES; i++) {
            if foam_next[i].alive == 0u || foam_next[i].held == 0u || foam_next[i].drain_time < 2.5 { continue; }
            let j = foam_next[i].drain_partner;
            if j >= MAX_BUBBLES || j == i || foam_next[j].alive == 0u || foam_next[j].held == 0u { continue; }
            let ai = foam_next[i].target_area;
            let aj = foam_next[j].target_area;
            if min(ai, aj) < 2.0 { continue; }
            let tot = ai + aj;
            let is_vert = foam_next[j].centroid.y > foam_next[i].centroid.y + 0.45;
            let max_tot = select(18.0, 26.0, is_vert);
            if tot > max_tot { continue; }
            // Merge lower bubble i into upper bubble j, conserving exact combined area.
            let cent = (ai * foam_next[i].centroid + aj * foam_next[j].centroid) / max(tot, 1e-4);
            var merged: array<vec2<f32>, FOAM_VERTS>;
            for (var k = 0u; k < FOAM_VERTS; k++) {
                let theta = 2.0 * 3.14159265 * f32(k) / f32(FOAM_VERTS);
                let dir = vec2<f32>(cos(theta), sin(theta));
                var max_r = 0.25;
                for (var m = 0u; m < FOAM_VERTS; m++) {
                    let di = foam_next[i].verts[m] - cent;
                    let proj_i = dot(di, dir);
                    let perp_i = abs(di.x * dir.y - di.y * dir.x);
                    if proj_i > 0.0 && perp_i < 0.85 * proj_i + 0.5 {
                        max_r = max(max_r, proj_i);
                    }
                    let dj = foam_next[j].verts[m] - cent;
                    let proj_j = dot(dj, dir);
                    let perp_j = abs(dj.x * dir.y - dj.y * dir.x);
                    if proj_j > 0.0 && perp_j < 0.85 * proj_j + 0.5 {
                        max_r = max(max_r, proj_j);
                    }
                }
                merged[k] = clamp(
                    cent + dir * max_r,
                    vec2<f32>(FOAM_WALL_PAD),
                    vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD),
                );
            }
            var a2 = 0.0;
            for (var k = 0u; k < FOAM_VERTS; k++) {
                let p0 = merged[k];
                let p1 = merged[(k + 1u) % FOAM_VERTS];
                a2 += p0.x * p1.y - p1.x * p0.y;
            }
            let cur_a = max(0.5 * a2, 1e-4);
            let scale = clamp(sqrt(tot / cur_a), 0.7, 1.4);
            var bmin = vec2<f32>(f32(COLUMNS), f32(ROWS));
            var bmax = vec2<f32>(0.0, 0.0);
            for (var k = 0u; k < FOAM_VERTS; k++) {
                let p = clamp(
                    cent + (merged[k] - cent) * scale,
                    vec2<f32>(FOAM_WALL_PAD),
                    vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD),
                );
                foam_next[j].verts[k] = p;
                bmin = min(bmin, p);
                bmax = max(bmax, p);
            }
            foam_next[j].target_area = tot;
            foam_next[j].centroid = cent;
            foam_next[j].vel = (ai * foam_next[i].vel + aj * foam_next[j].vel) / max(tot, 1e-4);
            foam_next[j].disp_vel = vec2<f32>(0.0);
            foam_next[j].bbox_min = bmin;
            foam_next[j].bbox_max = bmax;
            foam_next[j].drain_time = 0.0;
            foam_next[j].drain_partner = 0xffffffffu;
            foam_next[i].alive = 0u;
            foam_next[i].target_area = 0.0;
            foam_next[i].drain_time = 0.0;
            foam_next[i].disp_vel = vec2<f32>(0.0);
            sh_alive[i] = 0u;
            sh_bmin[j] = bmin;
            sh_bmax[j] = bmax;
        }

        // B. Rayleigh-Taylor / flip breakup: an over-wide freely-rising air ribbon
        // (exceeding the Hele-Shaw splitting scale ~5.8 cells wide, area > 16.5 cells^2)
        // pinches off asymmetrically into unequal daughter bubbles whenever a free slot
        // exists, preserving wide flat cushions alongside crisp smaller satellite beads.
        var splits = 0u;
        for (var j = 0u; j < MAX_BUBBLES && splits < 4u; j++) {
            if sh_alive[j] == 0u || foam_next[j].held != 0u { continue; }
            let area = foam_next[j].target_area;
            let w_bub = sh_bmax[j].x - sh_bmin[j].x;
            // Only split once rising freely below the ceiling.
            if area <= 16.5 || w_bub <= 5.80 || sh_bmax[j].y >= f32(ROWS) - 1.2 { continue; }
            var free_slot = 0xffffffffu;
            for (var s = 0u; s < MAX_BUBBLES; s++) {
                if sh_alive[s] == 0u {
                    free_slot = s;
                    break;
                }
            }
            if free_slot == 0xffffffffu { break; }
            // Deterministic off-center necking ratio in {0.26, 0.36, 0.45, 0.62}
            // so pinch-off continuously replenishes small round satellite pearls
            // alongside medium cushions and wide horizontal ribbons.
            let phase = (j * 3u + splits) & 3u;
            var frac = 0.45;
            if phase == 0u { frac = 0.26; }
            else if phase == 1u { frac = 0.36; }
            else if phase == 3u { frac = 0.62; }
            let a_left = frac * area;
            let a_right = area - a_left;
            let cent = foam_next[j].centroid;
            let rx_l_init = clamp(0.42 * frac * w_bub, 0.45, 2.0);
            let ry_l = clamp(a_left / (3.14159265 * rx_l_init), 0.45, 1.8);
            let rx_l = a_left / (3.14159265 * ry_l);
            let rx_r_init = clamp(0.42 * (1.0 - frac) * w_bub, 0.45, 2.0);
            let ry_r = clamp(a_right / (3.14159265 * rx_r_init), 0.45, 1.8);
            let rx_r = a_right / (3.14159265 * ry_r);
            let sep = rx_l + rx_r + 0.12;
            let c_left = clamp(
                cent - vec2<f32>((1.0 - frac) * sep, 0.0),
                vec2<f32>(FOAM_WALL_PAD + rx_l, FOAM_WALL_PAD + ry_l),
                vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD - rx_l, f32(ROWS) - FOAM_WALL_PAD - ry_l),
            );
            let c_right = clamp(
                cent + vec2<f32>(frac * sep, 0.0),
                vec2<f32>(FOAM_WALL_PAD + rx_r, FOAM_WALL_PAD + ry_r),
                vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD - rx_r, f32(ROWS) - FOAM_WALL_PAD - ry_r),
            );
            var bmin_l = vec2<f32>(f32(COLUMNS), f32(ROWS));
            var bmax_l = vec2<f32>(0.0);
            var bmin_r = vec2<f32>(f32(COLUMNS), f32(ROWS));
            var bmax_r = vec2<f32>(0.0);
            for (var k = 0u; k < FOAM_VERTS; k++) {
                let theta = 2.0 * 3.14159265 * f32(k) / f32(FOAM_VERTS);
                let off_l = vec2<f32>(rx_l * cos(theta), ry_l * sin(theta));
                let off_r = vec2<f32>(rx_r * cos(theta), ry_r * sin(theta));
                let pl = clamp(c_left + off_l, vec2<f32>(FOAM_WALL_PAD), vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD));
                let pr = clamp(c_right + off_r, vec2<f32>(FOAM_WALL_PAD), vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD));
                foam_next[j].verts[k] = pl;
                foam_next[free_slot].verts[k] = pr;
                bmin_l = min(bmin_l, pl);
                bmax_l = max(bmax_l, pl);
                bmin_r = min(bmin_r, pr);
                bmax_r = max(bmax_r, pr);
            }
            foam_next[j].target_area = a_left;
            foam_next[j].centroid = c_left;
            foam_next[j].bbox_min = bmin_l;
            foam_next[j].bbox_max = bmax_l;
            foam_next[j].drain_time = 0.0;
            foam_next[j].drain_partner = 0xffffffffu;
            foam_next[j].disp_vel = vec2<f32>(0.0);
            foam_next[free_slot].target_area = a_right;
            foam_next[free_slot].drain_time = 0.0;
            foam_next[free_slot].slot = (foam_next[j].slot + 1u) % GAS_SLOTS;
            foam_next[free_slot].alive = 1u;
            foam_next[free_slot].centroid = c_right;
            foam_next[free_slot].vel = foam_next[j].vel;
            foam_next[free_slot].bbox_min = bmin_r;
            foam_next[free_slot].bbox_max = bmax_r;
            foam_next[free_slot].drain_partner = 0xffffffffu;
            foam_next[free_slot].held = 0u;
            foam_next[free_slot].disp_vel = vec2<f32>(0.0);
            sh_alive[free_slot] = 1u;
            sh_bmin[j] = bmin_l;
            sh_bmax[j] = bmax_l;
            sh_bmin[free_slot] = bmin_r;
            sh_bmax[free_slot] = bmax_r;
            splits += 1u;
        }

        // Refresh greedy slot coloring in shared memory so touching bubbles keep distinct GAS_SLOTS.
        let margin = 1.8;
        for (var i = 0u; i < MAX_BUBBLES; i++) {
            if sh_alive[i] == 0u { continue; }
            var used = 0u;
            let imin = sh_bmin[i];
            let imax = sh_bmax[i];
            for (var j = 0u; j < i; j++) {
                if sh_alive[j] == 0u { continue; }
                let jmin = sh_bmin[j];
                let jmax = sh_bmax[j];
                if imin.x <= jmax.x + margin && jmin.x <= imax.x + margin
                    && imin.y <= jmax.y + margin && jmin.y <= imax.y + margin {
                    used |= 1u << foam_next[j].slot;
                }
            }
            let cur = foam_next[i].slot;
            if (used & (1u << cur)) != 0u {
                for (var s = 0u; s < GAS_SLOTS; s++) {
                    if (used & (1u << s)) == 0u {
                        foam_next[i].slot = s;
                        break;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // Commit ping-ponged bubbles to `foam` and record component areas in `labels`.
    if tid < MAX_BUBBLES {
        foam[tid] = foam_next[tid];
        let area_u = select(0u, u32(round(foam_next[tid].target_area * LABEL_AREA_UNIT)), foam_next[tid].alive != 0u);
        atomicStore(&labels[label_compact(tid)], tid);
        atomicStore(&labels[label_field(tid, 2u)], area_u);
    }
}

// 3. Rasterize closed-curve bubbles into main's grid buffers (`cells`, `geometry`, `labels`)
// without ever modifying sand (`fine`, `coarse`, `bed`).
// Uses workgroup shared memory so all 128 bubble bounding boxes are loaded once per workgroup.
@compute @workgroup_size(128)
fn foam_raster(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    sh_alive[lane] = foam[lane].alive;
    sh_bmin[lane] = foam[lane].bbox_min;
    sh_bmax[lane] = foam[lane].bbox_max;
    workgroupBarrier();

    let i = id.x;
    if i >= cfg.n { return; }
    if i < MAX_BUBBLES {
        let area_u = select(0u, u32(round(foam[i].target_area * LABEL_AREA_UNIT)), sh_alive[i] != 0u);
        atomicStore(&labels[label_compact(i)], i);
        atomicStore(&labels[label_field(i, 2u)], area_u);
    }
    let p_i = point(i);
    let x0 = f32(p_i.x);
    let y0 = f32(p_i.y);
    let center = vec2<f32>(x0 + 0.5, y0 + 0.5);

    var slot_frac: array<f32, GAS_SLOTS>;
    var slot_norm: array<vec2<f32>, GAS_SLOTS>;
    var slot_bub: array<u32, GAS_SLOTS>;
    for (var l = 0u; l < GAS_SLOTS; l++) {
        slot_frac[l] = 0.0;
        slot_norm[l] = vec2<f32>(0.0, 1.0);
        slot_bub[l] = NO_LABEL;
    }

    var total_gas = 0.0;
    var gas_mom = vec2<f32>(0.0);

    for (var bi = 0u; bi < MAX_BUBBLES; bi++) {
        if sh_alive[bi] == 0u { continue; }
        let bmin = sh_bmin[bi];
        let bmax = sh_bmax[bi];
        if x0 + 1.0 < bmin.x || x0 > bmax.x || y0 + 1.0 < bmin.y || y0 > bmax.y {
            continue;
        }
        let res = foam_sdf_bubble(center, bi);
        let n_unit = vec2<f32>(res.y, res.z);
        let frac = rectangle_fraction(n_unit, -res.x, vec2<f32>(-0.5), vec2<f32>(0.5));
        if frac <= 1e-4 { continue; }
        let s = foam[bi].slot;
        if frac > slot_frac[s] {
            slot_norm[s] = n_unit;
            slot_bub[s] = bi;
        }
        slot_frac[s] += frac;
        total_gas += frac;
        gas_mom += frac * foam[bi].disp_vel;
    }

    // Clamp total gas to available open-water room so sand conservation is exact
    // and hydrophilic silica grains always retain their irreducible saturated pore
    // water at BED_PACKING (capillary entry pressure ~5 kPa >> bubble buoyancy ~20 Pa):
    // air cannot dry out wet sand above BED_PACKING or crush cell water to zero.
    var c = cells[i];
    let mobile_solid = max(c.volume.fine, 0.0) + max(c.volume.coarse, 0.0);
    let bed_occ = max(c.bed.x, 0.0) + max(c.bed.y, 0.0) + max(c.bed.w, 0.0);
    let room = max(1.0 - mobile_solid - bed_occ, 0.0);
    let gas_room = max(1.0 - mobile_solid / BED_PACKING - bed_occ, 0.0);
    var scale = 1.0;
    if total_gas > gas_room && total_gas > 1e-6 {
        scale = gas_room / total_gas;
        total_gas = gas_room;
        gas_mom *= scale;
    }

    var held_mask = 0u;
    var anchor_mask = 0u;
    var c0_f = 0.0;
    var c0_shape = vec4<f32>(0.0);
    var c1_f = 0.0;
    var c1_shape = vec4<f32>(0.0);
    for (var l = 0u; l < GAS_SLOTS; l++) {
        let f = slot_frac[l] * scale;
        let keep_w = geometry[layer_index(i, l)].w;
        if f > 1e-5 {
            let n = slot_norm[l];
            let l1_sum = abs(n.x) + abs(n.y);
            let n_l1 = select(vec2<f32>(0.0, 1.0), n / l1_sum, l1_sum > 1e-6);
            let alpha = intercept(n_l1, clamp(f, 0.0, 1.0));
            let plane = vec4<f32>(n_l1, alpha, 0.0);
            geometry[layer_index(i, l)] = vec4<f32>(f, f32(l + 1u), 0.0, keep_w);
            geometry[shape_index(i, l)] = plane;
            if f > c0_f {
                c1_f = c0_f;
                c1_shape = c0_shape;
                c0_f = f;
                c0_shape = plane;
            } else if f > c1_f {
                c1_f = f;
                c1_shape = plane;
            }
            atomicStore(&labels[label_node(i, l)], slot_bub[l]);
            held_mask |= 1u << l;
            if slot_bub[l] < MAX_BUBBLES && foam[slot_bub[l]].held != 0u {
                anchor_mask |= 1u << l;
            }
        } else {
            geometry[layer_index(i, l)] = vec4<f32>(0.0, f32(l + 1u), 0.0, keep_w);
            geometry[shape_index(i, l)] = vec4<f32>(0.0);
            atomicStore(&labels[label_node(i, l)], NO_LABEL);
        }
    }
    // Populate CONTACT_ROW with the two largest bubbles' exact planes so curved_passage,
    // free_wet_aperture, grain_crosses, and slot_width seal bubble boundaries cleanly.
    geometry[contact_index(i, 0u)] = c0_shape;
    geometry[contact_index(i, 1u)] = c1_shape;

    let old_water = max(c.volume.water, 0.0);
    let new_water = max(room - total_gas, 0.0);
    // Keep water velocity (momentum[0] / water) bounded when a moving bubble displaces
    // water out of a cell; otherwise momentum/water spikes and shatters adjacent sand
    // into spurious free powder (glowing milky haze beneath bubbles).
    if old_water > 1e-6 {
        c.momentum[0] *= min(new_water / old_water, 1.0);
    } else {
        c.momentum[0] = vec2<f32>(0.0);
    }
    c.volume.gas = total_gas;
    c.volume.water = new_water;
    c.momentum[3] = gas_mom;
    cells[i] = c;

    let word = CELL_KEPT | (held_mask << CELL_HELD) | anchor_mask;
    atomicStore(&labels[label_cell_anchor(i)], word);
}

// 4. Exact 180-degree rotation of all closed curves when the picture mirrors.
@compute @workgroup_size(128)
fn invert_foam(@builtin(global_invocation_id) id: vec3<u32>) {
    let bi = id.x;
    if bi >= MAX_BUBBLES { return; }
    if foam[bi].alive == 0u { return; }
    let box_sz = vec2<f32>(f32(COLUMNS), f32(ROWS));
    var bmin = box_sz;
    var bmax = vec2<f32>(0.0);
    for (var k = 0u; k < FOAM_VERTS; k++) {
        let v = box_sz - foam[bi].verts[k];
        foam[bi].verts[k] = v;
        bmin = min(bmin, v);
        bmax = max(bmax, v);
    }
    foam[bi].centroid = box_sz - foam[bi].centroid;
    foam[bi].vel = -foam[bi].vel;
    foam[bi].disp_vel = vec2<f32>(0.0);
    foam[bi].bbox_min = bmin;
    foam[bi].bbox_max = bmax;
    foam[bi].drain_time = 0.0;
    foam[bi].drain_partner = 0xffffffffu;
    foam[bi].held = 0u;
    foam_next[bi] = foam[bi];
}

// Shared scratch for `foam_adjust_air` and `measure_sand`.
var<workgroup> sh_air_scale: f32;
var<workgroup> sh_sum_total: array<f32, 256>;
var<workgroup> sh_sum_bed: array<f32, 256>;
var<workgroup> sh_sum_above: array<f32, 256>;
var<workgroup> sh_sum_gas: array<f32, 256>;

fn foam_open_water_spot(cx: i32, cy: i32) -> bool {
    if cx < 3 || cx > i32(COLUMNS) - 4 || cy < 3 || cy > i32(ROWS) - 4 {
        return false;
    }
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let c = cells[at(vec2<i32>(cx + dx, cy + dy))];
            let solid = max(c.volume.fine, 0.0) + max(c.volume.coarse, 0.0)
                      + max(c.bed.x, 0.0) + max(c.bed.y, 0.0);
            if solid > 0.06 || c.volume.gas > 0.05 || c.bed.z > 0.0 {
                return false;
            }
        }
    }
    return true;
}

// 5. Dynamic air adjustment:
// - `cfg.air_delta < 0`: uniformly shrink every alive bubble's target_area (and vertices
//   toward its centroid), retiring tiny bubbles (< 0.70 cells^2).
// - `cfg.air_delta > 0`: inject fresh round bubbles into clear open water above the
//   deposited sand (`terrain[x].y + 6`) so they rise and join the raft; fall back to
//   proportional scaling if no free slots or open water pockets are available.
@compute @workgroup_size(128)
fn foam_adjust_air(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    if tid == 0u {
        var total = 0.0;
        for (var bi = 0u; bi < MAX_BUBBLES; bi++) {
            if foam[bi].alive != 0u {
                total += foam[bi].target_area;
            }
        }
        var scale = 1.0;
        if cfg.air_delta < 0.0 && total > 1.0 {
            let goal = max(total + cfg.air_delta, 120.0);
            scale = clamp(goal / total, 0.25, 1.0);
        } else if cfg.air_delta > 0.0 {
            var rem = cfg.air_delta;
            let count = 3u;
            let per_bub = max(rem / f32(count), 4.0);
            var spawned_x: array<i32, 3>;
            var spawned = 0u;
            for (var k = 0u; k < count && rem >= 2.0; k++) {
                // Find an unused bubble slot.
                var free_slot = 0xffffffffu;
                for (var bi = 0u; bi < MAX_BUBBLES; bi++) {
                    if foam[bi].alive == 0u {
                        free_slot = bi;
                        break;
                    }
                }
                if free_slot == 0xffffffffu {
                    break;
                }
                // Scan candidate columns across the tank for the lowest clear open-water
                // pocket above the bottom deposit (`terrain[x].y`).
                let base_x = 22 + i32((cfg.air_seed * 73u + k * 83u) % 226u);
                var best_x = -1;
                var best_y = i32(ROWS);
                for (var attempt = 0; attempt < 18; attempt++) {
                    let off = select(-(attempt / 2), (attempt + 1) / 2, (attempt & 1) == 1) * 11;
                    let cx = 12 + ((base_x - 12 + off) % 246 + 246) % 246;
                    var too_close = false;
                    for (var s = 0u; s < spawned; s++) {
                        if abs(cx - spawned_x[s]) < 7 {
                            too_close = true;
                        }
                    }
                    if too_close { continue; }
                    let tcol = terrain[u32(cx)];
                    let bed_top = select(0, i32(ceil(tcol.y)) + 1, tcol.x > 0.0);
                    let y_start = max(bed_top + 6, 6);
                    for (var cy = y_start; cy <= i32(ROWS) - 10; cy += 2) {
                        if foam_open_water_spot(cx, cy) {
                            if cy < best_y {
                                best_x = cx;
                                best_y = cy;
                            }
                            break;
                        }
                    }
                    if best_x >= 0 && best_y <= y_start + 8 {
                        break;
                    }
                }
                if best_x < 0 {
                    break;
                }
                let area = min(rem, per_bub);
                let r = sqrt(area / 3.14159265);
                let cent = vec2<f32>(f32(best_x) + 0.5, f32(best_y) + 0.5);
                var bmin = vec2<f32>(f32(COLUMNS), f32(ROWS));
                var bmax = vec2<f32>(0.0);
                for (var m = 0u; m < FOAM_VERTS; m++) {
                    let th = 2.0 * 3.14159265 * f32(m) / f32(FOAM_VERTS);
                    let p = clamp(
                        cent + r * vec2<f32>(cos(th), sin(th)),
                        vec2<f32>(FOAM_WALL_PAD),
                        vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD),
                    );
                    foam[free_slot].verts[m] = p;
                    bmin = min(bmin, p);
                    bmax = max(bmax, p);
                }
                foam[free_slot].target_area = area;
                foam[free_slot].drain_time = 0.0;
                foam[free_slot].slot = (free_slot + k) % GAS_SLOTS;
                foam[free_slot].alive = 1u;
                foam[free_slot].centroid = cent;
                foam[free_slot].vel = vec2<f32>(0.0, 0.04);
                foam[free_slot].bbox_min = bmin;
                foam[free_slot].bbox_max = bmax;
                foam[free_slot].drain_partner = 0xffffffffu;
                foam[free_slot].held = 0u;
                foam[free_slot].disp_vel = vec2<f32>(0.0);
                spawned_x[spawned] = best_x;
                spawned += 1u;
                rem -= area;
            }
            // Fallback: if slots or open water were scarce, scale existing bubbles slightly.
            if rem > 0.5 && total > 1.0 {
                scale = clamp((total + rem) / total, 1.0, 1.35);
            }
        }
        sh_air_scale = scale;
    }
    workgroupBarrier();

    if tid < MAX_BUBBLES {
        let scale = sh_air_scale;
        if foam[tid].alive != 0u && abs(scale - 1.0) > 1e-5 {
            let new_a = foam[tid].target_area * scale;
            if new_a < 0.70 {
                foam[tid].alive = 0u;
                foam[tid].target_area = 0.0;
                foam[tid].disp_vel = vec2<f32>(0.0);
            } else {
                let cent = foam[tid].centroid;
                let rs = sqrt(scale);
                var bmin = vec2<f32>(f32(COLUMNS), f32(ROWS));
                var bmax = vec2<f32>(0.0);
                for (var m = 0u; m < FOAM_VERTS; m++) {
                    let p = clamp(
                        cent + (foam[tid].verts[m] - cent) * rs,
                        vec2<f32>(FOAM_WALL_PAD),
                        vec2<f32>(f32(COLUMNS) - FOAM_WALL_PAD, f32(ROWS) - FOAM_WALL_PAD),
                    );
                    foam[tid].verts[m] = p;
                    bmin = min(bmin, p);
                    bmax = max(bmax, p);
                }
                foam[tid].target_area = new_a;
                foam[tid].bbox_min = bmin;
                foam[tid].bbox_max = bmax;
            }
        }
        foam_next[tid] = foam[tid];
    }
}

// 6. Single-workgroup summary metrics for auto-flip completion detection and live air readout.
@compute @workgroup_size(256)
fn measure_sand(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var s_tot = 0.0;
    var s_bed = 0.0;
    var s_above = 0.0;
    for (var i = tid; i < cfg.n; i += 256u) {
        let c = cells[i];
        let mob = max(c.volume.fine, 0.0) + max(c.volume.coarse, 0.0);
        let bed = max(c.bed.x, 0.0) + max(c.bed.y, 0.0);
        s_tot += mob + bed;
        s_bed += bed;
        let x = i % cfg.nx;
        let y = i / cfg.nx;
        let tcol = terrain[x];
        let bed_top = select(0.0, tcol.y + 1.0, tcol.x > 0.0);
        if f32(y) > bed_top + 6.0 {
            s_above += mob;
        }
    }
    var g_tot = 0.0;
    if tid < MAX_BUBBLES && foam[tid].alive != 0u {
        g_tot = foam[tid].target_area;
    }
    sh_sum_total[tid] = s_tot;
    sh_sum_bed[tid] = s_bed;
    sh_sum_above[tid] = s_above;
    sh_sum_gas[tid] = g_tot;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            sh_sum_total[tid] += sh_sum_total[tid + stride];
            sh_sum_bed[tid] += sh_sum_bed[tid + stride];
            sh_sum_above[tid] += sh_sum_above[tid + stride];
            sh_sum_gas[tid] += sh_sum_gas[tid + stride];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        metrics[0] = u32(round(sh_sum_total[0] * 1024.0));
        metrics[1] = u32(round(sh_sum_bed[0] * 1024.0));
        metrics[2] = u32(round(sh_sum_above[0] * 1024.0));
        metrics[3] = u32(round(sh_sum_gas[0] * 1024.0));
    }
}
