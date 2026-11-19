//! The coupled water and interface response, solved twice per update by
//! flexible conjugate gradients preconditioned with a multigrid V-cycle.
use super::DT;
use super::buffers::blank;
use super::grid::{CELLS, FLOW_ITERATIONS, cell_width};
use super::kernel::Plan;
use super::material::SURFACE_TENSION;
use super::shader::Program;
use super::update::Plans;
use wgpu::util::DeviceExt;

/// Levels with at most this many points are smoothed, restricted and
/// prolongated by one workgroup in a single dispatch (`coarse_cycle`): their
/// dispatches cost more than their work.
const COARSE_CYCLE_POINTS: u32 = 1024;

/// Coarse rows per workgroup of `coarsen` (its children's rows are shared).
pub const COARSEN_ROWS: u32 = 5;

/// Threads per workgroup of the kernels that serve one point a thread and
/// share nothing: the most a workgroup may hold, since a dispatch costs by
/// its workgroups as well as by its points.
pub const POINT_THREADS: u32 = 256;

/// The multigrid's levels as (points across, points up, offset in the
/// solver's arrays), from the grid's vertices down.
fn levels() -> Vec<(u32, u32, u32)> {
    let [nx, ny] = CELLS.map(|v| v as u32);
    let mut sizes = vec![(nx + 1, ny + 1, 0)];
    while sizes.last().unwrap().0 * sizes.last().unwrap().1 > 112 {
        let &(x, y, offset) = sizes.last().unwrap();
        sizes.push((x.div_ceil(2), y.div_ceil(2), offset + x * y));
    }
    sizes
}

/// The first level the coarse cycle serves: the first small enough, and the
/// coarsest always is.
fn cycled(sizes: &[(u32, u32, u32)]) -> usize {
    sizes
        .iter()
        .position(|&(x, y, _)| x * y <= COARSE_CYCLE_POINTS)
        .unwrap_or(sizes.len() - 1)
}

/// The flow solver's fields.
pub struct Fields {
    /// The streamfunction, coarse levels included: the finest level is the
    /// next update's first guess.
    pub solution: wgpu::Buffer,
    /// A row's 5x5 stencil and its right-hand side, every level.
    rows: wgpu::Buffer,
    residual: wgpu::Buffer,
    anchors: wgpu::Buffer,
    /// The interfaces' tension as a tangent quadratic, per finest vertex.
    tensor: wgpu::Buffer,
    /// The conjugate gradients' vectors and sums.
    work: wgpu::Buffer,
    /// The levels of the coarse cycle, as it reads them.
    hierarchy: wgpu::Buffer,
}
impl Fields {
    /// Every field, for a reset: all start as zeros.
    pub fn all(&self) -> [&wgpu::Buffer; 6] {
        [
            &self.solution,
            &self.rows,
            &self.residual,
            &self.anchors,
            &self.tensor,
            &self.work,
        ]
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let sizes = levels();
        let &(x, y, offset) = sizes.last().unwrap();
        let total = u64::from(offset + x * y);
        let vertices = sizes[0].0 * sizes[0].1;
        Self {
            solution: blank(device, "streamfunction", total * 4),
            rows: blank(device, "coupled circulation energy", total * 26 * 4),
            residual: blank(device, "flow residual and initial inverse", total * 4),
            anchors: blank(
                device,
                "anchoring of each finest row",
                u64::from(vertices) * 4,
            ),
            tensor: blank(
                device,
                "interface tangent quadratic",
                u64::from(vertices) * 16 * 10,
            ),
            work: blank(
                device,
                "flexible conjugate gradient vectors",
                u64::from(6 * vertices + 2 * vertices.div_ceil(128) + 8) * 4,
            ),
            hierarchy: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("levels of the coarse cycle"),
                contents: &sizes[cycled(&sizes)..]
                    .iter()
                    .flat_map(|&(x, y, offset)| [x, y, offset, 0])
                    .chain([0; 4])
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        }
    }

    /// The field that the flow solver's program calls `name`.
    pub fn get(&self, name: &str) -> &wgpu::Buffer {
        match name {
            "rows" => &self.rows,
            "residual" => &self.residual,
            "anchors" => &self.anchors,
            "tensor" => &self.tensor,
            "fcg" => &self.work,
            "hierarchy" => &self.hierarchy,
            other => panic!("the model has no buffer {other}"),
        }
    }
}

/// A level smoothed by dispatches of its own, above the coarse cycle.
struct Level {
    smooth: Vec<usize>,
    residual: usize,
    restrict: usize,
    prolong: usize,
}

/// The two flow solves of an update.
#[derive(Clone, Copy)]
pub enum Solve {
    First,
    Second,
}

pub struct Stream {
    assemble: usize,
    assemble_cold: usize,
    scale: usize,
    anchor_plan: usize,
    anchor_predict: usize,
    anchor_apply: usize,
    interface: usize,
    coarsen: Vec<usize>,
    levels: Vec<Level>,
    coarse_cycle: usize,
    fcg_start: [usize; 2],
    fcg_iterate: [usize; 5],
    fcg_finish: usize,
}

impl Stream {
    pub fn new(plans: &mut Plans) -> Self {
        let [nx, ny] = CELLS.map(|v| v as u32);
        let h = cell_width();
        let sizes = levels();
        // A kernel's own configuration: the dispatched level, the next
        // coarser one, the grid, the kernel's colour and the capillary
        // coefficient dt^2 sigma / (rho h^3).
        let config = |level: usize, color: u32| {
            let (x, y, offset) = sizes[level];
            let (cx, cy, coarse) = sizes.get(level + 1).copied().unwrap_or((0, 0, 0));
            [
                x,
                y,
                offset,
                cx,
                cy,
                coarse,
                nx,
                ny,
                h.to_bits(),
                color,
                (DT * DT * SURFACE_TENSION / (997. * h * h * h)).to_bits(),
                9,
            ]
        };
        let build = |level: usize, color, entry, groups| {
            Plan::new(Program::Stream, entry, groups).with_config(&config(level, color))
        };
        let coarsen = sizes
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                plans.add(build(
                    i,
                    0,
                    "coarsen",
                    (pair[1].0 * pair[1].1).div_ceil(COARSEN_ROWS),
                ))
            })
            .collect();
        let cycled = cycled(&sizes);
        let levels = sizes[..cycled]
            .iter()
            .enumerate()
            .map(|(i, &(x, y, _))| {
                let (cx, cy, _) = sizes[i + 1];
                Level {
                    smooth: (0..9)
                        .map(|c| {
                            plans.add(build(
                                i,
                                c,
                                "relax",
                                (x.div_ceil(3) * y.div_ceil(3)).div_ceil(POINT_THREADS),
                            ))
                        })
                        .collect(),
                    residual: plans.add(build(
                        i,
                        0,
                        "evaluate_residual",
                        (x * y).div_ceil(POINT_THREADS),
                    )),
                    restrict: plans.add(build(
                        i,
                        0,
                        "restrict_rhs",
                        (cx * cy).div_ceil(POINT_THREADS),
                    )),
                    prolong: plans.add(build(i, 0, "prolong", (x * y).div_ceil(POINT_THREADS))),
                }
            })
            .collect();
        let (x, y, offset) = sizes[cycled];
        let (cx, cy, coarse) = sizes.get(cycled + 1).copied().unwrap_or((0, 0, 0));
        let coarse_cycle = plans.add(Plan::new(Program::Stream, "coarse_cycle", 1).with_config(&[
            x,
            y,
            offset,
            cx,
            cy,
            coarse,
            nx,
            ny,
            h.to_bits(),
            (sizes.len() - 1 - cycled) as u32,
            0,
            9,
        ]));
        let vertices = (nx + 1) * (ny + 1);
        // The conjugate gradients' sums are gathered by workgroups of 128.
        let groups = vertices.div_ceil(128);
        let points = vertices.div_ceil(POINT_THREADS);
        let (x0, y0, _) = sizes[0];
        let (cx0, cy0, coarse0) = sizes.get(1).copied().unwrap_or((0, 0, 0));
        let finest = [x0, y0, 0, cx0, cy0, coarse0, nx, ny, h.to_bits(), 0, 0, 9];
        let fcg = |entry, groups| Plan::new(Program::Stream, entry, groups).with_config(&finest);
        Self {
            fcg_start: [
                plans.add(fcg("fcg_start", groups)),
                plans.add(fcg("fcg_residual", groups)),
            ],
            fcg_iterate: [
                plans.add(fcg("fcg_dots_z", groups)),
                plans.add(fcg("fcg_beta", 1)),
                plans.add(fcg("fcg_direction", groups)),
                plans.add(fcg("fcg_matvec", groups)),
                plans.add(fcg("fcg_update", groups)),
            ],
            fcg_finish: plans.add(fcg("fcg_finish", groups)),
            assemble: plans.add(build(0, 0, "assemble", points)),
            assemble_cold: plans.add(build(0, 1, "assemble", points)),
            scale: plans.add(build(0, 0, "scale_interpolation", points)),
            anchor_plan: plans.add(build(0, 0, "anchor_plan", points)),
            anchor_predict: plans.add(build(0, 0, "anchor_predict", points)),
            anchor_apply: plans.add(build(0, 0, "anchor_apply", points)),
            // The tension tensor reads the gas by the relabelling's words.
            interface: plans.add(
                Plan::new(Program::Stream, "interface_tensor", points).with_config(&config(0, 0)),
            ),
            coarsen,
            levels,
            coarse_cycle,
        }
    }

    /// Capillary tangent tensor and its gradient basis from the current gas.
    pub fn tensor(&self) -> Vec<usize> {
        vec![self.interface]
    }

    /// Finest rows and right-hand side. A cold assembly also discards the
    /// algebraic warm start: geometry can close a passage between the two
    /// solves, and the old circulation then has a very large energy in the
    /// new operator. The physical inertial load is kept either way.
    pub fn assemble(&self, cold: bool) -> Vec<usize> {
        vec![if cold {
            self.assemble_cold
        } else {
            self.assemble
        }]
    }

    /// Galerkin operators of every coarser level.
    pub fn coarsen(&self) -> Vec<usize> {
        [vec![self.scale], self.coarsen.clone()].concat()
    }

    /// Flexible conjugate gradients from the current streamfunction; each
    /// iteration preconditions its residual with one V-cycle from zero. The
    /// rows' anchoring is found once per solve.
    pub fn solve(&self, solve: Solve) -> Vec<usize> {
        let mut kernels = vec![self.anchor_plan];
        kernels.extend(self.fcg_start);
        for _ in 0..FLOW_ITERATIONS[solve as usize] {
            kernels.extend(self.cycle());
            kernels.extend(self.fcg_iterate);
        }
        kernels.push(self.fcg_finish);
        kernels
    }

    fn cycle(&self) -> Vec<usize> {
        let mut kernels = vec![self.anchor_predict, self.anchor_apply];
        for level in &self.levels {
            kernels.extend(&level.smooth);
            kernels.push(level.residual);
            kernels.push(level.restrict);
        }
        kernels.push(self.coarse_cycle);
        for level in self.levels.iter().rev() {
            kernels.push(level.prolong);
            kernels.extend(level.smooth.iter().rev());
        }
        kernels
    }
}
