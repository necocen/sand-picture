//! Conservative water, two granular species and separately owned gas domains.
//! The fixed-cost approximation: 8 ms steps, two coupled flow cycles,
//! adaptive conservative transport, curved clearance and interspecies contact.
//! `Model` holds the fields and the kernels, records one update as its list of
//! steps, and turns the picture between updates.
mod bed;
mod buffers;
mod grid;
mod interface;
mod kernel;
mod labels;
mod material;
mod mechanics;
mod scene;
mod shader;
mod slab;
mod stream;
mod transport;
mod update;

use bed::Bed;
pub use buffers::Buffers;
pub use grid::{CELLS, TANK};
use interface::Interface;
use labels::Labels;
use material::Grains;
use mechanics::Mechanics;
pub use scene::Scene;
use shader::{Program, Shaders};
use slab::Slab;
use stream::Stream;
use transport::Transport;
use update::{Kernels, Plans};
pub use update::{Stage, Step};

/// Seconds per update: the single source for the shaders and the tools.
pub const DT: f32 = 0.008;
/// The gravity at which the material table's settling speeds are taken: the
/// in-plane gravity while the picture stands.
const STANDARD_GRAVITY: f32 = 9.80665;
/// Seconds a person takes to turn the picture over, tipping it front to back
/// about its horizontal axis as on a pivot stand.
const TURN_SECONDS: f32 = 2.;
/// Before turning it the person shakes the picture left and right for a
/// moment, which gathers the sand on the floor and breaks the air into small
/// bubbles (the makers' instructions): seconds of shaking, its frequency (Hz)
/// and the tank's displacement amplitude (m), a hand's motion.
const SHAKE_SECONDS: f32 = 3.;
const SHAKE_FREQUENCY: f32 = 3.;
const SHAKE_DISPLACEMENT: f32 = 0.02;
/// The gas identities a cell can hold; `GAS_SLOTS` of geometry_rows.wgsl
/// must agree with it.
const GAS_SLOTS: usize = 9;

pub struct Model {
    buffers: Buffers,
    kernels: Kernels,
    steps: Vec<Step>,
    /// The kernels that prepare the first update from the scene.
    prepare: Vec<usize>,
    /// The kernels that mirror the grid when a turn passes the flat.
    invert: Vec<usize>,
    turn: Option<Turn>,
}
/// A turn of the picture in progress: the seconds since it began, and whether
/// the grid has been mirrored (when the picture lay flat).
struct Turn {
    elapsed: f32,
    mirrored: bool,
}
impl Model {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) -> Self {
        let grains = Grains::default();
        let buffers = Buffers::new(device, queue, scene, &grains);
        let shaders = Shaders::new(&grains);
        let mut plans = Plans::default();
        let slab = Slab::new(&mut plans, &buffers);
        let bed = Bed::new(&mut plans, &buffers);
        let mechanics = Mechanics::new(&mut plans, &buffers);
        let stream = Stream::new(&mut plans, device, &buffers);
        let transport = Transport::new(&mut plans, &buffers);
        let labels = Labels::new(&mut plans, &buffers);
        let interface = Interface::new(&mut plans, &buffers);
        let mirrored = [
            (0, &buffers.config),
            (1, &buffers.cells),
            (4, &buffers.geometry),
            (7, &buffers.stream.solution),
            (16, &buffers.labels),
        ];
        let vertices = (buffers.nx + 1) * (buffers.ny + 1);
        let invert = [
            vec![
                plans.add(kernel::Plan::new(
                    Program::Invert,
                    "invert_cells",
                    &mirrored,
                    buffers.n.div_ceil(2).div_ceil(128),
                )),
                plans.add(kernel::Plan::new(
                    Program::Invert,
                    "invert_stream",
                    &mirrored,
                    vertices.div_ceil(2).div_ceil(128),
                )),
            ],
            interface.kernels(),
            vec![bed.clear_strata],
        ]
        .concat();
        let prepare = [
            vec![bed.geometry, bed.release],
            interface.kernels(),
            vec![
                bed.occupancy,
                bed.passage,
                mechanics.predict,
                mechanics.grain_stress,
                mechanics.prepare_stream,
                mechanics.cache_contact,
            ],
            stream.tensor(),
            stream.assemble(false),
            stream.coarsen(),
        ]
        .concat();
        let steps = update::steps(
            &slab, &bed, &mechanics, &stream, &transport, &labels, &interface,
        );
        let model = Self {
            kernels: Kernels::new(device, &shaders, plans),
            buffers,
            steps,
            prepare,
            invert,
            turn: None,
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for &kernel in &model.prepare {
                model.kernels.record(kernel, &mut pass);
            }
        }
        queue.submit([encoder.finish()]);
        model
    }

    /// One complete 8 ms update: every step, in order, in the caller's pass.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        for step in &self.steps {
            self.kernels.record(step.kernel, pass);
        }
    }

    /// The update's steps. Only their order forms a valid update; tools may
    /// give each step or stage its own pass to time or probe it.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// One step of the update.
    pub fn encode_step(&self, step: Step, pass: &mut wgpu::ComputePass<'_>) {
        self.kernels.record(step.kernel, pass);
    }

    /// The entry point a step runs, for profiles.
    pub fn entry(&self, step: Step) -> &'static str {
        self.kernels.entry(step.kernel)
    }

    /// Begin turning the picture over; a turn in progress goes on unchanged.
    pub fn start_turn(&mut self) {
        if self.turn.is_none() {
            self.turn = Some(Turn {
                elapsed: 0.,
                mirrored: false,
            });
        }
    }

    pub fn turning(&self) -> bool {
        self.turn.is_some()
    }

    /// Advance a turn by one update, before that update is encoded: writes the
    /// update's apparent gravity and returns whether the grid must be mirrored
    /// first. The picture is first shaken left and right: the tank moves as
    /// d sin(2 pi f t), and in its frame the sand feels the lateral gravity
    /// -a(t) = d (2 pi f)^2 sin(2 pi f t). Then it tips front to back through the
    /// angle theta = pi (1 - cos(pi t / T)) / 2: gravity in the glass is
    /// g cos(theta), zero when the picture lies flat, where the grid is mirrored
    /// so that it points down the rows again; the model sees g |cos(theta)|.
    pub fn advance_turn(&mut self, queue: &wgpu::Queue) -> bool {
        use std::f32::consts::PI;
        let Some(turn) = self.turn.as_mut() else {
            return false;
        };
        turn.elapsed += DT;
        let mut lateral = 0.;
        let mut gravity = STANDARD_GRAVITY;
        let mut mirror = false;
        if turn.elapsed < SHAKE_SECONDS {
            let omega = 2. * PI * SHAKE_FREQUENCY;
            lateral = SHAKE_DISPLACEMENT * omega * omega * (omega * turn.elapsed).sin();
        } else {
            let t = ((turn.elapsed - SHAKE_SECONDS) / TURN_SECONDS).min(1.);
            let theta = 0.5 * PI * (1. - (PI * t).cos());
            mirror = !turn.mirrored && theta >= 0.5 * PI;
            turn.mirrored |= mirror;
            if t >= 1. {
                self.turn = None;
            } else {
                gravity = STANDARD_GRAVITY * theta.cos().abs();
            }
        }
        queue.write_buffer(&self.buffers.config, 24, &gravity.to_le_bytes());
        queue.write_buffer(&self.buffers.config, 32, &lateral.to_le_bytes());
        mirror
    }

    /// Turn the current material configuration without advancing physical time.
    /// Face and transport caches are regenerated by the next regular update.
    pub fn invert(&self, pass: &mut wgpu::ComputePass<'_>) {
        for &kernel in &self.invert {
            self.kernels.record(kernel, pass);
        }
    }

    pub fn buffers(&self) -> &Buffers {
        &self.buffers
    }
}
