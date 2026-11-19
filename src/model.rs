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
mod rows;
mod scene;
mod shader;
mod slab;
mod stream;
mod transport;
mod update;
mod uses;

use bed::Bed;
pub use buffers::Buffers;
pub use grid::{CELLS, TANK, cell_width};
use interface::Interface;
use kernel::{Kernel, Pipelines, Plan};
use labels::Labels;
pub use material::{Grains, POWDER_DIAMETER};
use material::{TABLE_STRIPS, table_strip};
use mechanics::Mechanics;
pub use rows::constants as row_constants;
pub use scene::Scene;
use shader::Program;
use slab::Slab;
use stream::Stream;
use transport::Transport;
use update::Plans;
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
/// The gas identities a cell can hold: the shaders' `GAS_SLOTS`.
pub const GAS_SLOTS: usize = 9;

pub struct Model {
    /// The picture at rest that a reset returns to.
    scene: Scene,
    buffers: Buffers,
    /// The modules' kernels by place: those of the update and those that run
    /// only at the start or at a turn.
    kernels: Vec<Kernel>,
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

/// A model being built, piece by piece: its materials' table takes seconds
/// on the CPU and its pipelines can take seconds to compile, and a window
/// draws its progress between pieces.
pub struct Building {
    scene: Scene,
    grains: Grains,
    /// The strips of the materials' table computed so far.
    table: Vec<[f32; 4]>,
    /// What follows the table: the buffers, and the kernels made so far.
    kernels: Option<Assembly>,
}
struct Assembly {
    buffers: Buffers,
    pipelines: Pipelines,
    plans: std::vec::IntoIter<Plan>,
    kernels: Vec<Kernel>,
    steps: Vec<Step>,
    prepare: Vec<usize>,
    invert: Vec<usize>,
}
/// What a piece of the work leaves: more work, or the model.
pub enum Build {
    Unfinished(Building),
    Finished(Model),
}
/// What a model being built is working on, and how far it is.
#[derive(Clone, Copy)]
pub enum Progress {
    /// The materials' table: strips computed, of how many.
    Materials(usize, usize),
    /// The kernels: made, of how many.
    Kernels(usize, usize),
}
impl Building {
    /// Starts building the model of the picture at rest in `scene`.
    pub fn new(scene: Scene) -> Self {
        Self {
            scene,
            grains: Grains::default(),
            table: Vec::new(),
            kernels: None,
        }
    }

    pub fn progress(&self) -> Progress {
        match &self.kernels {
            None => Progress::Materials(self.table.len() / (2 * material::SIDE), TABLE_STRIPS),
            Some(assembly) => Progress::Kernels(
                assembly.kernels.len(),
                assembly.kernels.len() + assembly.plans.len(),
            ),
        }
    }

    /// Does one piece of the work: a strip of the table, or the buffers, or a
    /// kernel; once nothing is left, the model.
    pub fn advance(mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Build {
        let Some(mut assembly) = self.kernels.take() else {
            let strip = self.table.len() / (2 * material::SIDE);
            if strip < TABLE_STRIPS {
                self.table.extend(table_strip(&self.grains, strip));
            } else {
                self.kernels = Some(Assembly::new(device, queue, &self));
            }
            return Build::Unfinished(self);
        };
        if let Some(plan) = assembly.plans.next() {
            let kernel = assembly.pipelines.kernel(device, &assembly.buffers, plan);
            assembly.kernels.push(kernel);
            self.kernels = Some(assembly);
            return Build::Unfinished(self);
        }
        let model = Model {
            scene: self.scene,
            buffers: assembly.buffers,
            kernels: assembly.kernels,
            steps: assembly.steps,
            prepare: assembly.prepare,
            invert: assembly.invert,
            turn: None,
        };
        model.prepare(device, queue);
        Build::Finished(model)
    }
}
/// Every kernel of the model as a plan, by place, and the lists that run
/// them: the update's steps, the preparation of the first update, and the
/// mirroring of a turn.
struct Layout {
    plans: Vec<Plan>,
    steps: Vec<Step>,
    prepare: Vec<usize>,
    invert: Vec<usize>,
}
fn layout() -> Layout {
    let mut plans = Plans::default();
    let slab = Slab::new(&mut plans);
    let bed = Bed::new(&mut plans);
    let mechanics = Mechanics::new(&mut plans);
    let stream = Stream::new(&mut plans);
    let transport = Transport::new(&mut plans);
    let labels = Labels::new(&mut plans);
    let interface = Interface::new(&mut plans);
    let [nx, ny] = CELLS.map(|v| v as u32);
    let invert = [
        vec![
            plans.add(Plan::new(
                Program::Invert,
                "invert_cells",
                (nx * ny).div_ceil(2).div_ceil(128),
            )),
            plans.add(Plan::new(
                Program::Invert,
                "invert_stream",
                ((nx + 1) * (ny + 1)).div_ceil(2).div_ceil(128),
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
    Layout {
        plans: plans.take(),
        steps,
        prepare,
        invert,
    }
}
impl Assembly {
    /// The buffers of the scene, the programs' modules, and the plans of
    /// every kernel; the kernels themselves are made one by one.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, building: &Building) -> Self {
        let layout = layout();
        Self {
            buffers: Buffers::new(device, queue, &building.scene, &building.table),
            pipelines: Pipelines::new(device, &building.grains),
            plans: layout.plans.into_iter(),
            kernels: Vec::new(),
            steps: layout.steps,
            prepare: layout.prepare,
            invert: layout.invert,
        }
    }
}

impl Model {
    /// The model of the picture at rest in `scene`, built in one go.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: Scene) -> Self {
        let mut building = Building::new(scene);
        loop {
            match building.advance(device, queue) {
                Build::Finished(model) => return model,
                Build::Unfinished(unfinished) => building = unfinished,
            }
        }
    }

    /// Prepares the first update from the buffers of a picture at rest.
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for &kernel in &self.prepare {
                self.kernels[kernel].record(&mut pass);
            }
        }
        queue.submit([encoder.finish()]);
    }

    /// Back to the picture at rest in the scene the model was built from, as
    /// if built anew: no pipeline is made again. Call it between updates.
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.turn = None;
        self.buffers.restore(device, queue, &self.scene);
        self.prepare(device, queue);
    }

    /// One complete 8 ms update: every step, in order, in the caller's pass.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        for step in &self.steps {
            self.kernels[step.kernel].record(pass);
        }
    }

    /// The update's steps. Only their order forms a valid update.
    pub fn steps(&self) -> &[Step] {
        &self.steps
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
            self.kernels[kernel].record(pass);
        }
    }

    pub fn buffers(&self) -> &Buffers {
        &self.buffers
    }
}

#[cfg(test)]
mod tests {
    use super::kernel::declarations;
    use super::uses::USES;
    use super::{Grains, Program, layout};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    /// The table of what each entry point uses is what the shaders say, and
    /// no entry point uses more storage buffers than WebGPU promises a stage
    /// (eight). When a shader has changed, the failure prints the table.
    #[test]
    fn the_table_of_uses_is_the_shaders() {
        let grains = Grains::default();
        let mut table = String::new();
        let mut crowded = Vec::new();
        for program in Program::ALL {
            let source = program.source(&grains);
            let declared = declarations(&source);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let info = Validator::new(ValidationFlags::all(), Capabilities::default())
                .validate(&module)
                .unwrap();
            let mut entries: Vec<_> = module.entry_points.iter().enumerate().collect();
            entries.sort_by_key(|(_, entry)| entry.name.clone());
            for (index, entry) in entries {
                let uses = info.get_entry_point(index);
                // The buffers among the globals the entry point uses, in the
                // order of their bindings.
                let mut used: Vec<_> = module
                    .global_variables
                    .iter()
                    .filter(|(handle, global)| {
                        global.binding.is_some() && !uses[*handle].is_empty()
                    })
                    .map(|(_, global)| global.name.as_deref().unwrap())
                    .collect();
                used.sort_by_key(|name| (declared[*name].binding, *name));
                let storage = used.iter().filter(|name| !declared[**name].uniform).count();
                if storage > 8 {
                    crowded.push(format!(
                        "{program:?} {}: {storage} storage buffers",
                        entry.name
                    ));
                }
                table.push_str(&format!(
                    "    (Program::{program:?}, {:?}, &{used:?}),\n",
                    entry.name
                ));
            }
        }
        assert!(crowded.is_empty(), "{}", crowded.join("\n"));
        let current: String = USES
            .iter()
            .map(|(program, entry, used)| {
                format!("    (Program::{program:?}, {entry:?}, &{used:?}),\n")
            })
            .collect();
        assert!(
            current == table,
            "the shaders use their buffers otherwise than uses.rs says; its table should be:\n{table}"
        );
        // Every kernel runs an entry point of the table.
        for plan in layout().plans {
            super::uses::uses(plan.program, plan.entry);
        }
    }
}
