//! Conservative water, two granular species and separately owned gas domains.
//! The fixed-cost approximation: 8 ms steps, two coupled flow cycles,
//! adaptive conservative transport, curved clearance and interspecies contact.
//! `Model` holds the fields and the kernels, records one update as its list of
//! steps, and turns the picture between updates.
mod bed;
mod buffers;
mod foam;
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
use foam::Foam;
pub use foam::{DEFAULT_AIR_AREA, FOAM_VERTS, MAX_AIR_AREA, MAX_BUBBLES, MIN_AIR_AREA};
pub use grid::{CELLS, TANK, cell_width};
use interface::Interface;
use kernel::{Kernel, Pipelines, Plan};
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
/// Seconds over which gravity ramps from flat (90 deg) to upright (180 deg)
/// immediately after the picture mirrors, giving bubbles a smooth upward start
/// while glass-plate friction briefly steadies the sand.
const TURN_SECONDS: f32 = 0.5;
/// A person can shake the picture left and right for a moment, which slides
/// the top of the sand on the floor toward the level (some makers advise it
/// before a turn; no turn does it by itself): seconds of shaking, its
/// frequency (Hz) and the tank's displacement amplitude (m), a brisk wrist
/// motion (5 mm at 6 Hz = 7.1 m/s^2 ~ 0.72 g lateral peak acceleration).
const SHAKE_SECONDS: f32 = 3.;
const SHAKE_FREQUENCY: f32 = 6.;
const SHAKE_DISPLACEMENT: f32 = 0.005;
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
    /// The kernels that mirror the grid when a turn begins.
    invert: Vec<usize>,
    /// The kernels that adjust live air volume and refresh grid gas/apertures.
    adjust_air_steps: Vec<usize>,
    /// The kernel that writes summary sand/air metrics for auto-flip detection.
    measure_kernel: usize,
    /// What the person's hands are doing to the picture, if anything.
    motion: Option<Motion>,
    /// Live apparent gravity `[gx (+right), gy (+down), gz (out of glass)]`
    /// from an accelerometer/tilt sensor, when enabled.
    sensor_gravity: Option<[f32; 3]>,
    /// True once sensor gravity has been written so disabling restores upright.
    sensor_applied: bool,
    /// Target total air area (cells^2), persisted across resets.
    air_area: f32,
    /// Monotonic counter diversifying injection columns across clicks.
    air_seed: u32,
}
/// A motion of the hands in progress, with the seconds since it began.
enum Motion {
    /// The picture is shaken left and right.
    Shake { elapsed: f32 },
    /// The picture is turned over; `mirrored` once the grid has been mirrored
    /// (when the picture lay flat).
    Turn { elapsed: f32, mirrored: bool },
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
    adjust_air_steps: Vec<usize>,
    measure_kernel: usize,
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
            adjust_air_steps: assembly.adjust_air_steps,
            measure_kernel: assembly.measure_kernel,
            motion: None,
            sensor_gravity: None,
            sensor_applied: false,
            air_area: DEFAULT_AIR_AREA,
            air_seed: 1,
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
    adjust_air_steps: Vec<usize>,
    measure_kernel: usize,
}
fn layout() -> Layout {
    let mut plans = Plans::default();
    let slab = Slab::new(&mut plans);
    let bed = Bed::new(&mut plans);
    let mechanics = Mechanics::new(&mut plans);
    let stream = Stream::new(&mut plans);
    let transport = Transport::new(&mut plans);
    let foam = Foam::new(&mut plans);
    let interface = Interface::new(&mut plans);
    let [nx, ny] = CELLS.map(|v| v as u32);
    let invert = vec![
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
        foam.invert,
        foam.raster,
        interface.apertures,
        bed.clear_strata,
    ];
    let prepare = [
        vec![
            bed.geometry,
            bed.release,
            foam.raster,
            interface.apertures,
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
        &slab, &bed, &mechanics, &stream, &transport, &foam, &interface,
    );
    let adjust_air_steps = vec![foam.adjust_air, foam.raster, interface.apertures];
    let measure_kernel = foam.measure;
    Layout {
        plans: plans.take(),
        steps,
        prepare,
        invert,
        adjust_air_steps,
        measure_kernel,
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
            adjust_air_steps: layout.adjust_air_steps,
            measure_kernel: layout.measure_kernel,
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
        self.motion = None;
        self.sensor_applied = false;
        self.write_gravity(queue, STANDARD_GRAVITY, 0., 0.);
        self.buffers
            .restore(device, queue, &self.scene, self.air_area);
        self.prepare(device, queue);
    }

    /// Sets or clears live 3D apparent gravity `[gx (+right), gy (+down), gz]`
    /// from an external motion sensor (m/s^2). Active whenever no Turn/Shake
    /// button motion is running.
    pub fn set_sensor_gravity(&mut self, gravity: Option<[f32; 3]>) {
        self.sensor_gravity = gravity;
    }

    /// Current target total air area (cells^2).
    pub fn air_area(&self) -> f32 {
        self.air_area
    }

    /// Dynamically adjust live air volume by `delta` cells^2 (clamped to
    /// `[MIN_AIR_AREA, MAX_AIR_AREA]`), updating foam curves and grid gas
    /// immediately without modifying sand.
    pub fn adjust_air(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, delta: f32) -> bool {
        let next = (self.air_area + delta).clamp(MIN_AIR_AREA, MAX_AIR_AREA);
        let applied = next - self.air_area;
        if applied.abs() < 0.5 {
            return false;
        }
        self.air_area = next;
        self.air_seed = self.air_seed.wrapping_add(1);
        queue.write_buffer(&self.buffers.config, 36, &applied.to_le_bytes());
        queue.write_buffer(&self.buffers.config, 40, &self.air_seed.to_le_bytes());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for &kernel in &self.adjust_air_steps {
                self.kernels[kernel].record(&mut pass);
            }
        }
        queue.submit([encoder.finish()]);
        true
    }

    /// Record the single-workgroup sand/air summary kernel into `pass`.
    pub fn measure(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.kernels[self.measure_kernel].record(pass);
    }

    /// One complete 8 ms update: every step, in order, in the caller's pass.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_steps(pass, 0..self.steps.len());
    }

    /// The steps `range` of an update, in order, in the caller's pass: a tool
    /// that times the stages gives each a pass of its own. Only all the
    /// steps, in order, form an update.
    pub fn encode_steps(&self, pass: &mut wgpu::ComputePass<'_>, range: std::ops::Range<usize>) {
        for step in &self.steps[range] {
            self.kernels[step.kernel].record(pass);
        }
    }

    /// The update's steps. Only their order forms a valid update.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Begin turning the picture over, and tell whether a turn began: one in
    /// progress goes on unchanged. Hands that were shaking the picture stop
    /// shaking it.
    pub fn start_turn(&mut self) -> bool {
        let begins = !matches!(self.motion, Some(Motion::Turn { .. }));
        if begins {
            self.motion = Some(Motion::Turn {
                elapsed: 0.,
                mirrored: false,
            });
        }
        begins
    }

    /// True while a turn is currently ramping gravity.
    pub fn is_turning(&self) -> bool {
        matches!(self.motion, Some(Motion::Turn { .. }))
    }

    /// Begin shaking the picture left and right, unless the hands are at a
    /// shake or a turn already.
    pub fn start_shake(&mut self) {
        if self.motion.is_none() {
            self.motion = Some(Motion::Shake { elapsed: 0. });
        }
    }

    /// Current horizontal tank displacement as a fraction of tank width
    /// (`[-d/W, +d/W]`) and smooth lift envelope (`[0, 1]`) while shaken,
    /// matching the physical `d sin(2 pi f t)` frame motion; `(0, 0)` at rest.
    pub fn shake_pose(&self) -> (f32, f32) {
        use std::f32::consts::PI;
        let Some(Motion::Shake { elapsed }) = self.motion else {
            return (0.0, 0.0);
        };
        let t = elapsed.clamp(0.0, SHAKE_SECONDS);
        let omega = 2.0 * PI * SHAKE_FREQUENCY;
        let disp = SHAKE_DISPLACEMENT * (omega * t).sin() / (TANK[0] as f32);
        let ease_out = |x: f32, span: f32| {
            let u = (x / span).clamp(0.0, 1.0);
            let inv = 1.0 - u;
            1.0 - inv * inv * inv
        };
        let env = ease_out(t, 0.10) * ease_out(SHAKE_SECONDS - t, 0.10);
        (disp, env)
    }

    /// Advance the hands' motion by one update, before that update is
    /// encoded: writes the update's apparent gravity and returns whether the
    /// grid must be mirrored first.
    ///
    /// Shaken, the tank moves as d sin(2 pi f t), and in its frame the sand
    /// feels the lateral gravity -a(t) = d (2 pi f)^2 sin(2 pi f t).
    ///
    /// Turned over, the picture tips front to back through the angle
    /// theta = pi (1 - cos(pi t / T)) / 2: gravity in the glass is
    /// g cos(theta), zero when the picture lies flat, where the grid is
    /// mirrored so that it points down the rows again; the model sees
    /// g |cos(theta)|.
    pub fn advance_motion(&mut self, queue: &wgpu::Queue) -> bool {
        use std::f32::consts::PI;
        let mut lateral = 0.;
        let mut gravity = STANDARD_GRAVITY;
        let mut plate = 0.;
        let mut mirror = false;
        let ended = match self.motion.as_mut() {
            None => {
                if let Some(raw) = self.sensor_gravity {
                    let (g, lat, pl) = map_sensor_gravity(raw);
                    self.write_gravity(queue, g, lat, pl);
                    self.sensor_applied = true;
                } else if self.sensor_applied {
                    self.write_gravity(queue, STANDARD_GRAVITY, 0., 0.);
                    self.sensor_applied = false;
                }
                return false;
            }
            Some(Motion::Shake { elapsed }) => {
                *elapsed += DT;
                if *elapsed < SHAKE_SECONDS {
                    let omega = 2. * PI * SHAKE_FREQUENCY;
                    lateral = SHAKE_DISPLACEMENT * omega * omega * (omega * *elapsed).sin();
                }
                *elapsed >= SHAKE_SECONDS
            }
            Some(Motion::Turn { elapsed, mirrored }) => {
                // Mirror immediately on the first update (eliminating the dead
                // 1 s wait before horizontal), then ramp in-plane gravity from
                // 0 -> g over TURN_SECONDS while plate_gravity eases g -> 0.
                mirror = !*mirrored;
                *mirrored = true;
                *elapsed += DT;
                let t = (*elapsed / TURN_SECONDS).min(1.);
                if t < 1. {
                    gravity = STANDARD_GRAVITY * (0.5 * PI * t).sin();
                    plate = (STANDARD_GRAVITY * STANDARD_GRAVITY - gravity * gravity)
                        .max(0.)
                        .sqrt();
                }
                t >= 1.
            }
        };
        if ended {
            self.motion = None;
        }
        self.write_gravity(queue, gravity, lateral, plate);
        mirror
    }

    /// Writes an update's apparent gravity: downward along the rows
    /// (`cfg.gravity`), rightward along x (`cfg.lateral`), and perpendicular
    /// to the glass plates (`cfg.plate`).
    fn write_gravity(&self, queue: &wgpu::Queue, gravity: f32, lateral: f32, plate: f32) {
        queue.write_buffer(&self.buffers.config, 24, &gravity.to_le_bytes());
        queue.write_buffer(&self.buffers.config, 32, &lateral.to_le_bytes());
        queue.write_buffer(&self.buffers.config, 44, &plate.to_le_bytes());
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

/// Maps live 3D apparent gravity `[gx (+right), gy (+down), gz (out of glass)]`
/// (m/s^2) into the simulation's `(gravity, lateral, plate)` triplet.
///
/// Up to +-55 deg from upright (covering normal tilts and vigorous upright
/// hand shakes), `(gy, gx, |gz|)` passes through directly. Past 55 deg toward
/// sideways (90 deg) or upside-down (`gy <= 0`), where the current `y = 0` bed
/// model cannot yet build a vertical-wall deposit, the in-plane component fades
/// smoothly into out-of-plane glass friction `plate` so the sand holds safely.
pub fn map_sensor_gravity(raw: [f32; 3]) -> (f32, f32, f32) {
    const MAX_ACCEL: f32 = 14.0;
    const FADE_START: f32 = 55.0_f32 * std::f32::consts::PI / 180.0;
    const FADE_SPAN: f32 = 30.0_f32 * std::f32::consts::PI / 180.0;

    let gx = raw[0].clamp(-MAX_ACCEL, MAX_ACCEL);
    let gy = raw[1].clamp(-MAX_ACCEL, MAX_ACCEL);
    let gz = raw[2].abs().min(MAX_ACCEL);
    let in_plane = gx.hypot(gy);
    let blend = if gy <= 0.0 {
        1.0
    } else {
        let theta = gx.abs().atan2(gy);
        let u = ((theta - FADE_START) / FADE_SPAN).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    };
    let keep = 1.0 - blend;
    (gy.max(0.0) * keep, gx * keep, gz.hypot(in_plane * blend))
}

#[cfg(test)]
mod tests {
    use super::kernel::declarations;
    use super::uses::USES;
    use super::{Grains, Program, STANDARD_GRAVITY, layout, map_sensor_gravity};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    /// The table of what each entry point uses is what the shaders say, and
    /// no entry point uses more storage buffers than WebGPU promises a stage
    /// (eight). When a shader has changed, `WRITE_USES=1 cargo test uses`
    /// (`mise run uses`) writes the table anew.
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
        let current: String = USES
            .iter()
            .map(|(program, entry, used)| {
                format!("    (Program::{program:?}, {entry:?}, &{used:?}),\n")
            })
            .collect();
        if current != table && std::env::var_os("WRITE_USES").is_some() {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/uses.rs");
            let text = std::fs::read_to_string(path).unwrap();
            let head = &text[..text.find("pub const USES").unwrap()];
            let list = "pub const USES: &[(Program, &str, &[&str])] = &[\n";
            std::fs::write(path, format!("{head}{list}{table}];\n")).unwrap();
            panic!("uses.rs is written anew: run the tests again");
        }
        assert!(
            current == table,
            "the shaders use their buffers otherwise than uses.rs says (WRITE_USES=1 writes its table anew)"
        );
        assert!(crowded.is_empty(), "{}", crowded.join("\n"));
        // Every kernel runs an entry point of the table.
        for plan in layout().plans {
            super::uses::uses(plan.program, plan.entry);
        }
    }

    #[test]
    fn sensor_gravity_maps_tilt_flat_shake_and_guard() {
        let g = STANDARD_GRAVITY;
        // Upright at rest.
        let (gy, gx, gz) = map_sensor_gravity([0.0, g, 0.0]);
        assert!((gy - g).abs() < 1e-5 && gx.abs() < 1e-5 && gz.abs() < 1e-5);
        // 30-degree clockwise tilt passes through unattenuated.
        let rad = 30.0_f32.to_radians();
        let (gy, gx, gz) = map_sensor_gravity([g * rad.sin(), g * rad.cos(), 0.0]);
        assert!((gy - g * rad.cos()).abs() < 1e-4);
        assert!((gx - g * rad.sin()).abs() < 1e-4);
        assert!(gz.abs() < 1e-5);
        // Lying flat on a table routes all gravity into plate friction.
        let (gy, gx, gz) = map_sensor_gravity([0.0, 0.0, g]);
        assert!(gy.abs() < 1e-5 && gx.abs() < 1e-5 && (gz - g).abs() < 1e-5);
        // Upright hand shake (0.72 g lateral) passes through unattenuated.
        let (gy, gx, gz) = map_sensor_gravity([7.1, g, 0.0]);
        assert!((gy - g).abs() < 1e-5 && (gx - 7.1).abs() < 1e-5 && gz.abs() < 1e-5);
        // Sideways 90 deg and upside-down fold safely into plate gravity.
        for raw in [[g, 0.0, 0.0], [0.0, -g, 0.0]] {
            let (gy, gx, gz) = map_sensor_gravity(raw);
            assert!(gy.abs() < 1e-5 && gx.abs() < 1e-5 && (gz - g).abs() < 1e-4);
        }
    }
}
