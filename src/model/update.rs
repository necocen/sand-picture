//! One complete 8 ms update as a list of steps, each a kernel's dispatch, in
//! their only valid order. The product records the list into one compute
//! pass; a tool can walk the same list stage by stage.
use super::bed::Bed;
use super::foam::Foam;
use super::interface::Interface;
use super::kernel::Plan;
use super::mechanics::Mechanics;
use super::slab::Slab;
use super::stream::{Solve, Stream};
use super::transport::Transport;

/// The stages of an update, in order. Physically alike steps in another
/// order give another splitting error, history and rounding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The slab's teeth, body layers and joint columns, read by this update's
    /// transport.
    Slab,
    BedPrepare,
    FirstPredict,
    FirstTensor,
    FirstAssemble,
    FirstCoarsen,
    FirstSolve,
    FirstReturn,
    Capture,
    Surface,
    BedRefresh,
    SecondContact,
    SecondTensor,
    SecondAssemble,
    SecondCoarsen,
    SecondSolve,
    SecondReturn,
    TransportPrepare,
    TransportCarry,
    PlugDescent,
    /// Prototype A: Lagrangian closed-curve deformable foam bubbles.
    Foam,
    InterfaceApertures,
}
impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Slab => "slab",
            Self::BedPrepare => "bed prepare",
            Self::FirstPredict => "first predict",
            Self::FirstTensor => "first tensor",
            Self::FirstAssemble => "first assemble",
            Self::FirstCoarsen => "first coarsen",
            Self::FirstSolve => "first solve",
            Self::FirstReturn => "first return",
            Self::Capture => "capture",
            Self::Surface => "surface",
            Self::BedRefresh => "bed refresh",
            Self::SecondContact => "second contact",
            Self::SecondTensor => "second tensor",
            Self::SecondAssemble => "second assemble",
            Self::SecondCoarsen => "second coarsen",
            Self::SecondSolve => "second solve",
            Self::SecondReturn => "second return",
            Self::TransportPrepare => "transport prepare",
            Self::TransportCarry => "transport carry",
            Self::PlugDescent => "plug descent",
            Self::Foam => "foam",
            Self::InterfaceApertures => "interface apertures",
        }
    }
}

/// One dispatch of the update: a kernel of the model, by its place.
#[derive(Clone, Copy)]
pub struct Step {
    pub stage: Stage,
    pub kernel: usize,
}

/// The kernels the modules ask for, in the order they asked; a module keeps
/// the places.
#[derive(Default)]
pub struct Plans {
    plans: Vec<Plan>,
}
impl Plans {
    pub fn add(&mut self, plan: Plan) -> usize {
        self.plans.push(plan);
        self.plans.len() - 1
    }

    /// The plans in their places.
    pub fn take(self) -> Vec<Plan> {
        self.plans
    }
}

/// The steps of one update. Two flow cycles: the first supplies the
/// incoming suspension state, the bed captures and its surface moves once
/// between them, the second answers the changed bed; then the conservative
/// transport, and the geometry for the next whole update.
pub fn steps(
    slab: &Slab,
    bed: &Bed,
    mechanics: &Mechanics,
    stream: &Stream,
    transport: &Transport,
    foam: &Foam,
    interface: &Interface,
) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut stage = |stage: Stage, kernels: Vec<usize>| {
        steps.extend(kernels.into_iter().map(|kernel| Step { stage, kernel }));
    };
    stage(Stage::Slab, slab.kernels());
    stage(
        Stage::BedPrepare,
        vec![
            bed.begin_displacement,
            bed.geometry,
            bed.record_strata,
            bed.release,
            bed.occupancy,
            bed.passage,
        ],
    );
    stage(
        Stage::FirstPredict,
        vec![
            mechanics.predict,
            mechanics.transfer_contact,
            mechanics.grain_stress,
            mechanics.prepare_stream,
        ],
    );
    stage(Stage::FirstTensor, stream.tensor());
    stage(Stage::FirstAssemble, stream.assemble(false));
    stage(Stage::FirstCoarsen, stream.coarsen());
    stage(Stage::FirstSolve, stream.solve(Solve::First));
    stage(
        Stage::FirstReturn,
        [
            vec![mechanics.grain_velocity, mechanics.grain],
            mechanics.relax_pressure(),
            vec![
                mechanics.copy_stress,
                mechanics.project_faces,
                bed.capture_source,
            ],
        ]
        .concat(),
    );
    // Capture and surface remap occur once, before cycle two.
    stage(
        Stage::Capture,
        vec![bed.begin_displacement, bed.geometry, bed.exchange],
    );
    stage(Stage::Surface, vec![bed.surface_even, bed.surface_odd]);
    stage(Stage::BedRefresh, vec![bed.occupancy, bed.passage]);
    // Contact history and second mechanics update.
    stage(
        Stage::SecondContact,
        [
            vec![mechanics.predict, mechanics.grain_stress],
            mechanics.resolve_contact(),
        ]
        .concat(),
    );
    // Bed occupancy and its resistance changed: rebuild the operator, not
    // just its right-hand side.
    stage(Stage::SecondTensor, stream.tensor());
    stage(Stage::SecondAssemble, stream.assemble(true));
    stage(Stage::SecondCoarsen, stream.coarsen());
    stage(Stage::SecondSolve, stream.solve(Solve::Second));
    stage(
        Stage::SecondReturn,
        vec![
            mechanics.project_faces,
            mechanics.momentum,
            mechanics.hold_friction,
        ],
    );
    stage(Stage::TransportPrepare, transport.prepare());
    stage(Stage::TransportCarry, transport.carry());
    stage(Stage::PlugDescent, transport.plug_descent());
    stage(Stage::Foam, foam.kernels());
    stage(Stage::InterfaceApertures, vec![interface.apertures]);
    steps
}
