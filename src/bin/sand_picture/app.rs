//! The window's application: it builds the model while it shows its
//! progress, then runs the picture; keys, the panel and the frame.
use super::panel::{Command, Panel, Readout};
use super::product::Product;
use super::screen::Screen;
use mikage::winit::keyboard::KeyCode;
use mikage::{
    App, GpuContext, RedrawRequest, RenderContext, RenderTargetConfig, SimulationControl,
    SubmissionToken, TickContext, WindowInputContext, egui,
};
use sand_picture::model::{Build, Building, Model, Progress, Scene};
use sand_picture::view::{Mode, picture_size};
use std::time::Duration;
use web_time::Instant;

/// The least time between two pictures: a view is drawn at most thirty times
/// a second, however fast the updates come. A frame of the window in between
/// (the panel, a resize) shows the kept picture again.
const PICTURE_INTERVAL: Duration = Duration::from_nanos(1_000_000_000 / 30);
/// The time a frame gives to building the model while it shows the progress:
/// most of the time between two frames at thirty a second, so that showing
/// the progress costs the build little.
const BUILD_BUDGET: Duration = Duration::from_millis(30);

enum State {
    /// The window's first frame shows that work begins, before any is done.
    Opening,
    Building(Building),
    /// The model is built; its views are made next.
    Built(Model),
    Running(Product),
}

/// What the kept picture shows: the updates it includes, the view, and when
/// it was drawn.
struct Shown {
    updates: u64,
    mode: Mode,
    at: Instant,
}

pub struct SandPicture {
    state: State,
    screen: Screen,
    shown: Option<Shown>,
    panel: Panel,
    mode: Mode,
    /// The person has stopped the picture's time.
    paused: bool,
    redraw: Option<RedrawRequest>,
    control: Option<SimulationControl>,
}
impl SandPicture {
    /// `window` is the window's size in the display's pixels.
    pub fn new(gpu: &GpuContext, target: RenderTargetConfig, window: [u32; 2]) -> Self {
        Self {
            state: State::Opening,
            screen: Screen::new(gpu, target, picture_size(window)),
            shown: None,
            panel: Panel::default(),
            mode: Mode::default(),
            paused: false,
            redraw: None,
            control: None,
        }
    }

    fn request_redraw(&self) {
        if let Some(redraw) = &self.redraw {
            redraw.request_redraw();
        }
    }

    /// A frame's share of building the model. The updates begin when the
    /// last piece is done.
    fn build(&mut self, gpu: &GpuContext) {
        let started = Instant::now();
        loop {
            self.state = match std::mem::replace(&mut self.state, State::Opening) {
                State::Opening => State::Building(Building::new(Scene::product())),
                State::Building(building) => match building.advance(&gpu.device, &gpu.queue) {
                    Build::Unfinished(building) => State::Building(building),
                    Build::Finished(model) => State::Built(model),
                },
                State::Built(model) => {
                    let product = Product::new(gpu, model, self.screen.size());
                    if let Some(control) = self.control.as_ref().filter(|_| !self.paused) {
                        control.resume();
                    }
                    State::Running(product)
                }
                State::Running(product) => {
                    self.state = State::Running(product);
                    return;
                }
            };
            // A frame shows where the work stands before it goes on: the
            // first frame, a frame every BUILD_BUDGET, the frame before the
            // views are made - and a frame once every kernel is made, before
            // the model's first work goes to the GPU: a browser compiles the
            // kernels only then, and takes seconds over it.
            let goes_on = match &self.state {
                State::Building(building) => {
                    !matches!(building.progress(), Progress::Kernels(made, of) if made == of)
                        && started.elapsed() < BUILD_BUDGET
                }
                _ => false,
            };
            if !goes_on {
                self.request_redraw();
                return;
            }
        }
    }

    fn command(&mut self, command: Command) {
        match command {
            Command::Pause => {
                self.paused = !self.paused;
                if let State::Running(product) = &mut self.state {
                    product.play_begins();
                    if let Some(control) = &self.control {
                        if self.paused {
                            control.pause();
                        } else {
                            control.resume();
                        }
                    }
                }
            }
            Command::Turn => {
                if let State::Running(product) = &mut self.state {
                    product.turn();
                }
            }
            Command::Shake => {
                if let State::Running(product) = &mut self.state {
                    product.shake();
                }
            }
            Command::Reset => {
                if let State::Running(product) = &mut self.state {
                    product.reset();
                }
            }
        }
        self.request_redraw();
    }
}
impl App for SandPicture {
    type Camera = ();

    fn set_redraw_request(&mut self, request: RedrawRequest) {
        self.redraw = Some(request);
    }

    /// No update runs before the model is built.
    fn set_simulation_control(&mut self, control: SimulationControl) {
        control.pause();
        self.control = Some(control);
    }

    fn on_input(&mut self, ctx: &mut WindowInputContext<'_, ()>) {
        for (key, command) in [
            (KeyCode::KeyF, Command::Turn),
            (KeyCode::KeyS, Command::Shake),
            (KeyCode::Space, Command::Pause),
            (KeyCode::KeyR, Command::Reset),
        ] {
            if ctx.input.is_key_pressed(key) {
                self.command(command);
            }
        }
        if ctx.input.is_key_pressed(KeyCode::KeyV) {
            self.mode = self.mode.next();
        }
        if ctx.input.is_key_pressed(KeyCode::KeyH) {
            self.panel.toggle();
        }
        if ctx.input.is_key_pressed(KeyCode::KeyP) {
            self.panel.change_side();
        }
    }

    fn tick(&mut self, ctx: &mut TickContext<'_>) {
        if let State::Running(product) = &mut self.state {
            product.tick(ctx);
            // Whoever ran the update (the clock, or a tool through the
            // agent), the window now shows an older state.
            self.request_redraw();
        }
    }

    fn after_complete(&mut self, _gpu: &GpuContext, token: &SubmissionToken) {
        if let State::Running(product) = &mut self.state {
            product.completed(token);
        }
    }

    fn render(&mut self, ctx: &mut RenderContext<'_, ()>) {
        self.build(ctx.gpu);
        // The picture has the window's pixels: a window of another size
        // gets a picture of another size.
        let size = picture_size([ctx.target.size.width, ctx.target.size.height]);
        if size != self.screen.size() {
            self.screen.resize(ctx.gpu, size);
            if let State::Running(product) = &self.state {
                product.resize(ctx.gpu, size);
            }
            self.shown = None;
        }
        let State::Running(product) = &mut self.state else {
            self.screen.encode(ctx, false);
            return;
        };
        // A reset asked for while the time is stopped shows at once.
        if product.apply_pending(ctx.gpu) {
            self.shown = None;
        }
        let stale = self
            .shown
            .as_ref()
            .is_none_or(|shown| shown.updates != product.updates() || shown.mode != self.mode);
        let now = Instant::now();
        let due = self
            .shown
            .as_ref()
            .is_none_or(|shown| now.duration_since(shown.at) >= PICTURE_INTERVAL);
        if stale && due {
            product.draw(ctx.encoder, self.screen.image(), self.mode);
            self.shown = Some(Shown {
                updates: product.updates(),
                mode: self.mode,
                at: now,
            });
        } else if stale {
            // The newest state is drawn once the interval has passed, even
            // if no further update asks for a frame.
            self.request_redraw();
        }
        self.screen.encode(ctx, true);
    }

    fn gui(&mut self, root: &mut egui::Ui) {
        let readout = match &self.state {
            State::Running(product) => Readout {
                seconds: product.seconds(),
                paused: self.paused,
                speed: product.speed(),
                since_turn: product.since_turn(),
            },
            State::Opening => return progress(root, None),
            State::Building(building) => return progress(root, Some(building.progress())),
            State::Built(_) => return progress(root, None),
        };
        if let Some(command) = self.panel.show(root, &mut self.mode, &readout) {
            self.command(command);
        }
    }

    fn capture_targets(&self, registry: &mut mikage::capture::CaptureRegistry) {
        registry.register_texture("picture", self.screen.texture());
    }

    #[cfg(not(target_family = "wasm"))]
    fn agent_status(&self) -> serde_json::Value {
        let State::Running(product) = &self.state else {
            return serde_json::json!({"state": "building"});
        };
        serde_json::json!({
            "state": "running",
            "view": self.mode.label(),
            "paused": self.paused,
            "updates": product.updates(),
            "completed_updates": product.completed_updates(),
            "seconds": product.seconds(),
            "speed": product.speed(),
            "since_turn": product.since_turn(),
        })
    }

    /// `{"action": "turn" | "shake" | "pause" | "reset" | "view" | "panel" |
    /// "details"}`; a view is named by `"value"`, `panel` hides or shows the
    /// panel and `details` folds or unfolds its times since the turn.
    #[cfg(not(target_family = "wasm"))]
    fn on_agent_command(
        &mut self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let action = payload.get("action").and_then(serde_json::Value::as_str);
        match action.ok_or("an action is needed")? {
            "turn" => self.command(Command::Turn),
            "shake" => self.command(Command::Shake),
            "pause" => self.command(Command::Pause),
            "reset" => self.command(Command::Reset),
            "panel" => self.panel.toggle(),
            "details" => self.panel.toggle_details(),
            "view" => {
                let value = payload.get("value").and_then(serde_json::Value::as_str);
                self.mode = Mode::ALL
                    .into_iter()
                    .find(|mode| Some(mode.label()) == value)
                    .ok_or("a view is Simple, Photoreal or Flow")?;
            }
            other => return Err(format!("unknown action {other:?}")),
        }
        self.request_redraw();
        Ok(self.agent_status())
    }
}

/// The window while the model is built: what is being made, and how far.
fn progress(root: &mut egui::Ui, progress: Option<Progress>) {
    let (text, share) = match progress {
        None => ("Preparing the picture".to_owned(), 0.),
        Some(Progress::Materials(done, of)) => (
            "Working out the materials".to_owned(),
            done as f32 / of as f32,
        ),
        Some(Progress::Kernels(done, of)) => (
            format!("Compiling the kernels, {done} of {of}"),
            done as f32 / of as f32,
        ),
    };
    egui::Area::new(egui::Id::unique("progress"))
        .anchor(egui::Align2::CENTER_CENTER, [0., 0.])
        .show(root.ctx(), |ui| {
            ui.vertical_centered(|ui| {
                ui.label(text);
                ui.add(egui::ProgressBar::new(share).desired_width(240.));
            });
        });
}
