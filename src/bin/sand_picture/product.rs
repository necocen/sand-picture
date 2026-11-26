//! The picture as it runs in the window: the model, its views, the count of
//! its updates and what the person has asked of it.
use super::sensor::{QuarterAction, QuarterTurnDetector};
use mikage::{GpuContext, SubmissionToken, TickContext};
use sand_picture::model::{DEFAULT_AIR_AREA, DT, GridTurn, Model};
use sand_picture::view::{Mode, Views};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;
use web_time::Instant;

/// The span of play over which the shown speed is averaged.
const RECENT: Duration = Duration::from_secs(2);
/// Updates between GPU sand-completion checks while auto-flip is enabled (1.0 s).
const MEASURE_INTERVAL: u64 = 125;
/// Minimum picture time after a turn before auto-flip may consider drainage complete.
const AUTO_MIN_TURN_SECONDS: f64 = 25.0;
/// Fraction of total sand remaining > 6 cells above the bed below which drainage is done.
const AUTO_DONE_ABOVE_FRAC: f64 = 0.015;
/// Consecutive seconds of negligible bed growth (< 0.05%/s) that trigger safety auto-flip.
const AUTO_STALL_SECONDS: f64 = 30.0;
/// Viewing pause after completion before auto-flipping the picture.
const AUTO_VIEW_PAUSE: Duration = Duration::from_secs(8);
/// Air area change per Air - / Air + click (cells^2, ~7.5% of default).
const AIR_STEP: f32 = 28.0;

enum ReadbackState {
    Idle,
    CopySubmitted {
        tick_id: u64,
        epoch: u64,
    },
    /// `status`: 0 = pending, 1 = mapped ok, 2 = map failed.
    Mapping {
        status: Arc<AtomicU8>,
        epoch: u64,
    },
}

pub struct Product {
    model: Model,
    views: Views,
    /// 16-byte staging buffer for non-blocking auto-flip metric readback.
    metrics_staging: wgpu::Buffer,
    readback: ReadbackState,
    /// Incremented on every turn or reset so stale in-flight readbacks are ignored.
    turn_epoch: u64,
    /// Updates encoded since the picture last rested in its scene.
    updates: u64,
    /// Those of them the GPU has completed.
    completed: u64,
    /// The runtime's ticks not yet reported complete, each with the update
    /// it encoded.
    in_flight: VecDeque<(u64, u64)>,
    /// Completions within the last RECENT of play: when, and the update then
    /// complete.
    recent: VecDeque<(Instant, u64)>,
    /// When the picture was last turned over: by the clock, and the updates
    /// encoded until then.
    turned: (Instant, u64),
    auto_flip: bool,
    auto_flip_at: Option<Instant>,
    last_bed_frac: Option<f64>,
    stalled_secs: f64,
    reset_pending: bool,
    turn_pending: bool,
    shake_pending: bool,
    air_delta_pending: f32,
    /// Quarter-turns CW (`0..4`) the simulation grid has rotated relative to
    /// the display rectangle; `Screen` applies the inverse rotation to UVs.
    view_quarter: u8,
    /// Next `view_quarter` to commit atomically on the next discrete grid turn.
    pending_view_quarter: Option<u8>,
    quarter_detector: QuarterTurnDetector,
}
impl Product {
    /// The picture rests with its sand on the floor; the person turns it over
    /// at once, as at every reset, and its drainage begins.
    /// The views draw pictures of `size` pixels.
    pub fn new(gpu: &GpuContext, mut model: Model, size: [u32; 2]) -> Self {
        model.start_turn();
        let metrics_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-flip metrics staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            views: Views::new(&gpu.device, model.buffers(), size),
            model,
            metrics_staging,
            readback: ReadbackState::Idle,
            turn_epoch: 0,
            updates: 0,
            completed: 0,
            in_flight: VecDeque::new(),
            recent: VecDeque::new(),
            turned: (Instant::now(), 0),
            auto_flip: false,
            auto_flip_at: None,
            last_bed_frac: None,
            stalled_secs: 0.0,
            reset_pending: false,
            turn_pending: false,
            shake_pending: false,
            air_delta_pending: 0.0,
            view_quarter: 0,
            pending_view_quarter: None,
            quarter_detector: QuarterTurnDetector::default(),
        }
    }

    /// The person turns the picture over: it is tipped during the next
    /// updates. A turn under way goes on unchanged.
    pub fn turn(&mut self) {
        self.turn_pending = true;
    }

    /// Toggle automatic turning once sand finishes falling.
    pub fn toggle_auto_flip(&mut self) {
        self.auto_flip = !self.auto_flip;
        self.auto_flip_at = None;
        self.last_bed_frac = None;
        self.stalled_secs = 0.0;
    }

    pub fn auto_flip(&self) -> bool {
        self.auto_flip
    }

    /// Queue a dynamic air volume adjustment (`-1` removes air, `+1` injects bubbles).
    pub fn adjust_air_steps(&mut self, steps: i32) {
        self.air_delta_pending += steps as f32 * AIR_STEP;
    }

    pub fn air_area(&self) -> f32 {
        self.model.air_area()
    }

    pub fn air_percent(&self) -> u32 {
        (self.model.air_area() / DEFAULT_AIR_AREA * 100.0).round() as u32
    }

    /// The person shakes the picture left and right for a moment, unless
    /// the hands are at a shake or a turn already.
    pub fn shake(&mut self) {
        self.shake_pending = true;
    }

    /// Current horizontal shake displacement (fraction of tank width) and lift
    /// envelope (`[0, 1]`) for visual tank motion.
    pub fn shake_pose(&self) -> (f32, f32) {
        self.model.shake_pose()
    }

    /// Quarter-turns CW (`0..4`) the simulation grid is rotated relative to the display.
    pub fn view_quarter(&self) -> u8 {
        self.view_quarter
    }

    /// True when the simulation grid is currently transposed (`144x216`).
    pub fn is_transposed(&self) -> bool {
        self.model.is_transposed()
    }

    /// Feeds live 3D apparent gravity and OS 180-deg orientation flips from a
    /// device motion sensor, handling full 360-deg four-wall rotation seamlessly.
    pub fn update_sensor(&mut self, raw: Option<[f32; 3]>, os_flipped_180: bool) {
        if raw.is_none() && !os_flipped_180 {
            self.quarter_detector.reset();
            self.model.set_sensor_gravity(None);
            match self.view_quarter & 3 {
                2 => {
                    self.view_quarter = 0;
                    self.pending_view_quarter = None;
                }
                1 if self.model.start_quarter_turn(false) => {
                    self.pending_view_quarter = Some(0);
                }
                3 if self.model.start_quarter_turn(true) => {
                    self.pending_view_quarter = Some(0);
                }
                _ => {}
            }
            return;
        }
        let (eff, action) = self.quarter_detector.step(
            raw,
            os_flipped_180,
            self.view_quarter,
            self.model.is_turning(),
            DT,
        );
        self.model.set_sensor_gravity(eff);
        match action {
            QuarterAction::None => {}
            QuarterAction::Turn {
                grid_turn,
                next_view_quarter,
            } => {
                let started = match grid_turn {
                    GridTurn::Invert180 => self.model.start_turn(),
                    GridTurn::RotateCw90 => self.model.start_quarter_turn(true),
                    GridTurn::RotateCcw90 => self.model.start_quarter_turn(false),
                    GridTurn::None => false,
                };
                if started {
                    self.pending_view_quarter = Some(next_view_quarter & 3);
                    self.turned = (Instant::now(), self.updates);
                    self.reset_auto_tracker();
                }
            }
            QuarterAction::SetViewQuarter(q) => {
                self.view_quarter = q & 3;
                self.pending_view_quarter = None;
            }
        }
    }

    /// Back to the scene at rest, turned over at once.
    pub fn reset(&mut self) {
        self.reset_pending = true;
    }

    fn reset_auto_tracker(&mut self) {
        self.turn_epoch = self.turn_epoch.wrapping_add(1);
        self.auto_flip_at = None;
        self.last_bed_frac = None;
        self.stalled_secs = 0.0;
    }

    /// Poll any completed non-blocking metrics readback and update auto-flip state.
    fn poll_metrics(&mut self) {
        let done_state = match &self.readback {
            ReadbackState::Mapping { status, epoch } => {
                let code = status.load(Ordering::Acquire);
                (code != 0).then_some((code == 1, *epoch))
            }
            _ => None,
        };
        let Some((ok, epoch)) = done_state else {
            return;
        };
        if !ok {
            self.readback = ReadbackState::Idle;
            return;
        }
        let words = {
            let slice = self.metrics_staging.slice(..);
            let Ok(mapped) = slice.get_mapped_range() else {
                self.metrics_staging.unmap();
                self.readback = ReadbackState::Idle;
                return;
            };
            let mut out = [0u32; 4];
            for (i, chunk) in mapped.chunks_exact(4).take(4).enumerate() {
                out[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        };
        self.metrics_staging.unmap();
        self.readback = ReadbackState::Idle;

        if !self.auto_flip || epoch != self.turn_epoch {
            return;
        }
        let total = f64::from(words[0]) / 1024.0;
        if total < 100.0 {
            return;
        }
        let bed_frac = (f64::from(words[1]) / 1024.0) / total;
        let above_frac = (f64::from(words[2]) / 1024.0) / total;
        let since_turn_sim = self.updates.saturating_sub(self.turned.1) as f64 * f64::from(DT);

        if since_turn_sim < AUTO_MIN_TURN_SECONDS {
            self.last_bed_frac = Some(bed_frac);
            self.stalled_secs = 0.0;
            return;
        }
        if let Some(prev) = self.last_bed_frac {
            if bed_frac >= 0.15 && (bed_frac - prev) < 0.0005 {
                self.stalled_secs += MEASURE_INTERVAL as f64 * f64::from(DT);
            } else if bed_frac - prev >= 0.001 {
                self.stalled_secs = 0.0;
            }
        }
        self.last_bed_frac = Some(bed_frac);

        let done = above_frac < AUTO_DONE_ABOVE_FRAC || self.stalled_secs >= AUTO_STALL_SECONDS;
        if done {
            if self.auto_flip_at.is_none() {
                self.auto_flip_at = Some(Instant::now() + AUTO_VIEW_PAUSE);
            }
        } else {
            self.auto_flip_at = None;
        }
    }

    /// Applies what was asked for since the last update. True when the
    /// picture changed without an update: after a reset or air adjustment.
    pub fn apply_pending(&mut self, gpu: &GpuContext) -> bool {
        self.poll_metrics();
        if self.auto_flip
            && let Some(at) = self.auto_flip_at
            && Instant::now() >= at
        {
            self.turn_pending = true;
            self.auto_flip_at = None;
        }
        let mut changed = false;
        let reset = std::mem::take(&mut self.reset_pending);
        if reset {
            self.model.reset(&gpu.device, &gpu.queue);
            self.model.start_turn();
            self.updates = 0;
            self.completed = 0;
            self.in_flight.clear();
            self.turned = (Instant::now(), 0);
            self.turn_pending = false;
            self.shake_pending = false;
            self.air_delta_pending = 0.0;
            self.view_quarter = 0;
            self.pending_view_quarter = None;
            self.quarter_detector.reset();
            self.reset_auto_tracker();
            self.play_begins();
            changed = true;
        }
        let air_delta = std::mem::take(&mut self.air_delta_pending);
        if air_delta.abs() >= 0.5 && self.model.adjust_air(&gpu.device, &gpu.queue, air_delta) {
            changed = true;
        }
        if std::mem::take(&mut self.turn_pending) && self.model.start_turn() {
            self.pending_view_quarter = None;
            self.turned = (Instant::now(), self.updates);
            self.reset_auto_tracker();
        }
        if std::mem::take(&mut self.shake_pending) {
            self.model.start_shake();
        }
        changed
    }

    /// One update, in the tick's encoder.
    pub fn tick(&mut self, ctx: &mut TickContext<'_>) {
        self.apply_pending(ctx.gpu);
        let grid_turn = self.model.advance_motion(&ctx.gpu.queue);
        if grid_turn != GridTurn::None
            && let Some(next_q) = self.pending_view_quarter.take()
        {
            self.view_quarter = next_q;
        }
        let want_measure = self.auto_flip
            && self.updates.is_multiple_of(MEASURE_INTERVAL)
            && matches!(self.readback, ReadbackState::Idle);
        {
            let mut pass = ctx.encoder.begin_compute_pass(&Default::default());
            if grid_turn != GridTurn::None {
                self.model.apply_turn(&mut pass, grid_turn);
            }
            self.model.encode(&mut pass);
            if want_measure {
                self.model.measure(&mut pass);
            }
        }
        if want_measure {
            ctx.encoder.copy_buffer_to_buffer(
                &self.model.buffers().metrics,
                0,
                &self.metrics_staging,
                0,
                16,
            );
            self.readback = ReadbackState::CopySubmitted {
                tick_id: ctx.tick_id,
                epoch: self.turn_epoch,
            };
        }
        self.updates += 1;
        self.in_flight.push_back((ctx.tick_id, self.updates));
    }

    /// The GPU has completed the work up to `token`.
    pub fn completed(&mut self, token: &SubmissionToken) {
        let before = self.completed;
        if let Some(tick) = token.tick_id {
            while self.in_flight.front().is_some_and(|(id, _)| *id <= tick) {
                self.completed = self.in_flight.pop_front().unwrap().1;
            }
            if let ReadbackState::CopySubmitted { tick_id, epoch } = self.readback
                && tick >= tick_id
            {
                let status = Arc::new(AtomicU8::new(0));
                let flag = Arc::clone(&status);
                self.metrics_staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |res| {
                        flag.store(if res.is_ok() { 1 } else { 2 }, Ordering::Release);
                    });
                self.readback = ReadbackState::Mapping { status, epoch };
            }
        }
        if self.completed > before {
            let now = Instant::now();
            self.recent.push_back((now, self.completed));
            // Keep one completion at least RECENT old, so that the span
            // covers the window; drop the ones older than that.
            while self.recent.len() > 2 && now.duration_since(self.recent[1].0) > RECENT {
                self.recent.pop_front();
            }
        }
    }

    /// A stretch of play begins: after a pause or a reset the speed is
    /// measured anew.
    pub fn play_begins(&mut self) {
        self.recent.clear();
    }

    /// Updates encoded since the scene: what a picture drawn now shows.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Updates the GPU has completed since the scene: for the agent's status.
    #[cfg(not(target_family = "wasm"))]
    pub fn completed_updates(&self) -> u64 {
        self.completed
    }

    /// The picture's own time (s): that of its completed updates.
    pub fn seconds(&self) -> f64 {
        self.completed as f64 * f64::from(DT)
    }

    /// Seconds of the picture per second of the clock over the last RECENT
    /// of play, once two completions fall in it.
    pub fn speed(&self) -> Option<f64> {
        let (first, last) = (self.recent.front()?, self.recent.back()?);
        let span = last.0.duration_since(first.0).as_secs_f64();
        (span > 0.).then(|| (last.1 - first.1) as f64 * f64::from(DT) / span)
    }

    /// Since the picture was last turned over: its own time (s), that of its
    /// completed updates, and the clock's (s), which runs on while the
    /// picture's time is stopped.
    pub fn since_turn(&self) -> [f64; 2] {
        let (at, updates) = self.turned;
        let picture = self.completed.saturating_sub(updates) as f64 * f64::from(DT);
        [picture, at.elapsed().as_secs_f64()]
    }

    /// From now on the kept picture is `size` pixels for the current grid orientation.
    pub fn sync_views(&mut self, gpu: &GpuContext, size: [u32; 2]) {
        self.views
            .set_grid_and_size(&gpu.queue, self.model.grid(), size);
    }

    /// Draws the view `mode` of the present state into the kept picture.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, image: &wgpu::TextureView, mode: Mode) {
        self.views.encode(encoder, image, mode);
    }
}
