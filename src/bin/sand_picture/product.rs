//! The picture as it runs in the window: the model, its views, the count of
//! its updates and what the person has asked of it.
use mikage::{GpuContext, SubmissionToken, TickContext};
use sand_picture::model::{DT, Model};
use sand_picture::view::{Mode, Views};
use std::collections::VecDeque;
use std::time::Duration;
use web_time::Instant;

/// The span of play over which the shown speed is averaged.
const RECENT: Duration = Duration::from_secs(2);

pub struct Product {
    model: Model,
    views: Views,
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
    /// Where the present stretch of play began: when, and the updates then
    /// complete.
    play: (Instant, u64),
    reset_pending: bool,
    turn_pending: bool,
}
impl Product {
    /// The picture rests with its sand on the floor; the person turns it over
    /// at once, as at every reset, and its drainage begins.
    pub fn new(gpu: &GpuContext, mut model: Model) -> Self {
        model.start_turn();
        Self {
            views: Views::new(&gpu.device, model.buffers()),
            model,
            updates: 0,
            completed: 0,
            in_flight: VecDeque::new(),
            recent: VecDeque::new(),
            play: (Instant::now(), 0),
            reset_pending: false,
            turn_pending: false,
        }
    }

    /// The person turns the picture over: it is shaken, then tipped, during
    /// the next updates. A turn under way goes on unchanged.
    pub fn turn(&mut self) {
        self.turn_pending = true;
    }

    /// Back to the scene at rest, turned over at once.
    pub fn reset(&mut self) {
        self.reset_pending = true;
    }

    /// Applies what was asked for since the last update. True when the
    /// picture changed without an update: after a reset.
    pub fn apply_pending(&mut self, gpu: &GpuContext) -> bool {
        let reset = std::mem::take(&mut self.reset_pending);
        if reset {
            self.model.reset(&gpu.device, &gpu.queue);
            self.model.start_turn();
            self.updates = 0;
            self.completed = 0;
            self.in_flight.clear();
            self.turn_pending = false;
            self.play_begins();
        }
        if std::mem::take(&mut self.turn_pending) {
            self.model.start_turn();
        }
        reset
    }

    /// One update, in the tick's encoder.
    pub fn tick(&mut self, ctx: &mut TickContext<'_>) {
        self.apply_pending(ctx.gpu);
        let mirror = self.model.advance_turn(&ctx.gpu.queue);
        {
            let mut pass = ctx.encoder.begin_compute_pass(&Default::default());
            if mirror {
                self.model.invert(&mut pass);
            }
            self.model.encode(&mut pass);
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

    /// A stretch of play begins: after a pause or a reset the speeds are
    /// measured anew.
    pub fn play_begins(&mut self) {
        self.recent.clear();
        self.play = (Instant::now(), self.completed);
    }

    /// Updates encoded since the scene: what a picture drawn now shows.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Updates the GPU has completed since the scene.
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

    /// The same over the present stretch of play, from its beginning.
    pub fn mean_speed(&self) -> Option<f64> {
        let last = self.recent.back()?;
        let span = last.0.duration_since(self.play.0).as_secs_f64();
        (span > 0.).then(|| (last.1 - self.play.1) as f64 * f64::from(DT) / span)
    }

    /// Draws the view `mode` of the present state into the kept picture.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, image: &wgpu::TextureView, mode: Mode) {
        self.views.encode(encoder, image, mode);
    }
}
