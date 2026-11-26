//! The control panel: one egui window at an edge of the picture, which hides
//! and comes back at a click on the picture. In a window narrower than its
//! row of controls the row wraps.
use super::sensor::SensorState;
use mikage::egui;
use sand_picture::view::Mode;

/// The window's edge the panel sits at. The top is the default: the deposit,
/// which is what one watches, builds at the bottom.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Side {
    #[default]
    Top,
    Bottom,
}

/// What a button of the panel, or its key, asks of the picture.
#[derive(Clone, Copy)]
pub enum Command {
    /// Turn the picture over.
    Turn,
    /// Toggle automatic turning once sand finishes falling.
    ToggleAuto,
    /// Toggle device tilt / accelerometer gravity when available.
    ToggleTilt,
    /// Shake the picture left and right for a moment.
    Shake,
    /// Stop the picture's time, or let it run again.
    Pause,
    /// Back to the scene at rest, turned over at once.
    Reset,
    /// Remove a step of air (uniform shrink across bubbles).
    AirLess,
    /// Inject a step of air (spawn bubbles in open water above the bed).
    AirMore,
}

/// What the panel shows of the running picture.
pub struct Readout {
    /// The picture's own time (s).
    pub seconds: f64,
    pub paused: bool,
    pub auto_flip: bool,
    pub sensor: SensorState,
    /// Target total air area (cells^2) and percentage of default.
    pub air_area: f32,
    pub air_percent: u32,
    /// Seconds of the picture per second of the clock over the last two
    /// seconds.
    pub speed: Option<f64>,
    /// Since the picture was last turned over: its own time, and the
    /// clock's (s).
    pub since_turn: [f64; 2],
}

pub struct Panel {
    visible: bool,
    side: Side,
    /// Whether the times since the turn are unfolded.
    details: bool,
    /// True when the top panel reaches close enough to the right edge that the
    /// web repo badge should move to the bottom-right to avoid overlapping it.
    #[cfg(target_family = "wasm")]
    badge_bottom: bool,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            visible: true,
            side: Side::default(),
            details: false,
            #[cfg(target_family = "wasm")]
            badge_bottom: false,
        }
    }
}

/// Horizontal clearance (points) reserved on the right edge for the web repo badge.
#[cfg(target_family = "wasm")]
const BADGE_CLEARANCE: f32 = 204.0;

/// The gap between the panel and the window's edges (points).
const GAP: f32 = 8.0;

/// The width of a line between two groups of the row (points).
const DIVIDER: f32 = 6.0;

/// A line between two groups of the row - unless the control after it,
/// `next` wide, would start the next row of a wrapped panel: a line at a
/// row's end divides nothing.
fn divider(ui: &mut egui::Ui, next: f32) {
    if ui.available_size_before_wrap().x >= DIVIDER + ui.spacing().item_spacing.x + next {
        ui.add(egui::Separator::default().spacing(DIVIDER));
    }
}

/// A line of text laid out in a text style, unwrapped.
fn line(ui: &egui::Ui, text: &str, style: egui::TextStyle) -> std::sync::Arc<egui::Galley> {
    let font = style.resolve(ui.style());
    ui.painter()
        .layout_no_wrap(text.to_owned(), font, ui.visuals().text_color())
}

/// A span of time as minutes and seconds, with hours from the first.
fn span(seconds: f64) -> String {
    let seconds = seconds as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

impl Panel {
    /// Hide the panel, or show it again.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Fold the times since the turn away, or unfold them: the agent's
    /// `details` (the button does it in the window).
    #[cfg(not(target_family = "wasm"))]
    pub fn toggle_details(&mut self) {
        self.details = !self.details;
    }

    /// Move the panel to the window's other edge.
    pub fn change_side(&mut self) {
        self.side = match self.side {
            Side::Top => Side::Bottom,
            Side::Bottom => Side::Top,
        };
    }

    /// Visibility and corner placement `(visible, badge_at_bottom)` for the
    /// web repository badge so it hides with the panel and avoids overlap.
    #[cfg(target_family = "wasm")]
    pub fn web_badge(&self) -> (bool, bool) {
        (self.visible, self.badge_bottom)
    }

    pub fn show(
        &mut self,
        root: &mut egui::Ui,
        mode: &mut Mode,
        readout: &Readout,
    ) -> Option<Command> {
        if !self.visible {
            // A click or a tap on the picture brings the hidden panel back:
            // a phone has no key to ask for it.
            if !root.ctx().input(|input| input.pointer.primary_clicked()) {
                return None;
            }
            self.visible = true;
        }
        let (anchor, offset, other) = match self.side {
            Side::Top => (egui::Align2::CENTER_TOP, [0.0, GAP], "To bottom"),
            Side::Bottom => (egui::Align2::CENTER_BOTTOM, [0.0, -GAP], "To top"),
        };
        let mut command = None;
        let window = egui::Window::new("controls")
            .title_bar(false)
            .resizable(false)
            .anchor(anchor, offset)
            .show(root.ctx(), |ui| {
                // The panel is no wider than the window less a gap at each
                // side: upright on a phone the row of controls is wider than
                // that and wraps, where its ends would lie out of reach.
                let room = ui.ctx().content_rect().width()
                    - 2.0 * GAP
                    - ui.spacing().window_margin.sum().x;
                ui.set_max_width(room);
                let row = ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Turn")
                        .on_hover_text("Turn the picture over (F)")
                        .clicked()
                    {
                        command = Some(Command::Turn);
                    }
                    if ui
                        .button("Shake")
                        .on_hover_text("Shake the picture left and right for a moment (S)")
                        .clicked()
                    {
                        command = Some(Command::Shake);
                    }
                    let pause = if readout.paused { "Play" } else { "Pause" };
                    if ui
                        .button(pause)
                        .on_hover_text("Stop or run the picture's time (Space)")
                        .clicked()
                    {
                        command = Some(Command::Pause);
                    }
                    if ui
                        .button("Reset")
                        .on_hover_text("Back to the picture at rest, turned over at once (R)")
                        .clicked()
                    {
                        command = Some(Command::Reset);
                    }
                    let mut auto = readout.auto_flip;
                    let auto_w = line(ui, "Auto Turn", egui::TextStyle::Button).size().x + 28.0;
                    divider(ui, auto_w);
                    if ui
                        .checkbox(&mut auto, "Auto Turn")
                        .on_hover_text(
                            "Automatically turn the picture over once sand finishes falling (A)",
                        )
                        .changed()
                    {
                        command = Some(Command::ToggleAuto);
                    }
                    if readout.sensor != SensorState::Unavailable {
                        let mut tilt = readout.sensor.is_enabled();
                        let label = match readout.sensor {
                            SensorState::Active => "Tilt",
                            SensorState::Starting => "Tilt…",
                            SensorState::NeedsTap => "Tilt (tap)",
                            SensorState::Denied => "Tilt (denied)",
                            SensorState::Off | SensorState::Unavailable => "Tilt",
                        };
                        let hint = match readout.sensor {
                            SensorState::Active => {
                                "Gravity follows device tilt, shake, and upside-down flip (G)"
                            }
                            SensorState::Starting => "Waiting for motion sensor readings (G)",
                            SensorState::NeedsTap => {
                                "Tap the button at the bottom to allow motion sensor access"
                            }
                            SensorState::Denied => {
                                "Motion sensor permission denied; click to retry (G)"
                            }
                            SensorState::Off | SensorState::Unavailable => {
                                "Use device tilt, shake, and upside-down flip for gravity (G)"
                            }
                        };
                        if ui.checkbox(&mut tilt, label).on_hover_text(hint).changed() {
                            command = Some(Command::ToggleTilt);
                        }
                    }
                    let details = line(ui, "Details", egui::TextStyle::Button).size().x
                        + 2.0 * ui.spacing().button_padding.x;
                    divider(ui, details);
                    ui.toggle_value(&mut self.details, "Details")
                        .on_hover_text("Air, view, panel position, and timings");
                    if ui
                        .button("Hide")
                        .on_hover_text("Hide the panel; a click on the picture or H shows it again")
                        .clicked()
                    {
                        self.visible = false;
                    }
                });
                if self.details {
                    let [picture, clock] = readout.since_turn;
                    // The clock runs on while the picture's time is stopped
                    // and no picture asks for a frame.
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_secs(1));
                    let speed = match (readout.paused, readout.speed) {
                        (true, _) => "paused".to_owned(),
                        (false, Some(speed)) => format!("{speed:.2}x"),
                        (false, None) => "-x".to_owned(),
                    };
                    let stats = format!(
                        "{:.1} s ({speed})  turn {} / clock {}",
                        readout.seconds,
                        span(picture),
                        span(clock)
                    );
                    let style = egui::TextStyle::Monospace;
                    let stats_w = line(ui, &stats, style).size().x;
                    ui.set_max_width(row.response.rect.width().max(stats_w).min(room));
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        ui.label("View");
                        egui::ComboBox::from_id_salt("view")
                            .selected_text(mode.label())
                            .width(92.0)
                            .show_ui(ui, |ui| {
                                for view in Mode::ALL {
                                    ui.selectable_value(mode, view, view.label());
                                }
                            })
                            .response
                            .on_hover_text("View (V)");
                        divider(ui, 110.0);
                        ui.label("Air");
                        if ui
                            .button("-")
                            .on_hover_text(format!(
                                "Reduce air volume ({:.0} cells^2) (-)",
                                readout.air_area
                            ))
                            .clicked()
                        {
                            command = Some(Command::AirLess);
                        }
                        ui.monospace(format!("{:3}%", readout.air_percent))
                            .on_hover_text(format!("{:.0} cells^2", readout.air_area));
                        if ui
                            .button("+")
                            .on_hover_text(format!(
                                "Inject air bubbles above the bed ({:.0} cells^2) (=)",
                                readout.air_area
                            ))
                            .clicked()
                        {
                            command = Some(Command::AirMore);
                        }
                        divider(ui, 72.0);
                        if ui
                            .button(other)
                            .on_hover_text("Move the panel (P)")
                            .clicked()
                        {
                            self.change_side();
                        }
                    });
                    ui.monospace(stats)
                        .on_hover_text("Picture time, speed, and time since last turn");
                }
            });
        #[cfg(target_family = "wasm")]
        if let Some(inner) = window {
            let viewport_right = root.ctx().content_rect().max.x;
            self.badge_bottom = self.side == Side::Top
                && inner.response.rect.max.x + BADGE_CLEARANCE > viewport_right;
        }
        #[cfg(not(target_family = "wasm"))]
        let _ = window;
        command
    }
}
