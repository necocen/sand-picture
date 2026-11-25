//! The control panel: one egui window at an edge of the picture, which hides
//! and comes back at a click on the picture. In a window narrower than its
//! row of controls the row wraps.
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
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            visible: true,
            side: Side::default(),
            details: false,
        }
    }
}

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

/// A number of the picture in a cell as high as the buttons: a bare label
/// that starts a row of a wrapped panel would sit at the row's top.
fn number(ui: &mut egui::Ui, text: std::sync::Arc<egui::Galley>) -> egui::Response {
    let buttons =
        ui.text_style_height(&egui::TextStyle::Button) + 2.0 * ui.spacing().button_padding.y;
    let size = egui::vec2(text.size().x, buttons.max(ui.spacing().interact_size.y));
    let centred = egui::Layout::left_to_right(egui::Align::Center);
    ui.allocate_ui_with_layout(size, centred, |ui| ui.label(text))
        .inner
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
        egui::Window::new("controls")
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
                    let mut auto = readout.auto_flip;
                    if ui
                        .toggle_value(&mut auto, "Auto")
                        .on_hover_text(
                            "Automatically turn the picture over once sand finishes falling (A)",
                        )
                        .changed()
                    {
                        command = Some(Command::ToggleAuto);
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
                    if ui
                        .button("Air -")
                        .on_hover_text(format!(
                            "Reduce air volume ({:.0} cells^2, {}%) (-)",
                            readout.air_area, readout.air_percent
                        ))
                        .clicked()
                    {
                        command = Some(Command::AirLess);
                    }
                    if ui
                        .button("Air +")
                        .on_hover_text(format!(
                            "Inject air bubbles above the bed ({:.0} cells^2, {}%) (=)",
                            readout.air_area, readout.air_percent
                        ))
                        .clicked()
                    {
                        command = Some(Command::AirMore);
                    }
                    egui::ComboBox::from_id_salt("view")
                        .selected_text(mode.label())
                        .width(96.0)
                        .show_ui(ui, |ui| {
                            for view in Mode::ALL {
                                ui.selectable_value(mode, view, view.label());
                            }
                        })
                        .response
                        .on_hover_text("View (V)");
                    let seconds = format!("{:7.1} s", readout.seconds);
                    let seconds = line(ui, &seconds, egui::TextStyle::Monospace);
                    divider(ui, seconds.size().x);
                    number(ui, seconds).on_hover_text("The picture's own time");
                    let speed = match (readout.paused, readout.speed) {
                        (true, _) => " paused".to_owned(),
                        (false, Some(speed)) => format!("{speed:5.2} x"),
                        (false, None) => "   -  x".to_owned(),
                    };
                    number(ui, line(ui, &speed, egui::TextStyle::Monospace))
                        .on_hover_text("Seconds of the picture per second, over the last 2 s");
                    let details = line(ui, "Details", egui::TextStyle::Button).size().x
                        + 2.0 * ui.spacing().button_padding.x;
                    divider(ui, details);
                    ui.toggle_value(&mut self.details, "Details")
                        .on_hover_text("The times since the picture was turned over");
                    if ui
                        .button(other)
                        .on_hover_text("Move the panel (P)")
                        .clicked()
                    {
                        self.change_side();
                    }
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
                    let lead = "since the turn  ";
                    let picture = format!("{lead}{:>7} of the picture", span(picture));
                    let clock = format!("{:>7} by the clock", span(clock));
                    let whole = format!("{picture}  {clock}");
                    // The times are as wide as the row above them, not as
                    // the window - or as the picture's time needs, where a
                    // wrapped row is narrower than that.
                    let style = egui::TextStyle::Monospace;
                    let least = line(ui, &picture, style.clone()).size().x;
                    ui.set_max_width(row.response.rect.width().max(least).min(room));
                    ui.separator();
                    let fits = line(ui, &whole, style).size().x <= ui.available_width();
                    if fits {
                        ui.monospace(whole);
                    } else {
                        // Too wide for the window: the clock's time goes
                        // under the picture's.
                        ui.monospace(picture);
                        ui.monospace(format!("{:1$}{clock}", "", lead.len()));
                    }
                }
            });
        command
    }
}
