//! The control panel: one egui window at an edge of the picture, which hides
//! and comes back at a click on the picture.
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
    /// Shake the picture left and right for a moment.
    Shake,
    /// Stop the picture's time, or let it run again.
    Pause,
    /// Back to the scene at rest, turned over at once.
    Reset,
}

/// What the panel shows of the running picture.
pub struct Readout {
    /// The picture's own time (s).
    pub seconds: f64,
    pub paused: bool,
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
            Side::Top => (egui::Align2::CENTER_TOP, [0.0, 8.0], "To bottom"),
            Side::Bottom => (egui::Align2::CENTER_BOTTOM, [0.0, -8.0], "To top"),
        };
        let mut command = None;
        egui::Window::new("controls")
            .title_bar(false)
            .resizable(false)
            .anchor(anchor, offset)
            .show(root.ctx(), |ui| {
                ui.horizontal(|ui| {
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
                    ui.separator();
                    ui.monospace(format!("{:7.1} s", readout.seconds))
                        .on_hover_text("The picture's own time");
                    let speed = match (readout.paused, readout.speed) {
                        (true, _) => " paused".to_owned(),
                        (false, Some(speed)) => format!("{speed:5.2} x"),
                        (false, None) => "   -  x".to_owned(),
                    };
                    ui.monospace(speed)
                        .on_hover_text("Seconds of the picture per second, over the last 2 s");
                    ui.separator();
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
                    ui.separator();
                    let [picture, clock] = readout.since_turn;
                    // The clock runs on while the picture's time is stopped
                    // and no picture asks for a frame.
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_secs(1));
                    ui.monospace(format!(
                        "since the turn  {:>7} of the picture  {:>7} by the clock",
                        span(picture),
                        span(clock)
                    ));
                }
            });
        command
    }
}
