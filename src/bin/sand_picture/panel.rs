//! The control panel: one egui window at an edge of the picture, which hides.
use mikage::egui;
use sand_picture::model::{CELLS, TANK};
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
    /// Seconds of the picture per second of the clock: over the last two
    /// seconds, and since play began.
    pub speed: Option<f64>,
    pub mean_speed: Option<f64>,
    /// Updates the GPU has completed, and updates encoded.
    pub updates: [u64; 2],
}

pub struct Panel {
    visible: bool,
    side: Side,
    /// Whether the finer numbers are unfolded.
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
impl Panel {
    /// Hide the panel, or show it again.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
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
            return None;
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
                        .button("Turn over")
                        .on_hover_text("Shake the picture and turn it over (F)")
                        .clicked()
                    {
                        command = Some(Command::Turn);
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
                    ui.toggle_value(&mut self.details, "Details");
                    if ui
                        .button(other)
                        .on_hover_text("Move the panel (P)")
                        .clicked()
                    {
                        self.change_side();
                    }
                    if ui
                        .button("Hide")
                        .on_hover_text("Hide the panel; H shows it again")
                        .clicked()
                    {
                        self.visible = false;
                    }
                });
                if self.details {
                    ui.separator();
                    let [done, encoded] = readout.updates;
                    ui.monospace(format!("updates  {done} done of {encoded}, 8 ms each"));
                    ui.monospace(match readout.mean_speed {
                        Some(speed) => format!("speed    {speed:.2} x since play began"),
                        None => "speed    -".to_owned(),
                    });
                    ui.monospace(format!(
                        "tank     {:.0} x {:.0} cm, {} x {} cells",
                        TANK[0] * 100.,
                        TANK[1] * 100.,
                        CELLS[0],
                        CELLS[1]
                    ));
                    ui.monospace("keys     F turn  Space pause  R reset  V view  H panel  P side");
                }
            });
        command
    }
}
