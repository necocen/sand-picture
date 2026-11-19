//! The control panel: one egui window at an edge of the picture, which hides.
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

pub struct Panel {
    visible: bool,
    side: Side,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            visible: true,
            side: Side::default(),
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

    pub fn show(&mut self, root: &mut egui::Ui, mode: &mut Mode) {
        if !self.visible {
            return;
        }
        let (anchor, offset, other) = match self.side {
            Side::Top => (egui::Align2::CENTER_TOP, [0.0, 8.0], "To bottom"),
            Side::Bottom => (egui::Align2::CENTER_BOTTOM, [0.0, -8.0], "To top"),
        };
        egui::Window::new("controls")
            .title_bar(false)
            .resizable(false)
            .anchor(anchor, offset)
            .show(root.ctx(), |ui| {
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("view")
                        .selected_text(mode.label())
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for view in Mode::ALL {
                                ui.selectable_value(mode, view, view.label());
                            }
                        })
                        .response
                        .on_hover_text("View (V)");
                    ui.separator();
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
            });
    }
}
