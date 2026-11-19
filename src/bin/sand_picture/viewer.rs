//! The window's application: keys, the control panel and the frame.
use super::panel::Panel;
use mikage::winit::keyboard::KeyCode;
use mikage::{App, RenderContext, WindowInputContext};
use sand_picture::view::Mode;

/// The window around the picture (linear RGB).
const SURROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

#[derive(Default)]
pub struct Viewer {
    panel: Panel,
    mode: Mode,
}
impl App for Viewer {
    type Camera = ();

    fn on_input(&mut self, ctx: &mut WindowInputContext<()>) {
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

    fn render(&mut self, ctx: &mut RenderContext<()>) {
        let attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(SURROUND),
            store: wgpu::StoreOp::Store,
        });
        ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("window"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        self.panel.show(ui, &mut self.mode);
    }
}
