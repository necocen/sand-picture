//! The picture's views: independent drawings of the model's state as the last
//! complete update left it. They read the model's buffers and write none.
mod photoreal;
mod picture;

use crate::model::{Buffers, Grains, POWDER_DIAMETER, cell_width};
use photoreal::Photoreal;
use picture::Picture;
use wgpu::util::DeviceExt;

/// The image of the whole tank that the views draw (pixels). Optical
/// sampling only: the simulation stays on its own grid.
pub const IMAGE_SIZE: [u32; 2] = [1440, 960];
/// The format of the image the views draw into.
pub const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The gap between the tank's plates (m), the model's PLATE_GAP.
const PLATE_GAP: f32 = 0.001;

/// The first fragment of every view's shader: the views' configuration.
const SCENE_WGSL: &str = include_str!("view/scene.wgsl");
/// The model's cells as the views read them.
const CELLS_WGSL: &str = include_str!("view/cells.wgsl");

fn simple_source() -> String {
    [SCENE_WGSL, CELLS_WGSL, include_str!("view/simple.wgsl")].concat()
}
fn flow_source() -> String {
    [SCENE_WGSL, CELLS_WGSL, include_str!("view/flow.wgsl")].concat()
}

/// How the picture is drawn: the cells' plain colours, the optical picture,
/// or the flow's direction and speed as colours.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    Simple,
    #[default]
    Photoreal,
    Flow,
}
impl Mode {
    pub const ALL: [Self; 3] = [Self::Simple, Self::Photoreal, Self::Flow];

    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Photoreal => "Photoreal",
            Self::Flow => "Flow",
        }
    }

    /// The view after this one, around the list.
    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }
}

pub struct Views {
    /// Every pixel in its cell's phase colours.
    simple: Picture,
    photoreal: Photoreal,
    /// Where everything in the tank is going, as colour.
    flow: Picture,
}
impl Views {
    pub fn new(device: &wgpu::Device, buffers: &Buffers) -> Self {
        let [fine, coarse] = Grains::default().diameters.map(|d| d as f32);
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("views' configuration"),
            contents: &[
                buffers.nx,
                buffers.ny,
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                fine.to_bits(),
                coarse.to_bits(),
                cell_width().to_bits(),
                PLATE_GAP.to_bits(),
                (POWDER_DIAMETER as f32).to_bits(),
                0,
                0,
                0,
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Simple and Flow read the cells alone.
        let cells = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.cells.as_entire_binding(),
            },
        ];
        Self {
            simple: Picture::new(device, "simple", simple_source(), &cells),
            photoreal: Photoreal::new(device, &config, buffers),
            flow: Picture::new(device, "flow", flow_source(), &cells),
        }
    }

    /// Draws the view `mode` of the model's present state into `image`, a
    /// view of an IMAGE_SIZE texture of IMAGE_FORMAT.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        image: &wgpu::TextureView,
        mode: Mode,
    ) {
        match mode {
            Mode::Simple => self.simple.encode(encoder, image),
            Mode::Photoreal => self.photoreal.encode(encoder, image),
            Mode::Flow => self.flow.encode(encoder, image),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, SCENE_WGSL, flow_source, photoreal, simple_source};
    use crate::wgsl;

    #[test]
    fn every_view_is_valid_wgsl() {
        wgsl::assert_valid("simple", &simple_source());
        wgsl::assert_valid("flow", &flow_source());
        wgsl::assert_valid("photoreal", &photoreal::source());
        for (name, source) in [
            ("distance field", photoreal::GEOMETRY_WGSL),
            ("near layers", photoreal::LAYERS_WGSL),
        ] {
            wgsl::assert_valid(name, &[SCENE_WGSL, source].concat());
        }
    }

    #[test]
    fn next_goes_around_every_view() {
        let mut mode = Mode::default();
        let mut seen = Vec::new();
        for _ in Mode::ALL {
            seen.push(mode);
            mode = mode.next();
        }
        assert_eq!(mode, Mode::default());
        for view in Mode::ALL {
            assert!(seen.contains(&view), "{view:?} is never reached");
        }
    }
}
