//! The picture's views: independent drawings of the model's state as the last
//! complete update left it. They read the model's buffers and write none.
mod photoreal;
mod picture;

use crate::model::{Buffers, CELLS, Grains, POWDER_DIAMETER, TANK, cell_width};
use photoreal::Photoreal;
use picture::Picture;
use wgpu::util::DeviceExt;

/// The size (pixels) of the pictures the regression compares. A picture is
/// optical sampling only, at whatever size: the simulation stays on its own
/// grid.
///
/// It is also the coarsest picture a window gets. Photoreal integrates a
/// bubble's rim over a pixel as if the outline were straight across it,
/// which holds for a pixel of about a fifth of the plates' gap, as here; at
/// half this size the bubbles of the raft come out grey. And a coarser
/// picture would save little: most of a picture's cost is the distance
/// field, which does not depend on its size.
pub const REFERENCE_SIZE: [u32; 2] = [1440, 960];
/// The format of the image the views draw into.
pub const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The gap between the tank's plates (m), the model's PLATE_GAP.
const PLATE_GAP: f32 = 0.001;

/// The first fragment of every view's shader: the views' configuration.
const SCENE_WGSL: &str = include_str!("view/scene.wgsl");
/// The model's cells as the views read them.
const CELLS_WGSL: &str = include_str!("view/cells.wgsl");

/// The size of the picture for a window of `available` pixels: the largest
/// in the tank's shape that fits, but no coarser than REFERENCE_SIZE and no
/// finer than the sand itself - a pixel is at least a fine grain wide. A
/// window outside that range shows the picture scaled.
///
/// Built with the feature `fixed-picture`, every window gets the reference
/// picture, scaled to it.
pub fn picture_size(available: [u32; 2]) -> [u32; 2] {
    if cfg!(feature = "fixed-picture") {
        return REFERENCE_SIZE;
    }
    // The tank's shape in whole numbers, as the grid has it.
    let [across, up] = CELLS.map(|cells| cells as u64);
    let finest = (TANK[0] / Grains::default().diameters[0]) as u64;
    let width = u64::from(available[0])
        .min(u64::from(available[1]) * across / up)
        .clamp(u64::from(REFERENCE_SIZE[0]), finest);
    let height = (2 * width * up + across) / (2 * across);
    [width as u32, height as u32]
}

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
    /// The views' configuration: the grid, the picture's size, the grains.
    config: wgpu::Buffer,
    /// Every pixel in its cell's phase colours.
    simple: Picture,
    photoreal: Photoreal,
    /// Where everything in the tank is going, as colour.
    flow: Picture,
}
impl Views {
    /// The views of the model's `buffers`, drawing pictures of `size` pixels.
    pub fn new(device: &wgpu::Device, buffers: &Buffers, size: [u32; 2]) -> Self {
        let [fine, coarse] = Grains::default().diameters.map(|d| d as f32);
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("views' configuration"),
            contents: &[
                buffers.nx,
                buffers.ny,
                size[0],
                size[1],
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            config,
        }
    }

    /// From now on the pictures are `size` pixels.
    pub fn resize(&self, queue: &wgpu::Queue, size: [u32; 2]) {
        let words: Vec<u8> = size.into_iter().flat_map(u32::to_le_bytes).collect();
        queue.write_buffer(&self.config, 8, &words);
    }

    /// Draws the view `mode` of the model's present state into `image`, a
    /// view of a texture of the views' size and of IMAGE_FORMAT.
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
    use super::{Mode, flow_source, photoreal, picture_size, simple_source};
    use crate::wgsl;

    #[cfg(feature = "fixed-picture")]
    #[test]
    fn every_window_gets_the_reference_picture() {
        for window in [[720, 480], [1440, 960], [2560, 1440], [6000, 4000]] {
            assert_eq!(picture_size(window), super::REFERENCE_SIZE);
        }
    }

    #[cfg(not(feature = "fixed-picture"))]
    #[test]
    fn a_picture_has_the_windows_pixels_within_what_the_views_can_draw() {
        // The window's own pixels, in the tank's shape.
        assert_eq!(picture_size([1440, 960]), [1440, 960]);
        assert_eq!(picture_size([2560, 1440]), [2160, 1440]);
        assert_eq!(picture_size([3440, 1300]), [1950, 1300]);
        // No coarser than the reference, no finer than a grain.
        assert_eq!(picture_size([720, 480]), [1440, 960]);
        assert_eq!(picture_size([1440, 900]), [1440, 960]);
        assert_eq!(picture_size([6000, 4000]), [2727, 1818]);
    }

    #[test]
    fn every_view_is_valid_wgsl() {
        wgsl::assert_valid("simple", &simple_source());
        wgsl::assert_valid("flow", &flow_source());
        wgsl::assert_valid("photoreal", &photoreal::source());
        wgsl::assert_valid("sketch", &photoreal::sketch_source());
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
