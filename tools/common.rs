//! What the tools share: a GPU without a window, readbacks, and the views'
//! pictures.
use mikage::{GpuContext, GpuDescriptor, GpuRequirements};
use sand_picture::model::Model;
use sand_picture::view::{IMAGE_FORMAT, Mode, Views};
use std::path::Path;

/// The GPU as the app asks for it: WebGPU's default limits. The tools also
/// take the GPU's timestamps where it has them (`run --time-stages`).
pub fn headless() -> GpuContext {
    pollster::block_on(GpuContext::headless(GpuDescriptor {
        requirements: GpuRequirements {
            required_limits: Some(wgpu::Limits::default()),
            optional_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap()
}

/// Waits until the GPU has done everything submitted.
pub fn wait(gpu: &GpuContext) {
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
}

/// A buffer's bytes now.
pub fn read(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    wait(gpu);
    staging.slice(..).get_mapped_range().unwrap().to_vec()
}

/// The views and the image they draw into, as a window keeps them: one set
/// for the whole run, since a view may keep marks from picture to picture.
pub struct Pictures {
    views: Views,
    image: wgpu::Texture,
    size: [u32; 2],
}
impl Pictures {
    /// Pictures of `size` pixels.
    pub fn new(gpu: &GpuContext, model: &Model, size: [u32; 2]) -> Self {
        Self::of(gpu, Views::new(&gpu.device, model.buffers(), size), size)
    }

    /// Pictures of `size` pixels drawn by `views`, which draw at that size.
    pub fn of(gpu: &GpuContext, views: Views, size: [u32; 2]) -> Self {
        Self {
            size,
            views,
            image: image(gpu, size),
        }
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, mode: Mode) {
        self.views
            .encode(encoder, &self.image.create_view(&Default::default()), mode);
    }

    /// The picture of `mode` now: its pixels as RGBA, the top row first.
    pub fn picture(&self, gpu: &GpuContext, mode: Mode) -> Vec<u8> {
        let [width, height] = self.size;
        // A copied row is padded to wgpu's row alignment.
        let row = (4 * width).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picture readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.draw(&mut encoder, mode);
        encoder.copy_texture_to_buffer(
            self.image.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            self.image.size(),
        );
        gpu.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        wait(gpu);
        staging
            .slice(..)
            .get_mapped_range()
            .unwrap()
            .chunks_exact(row as usize)
            .flat_map(|padded| &padded[..4 * width as usize])
            .copied()
            .collect()
    }
}

/// The texture a picture of `size` pixels is drawn into.
fn image(gpu: &GpuContext, size: [u32; 2]) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("picture"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: IMAGE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// The name a view's picture is kept under: `image` is Photoreal, as in
/// v2.20's baseline.
pub fn picture_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Simple => "simple",
        Mode::Photoreal => "image",
        Mode::Flow => "flow",
    }
}

/// Writes a picture's RGBA pixels, `size` pixels, as a PNG.
pub fn write_png(path: &Path, pixels: &[u8], size: [u32; 2]) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size[0], size[1]);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(pixels)
        .unwrap();
}
