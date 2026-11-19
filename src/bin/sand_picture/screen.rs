//! The picture as the window keeps it: the views draw into a texture now and
//! then, and every frame fits that texture into the window. The texture has
//! the window's own pixels wherever the views can draw at that size
//! (`picture_size`), so that a pixel of the picture is a pixel of the display.
use mikage::{GpuContext, RenderContext, RenderTargetConfig};
use sand_picture::view::IMAGE_FORMAT;

/// The window around the picture (linear RGB).
const SURROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

const FIT_WGSL: &str = r"
@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var picture_sampler: sampler;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> Corner {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return Corner(vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(corner.x, 1.0 - corner.y));
}

@fragment
fn fragment(in: Corner) -> @location(0) vec4<f32> {
    return textureSample(picture, picture_sampler, in.uv);
}
";

pub struct Screen {
    size: [u32; 2],
    texture: wgpu::Texture,
    image: wgpu::TextureView,
    bindings: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
}
impl Screen {
    /// A kept picture of `size` pixels.
    pub fn new(gpu: &GpuContext, target: RenderTargetConfig, size: [u32; 2]) -> Self {
        let device = &gpu.device;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("picture fitted to the window"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fit"),
            source: wgpu::ShaderSource::Wgsl(FIT_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: target.sample_count,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let (texture, image, bindings) = kept(device, &pipeline, &sampler, size);
        Self {
            size,
            texture,
            image,
            bindings,
            sampler,
            pipeline,
        }
    }

    /// The kept picture's size in pixels.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// A kept picture of another size; what it showed is gone.
    pub fn resize(&mut self, gpu: &GpuContext, size: [u32; 2]) {
        (self.texture, self.image, self.bindings) =
            kept(&gpu.device, &self.pipeline, &self.sampler, size);
        self.size = size;
    }

    /// What the views draw into.
    pub fn image(&self) -> &wgpu::TextureView {
        &self.image
    }

    /// The kept picture, for a tool that reads it.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The window's frame: its surround and, when there is one, the kept
    /// picture at the largest size of its shape that fits.
    pub fn encode(&self, ctx: &mut RenderContext<'_, ()>, picture: bool) {
        let attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(SURROUND),
            store: wgpu::StoreOp::Store,
        });
        let window = [ctx.target.size.width, ctx.target.size.height].map(|v| v.max(1) as f32);
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("window"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !picture {
            return;
        }
        let image = self.size.map(|v| v as f32);
        let scale = (window[0] / image[0]).min(window[1] / image[1]);
        let size = image.map(|v| v * scale);
        pass.set_viewport(
            (window[0] - size[0]) / 2.,
            (window[1] - size[1]) / 2.,
            size[0],
            size[1],
            0.,
            1.,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The texture the views draw into, at `size`, and its bindings for the fit.
fn kept(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    sampler: &wgpu::Sampler,
    size: [u32; 2],
) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kept picture"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: IMAGE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let image = texture.create_view(&Default::default());
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fit"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (texture, image, bindings)
}
