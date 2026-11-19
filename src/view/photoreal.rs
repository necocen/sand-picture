//! The Photoreal view: before each picture a compute pass rebuilds the
//! distance field of the gas outlines where gas is near (geometry.wgsl) and
//! marks the layers near each cell (layers.wgsl); then the picture is drawn
//! (photoreal.wgsl).
use super::picture::Picture;
use super::{CELLS_WGSL, SCENE_WGSL};
use crate::model::{Buffers, GAS_SLOTS};
use crate::wgsl;
use wgpu::util::DeviceExt;

pub const GEOMETRY_WGSL: &str = include_str!("geometry.wgsl");
pub const LAYERS_WGSL: &str = include_str!("layers.wgsl");

/// The picture's shader.
pub fn source() -> String {
    [
        SCENE_WGSL,
        include_str!("optics.wgsl"),
        include_str!("studio.wgsl"),
        CELLS_WGSL,
        include_str!("deposit.wgsl"),
        include_str!("photoreal.wgsl"),
    ]
    .concat()
}

pub struct Photoreal {
    /// The kernels before the picture, in their order, each with the layers
    /// it is dispatched over: the identities each cell shows, those shown
    /// within four cells of it, the distance field, the layers a cell's
    /// outline samples.
    kernels: [(wgpu::ComputePipeline, wgpu::BindGroup, u32); 4],
    groups: [u32; 2],
    picture: Picture,
}
impl Photoreal {
    pub fn new(device: &wgpu::Device, config: &wgpu::Buffer, buffers: &Buffers) -> Self {
        let (nx, ny) = (buffers.nx, buffers.ny);
        let distances = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("distance field of the gas outlines"),
                size: wgpu::Extent3d {
                    width: nx,
                    height: ny,
                    depth_or_array_layers: GAS_SLOTS as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let module = |label, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(
                    wgsl::compiled([SCENE_WGSL, source].concat()).into(),
                ),
            })
        };
        let geometry = module("distance field", GEOMETRY_WGSL);
        let near_layers = module("gas layers near each cell", LAYERS_WGSL);
        let words = |label, word: u32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &word.to_le_bytes().repeat((nx * ny) as usize),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let shown = words("gas identities each cell shows", 0);
        // Every identity counts as near at a picture before the first: the
        // first picture writes the whole field.
        let near = words("gas identities shown near each cell", (1 << GAS_SLOTS) - 1);
        let layers = words("gas layers near each cell", 0);
        let kernel = |module, entry, depth, entries: &[(u32, wgpu::BindingResource<'_>)]| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries
                    .iter()
                    .map(|(binding, resource)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: resource.clone(),
                    })
                    .collect::<Vec<_>>(),
            });
            (pipeline, bindings, depth)
        };
        let field = || wgpu::BindingResource::TextureView(&distances);
        let kernels = [
            kernel(
                &geometry,
                "mark_shown",
                1,
                &[
                    (0, config.as_entire_binding()),
                    (2, buffers.geometry.as_entire_binding()),
                    (4, shown.as_entire_binding()),
                    (5, near.as_entire_binding()),
                ],
            ),
            kernel(
                &geometry,
                "mark_near",
                1,
                &[
                    (0, config.as_entire_binding()),
                    (4, shown.as_entire_binding()),
                    (5, near.as_entire_binding()),
                ],
            ),
            kernel(
                &geometry,
                "prepare_geometry",
                GAS_SLOTS as u32,
                &[
                    (0, config.as_entire_binding()),
                    (2, buffers.geometry.as_entire_binding()),
                    (3, field()),
                    (5, near.as_entire_binding()),
                ],
            ),
            kernel(
                &near_layers,
                "mark_layers",
                1,
                &[
                    (0, config.as_entire_binding()),
                    (1, field()),
                    (2, layers.as_entire_binding()),
                ],
            ),
        ];
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("distance interpolation"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let picture = Picture::new(
            device,
            "photoreal",
            source(),
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: field(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.strata.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.terrain.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: layers.as_entire_binding(),
                },
            ],
        );
        Self {
            kernels,
            groups: [nx.div_ceil(8), ny.div_ceil(8)],
            picture,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, image: &wgpu::TextureView) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("distance field and near layers"),
                timestamp_writes: None,
            });
            for (pipeline, bindings, depth) in &self.kernels {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bindings, &[]);
                pass.dispatch_workgroups(self.groups[0], self.groups[1], *depth);
            }
        }
        self.picture.encode(encoder, image);
    }
}
