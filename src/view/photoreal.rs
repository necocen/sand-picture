//! The Photoreal view: before each picture a compute pass sketches the
//! state, cell by cell, as what the picture draws there (sketch.wgsl); then
//! the picture is drawn from the sketch as smooth fields (photoreal.wgsl).
use super::picture::Picture;
use super::{CELLS_WGSL, SCENE_WGSL};
use crate::model::{Buffers, row_constants};
use crate::wgsl;

/// The shader of the kernel that sketches the state. It addresses the
/// model's gas by the model's own constants.
pub fn sketch_source() -> String {
    [
        &row_constants(),
        SCENE_WGSL,
        CELLS_WGSL,
        include_str!("sketch.wgsl"),
    ]
    .concat()
}

/// The picture's shader.
pub fn source() -> String {
    [
        &row_constants(),
        SCENE_WGSL,
        include_str!("optics.wgsl"),
        include_str!("studio.wgsl"),
        CELLS_WGSL,
        include_str!("deposit.wgsl"),
        include_str!("photoreal.wgsl"),
    ]
    .concat()
}

/// The bytes of a cell's facts and of its sketch (three vec4 each,
/// sketch.wgsl).
const SKETCH_BYTES: u64 = 48;

/// A whole buffer at a binding.
fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub struct Photoreal {
    /// The kernels that sketch the state, in their order: each cell's own
    /// facts, then what the picture draws of it, which looks at the cells
    /// around it. Each with its bindings.
    sketch: [(wgpu::ComputePipeline, wgpu::BindGroup); 2],
    /// The kernels' groups across and up the grid.
    groups: [u32; 2],
    picture: Picture,
}
impl Photoreal {
    pub fn new(device: &wgpu::Device, config: &wgpu::Buffer, buffers: &Buffers) -> Self {
        let (nx, ny) = (buffers.nx, buffers.ny);
        let buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(nx * ny) * SKETCH_BYTES,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let facts = buffer("the cells' own facts");
        let cells = buffer("the picture's sketch of the state");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sketch"),
            source: wgpu::ShaderSource::Wgsl(wgsl::compiled(sketch_source()).into()),
        });
        let kernel = |name, entries: &[wgpu::BindGroupEntry<'_>]| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
            (pipeline, bindings)
        };
        let sketch = [
            kernel(
                "survey_cells",
                &[
                    entry(0, config),
                    entry(1, &buffers.cells),
                    entry(2, &buffers.geometry),
                    entry(3, &facts),
                    entry(5, &buffers.terrain),
                ],
            ),
            kernel(
                "sketch_cells",
                &[
                    entry(0, config),
                    entry(1, &buffers.cells),
                    entry(3, &facts),
                    entry(4, &cells),
                    entry(5, &buffers.terrain),
                ],
            ),
        ];
        let picture = Picture::new(
            device,
            "photoreal",
            source(),
            &[
                entry(0, config),
                entry(1, &buffers.cells),
                entry(2, &cells),
                entry(4, &buffers.strata),
                entry(5, &buffers.terrain),
            ],
        );
        Self {
            sketch,
            groups: [nx.div_ceil(8), ny.div_ceil(8)],
            picture,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, image: &wgpu::TextureView) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sketch"),
                timestamp_writes: None,
            });
            for (pipeline, bindings) in &self.sketch {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bindings, &[]);
                pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
            }
        }
        self.picture.encode(encoder, image);
    }
}
