//! Every buffer of the model and its exact GPU layout, created once and
//! filled from the scene. Sizes are bytes; `n` is the number of cells.
use super::grid::{CELLS, cell_width, subdivisions};
use super::material;
use super::scene::Scene;
use super::{DT, GAS_SLOTS, STANDARD_GRAVITY, labels, stream};
use wgpu::util::DeviceExt;

pub struct Buffers {
    pub n: u32,
    pub nx: u32,
    pub ny: u32,
    /// The tank, the update interval and the gravity of this update (uniform).
    pub config: wgpu::Buffer,
    /// 96 bytes a cell: mobile phase volumes, momenta and stress, the white's
    /// powder, deposited grains and pore water.
    pub cells: wgpu::Buffer,
    /// 64 rows of one vec4 a cell, named in shared/geometry_rows.wgsl.
    pub geometry: wgpu::Buffer,
    /// Two staggered faces a cell, 160 bytes each: response, projected flow
    /// and displacement.
    pub faces: wgpu::Buffer,
    /// Two faces a cell, 80 bytes each: the conservative phase flux.
    pub transport: wgpu::Buffer,
    /// Transport group membership.
    pub members: wgpu::Buffer,
    /// Scratch the stages reuse: the mechanics' working values, then the
    /// transport's capacities.
    pub next: wgpu::Buffer,
    pub compression: wgpu::Buffer,
    /// The cross-gap resistance table of the materials.
    pub table: wgpu::Buffer,
    /// The floor-connected bed's columns.
    pub terrain: wgpu::Buffer,
    /// The record of the bed's layers as they were laid (bed/strata.wgsl).
    pub strata: wgpu::Buffer,
    /// The arrival snapshot of the bed's capture.
    pub arrival: wgpu::Buffer,
    /// The bubbles' relabelling: components, films' counts, cells' words.
    pub labels: wgpu::Buffer,
    /// The transport's subdivision schedule.
    pub schedule: wgpu::Buffer,
    /// The slots between bubbles per cell (slab/joints.wgsl).
    pub gaps: wgpu::Buffer,
    /// The contact fits' list of cells and its indirect workgroup count.
    pub contact_list: wgpu::Buffer,
    pub contact_groups: wgpu::Buffer,
    /// The flow solver's fields.
    pub stream: stream::Fields,
}

impl Buffers {
    /// The buffers of a picture at rest in `scene`, with the materials'
    /// `table` (material::table_strip, every strip in order).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        table: &[[f32; 4]],
    ) -> Self {
        let [nx, ny] = CELLS.map(|v| v as u32);
        let n = nx * ny;
        let cells = u64::from(n);
        let blank = |name, size| blank(device, name, size);
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physical tank and update interval"),
            contents: &[
                nx,
                ny,
                n,
                material::SIDE as u32,
                DT.to_bits(),
                cell_width().to_bits(),
                STANDARD_GRAVITY.to_bits(),
                subdivisions(),
                0f32.to_bits(),
                0,
                0,
                0,
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
            // The in-plane gravity (word 6) changes while the picture is turned,
            // the lateral apparent gravity (word 8) while it is shaken.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffers = Self {
            n,
            nx,
            ny,
            config,
            cells: blank("phase volumes momenta and stress", cells * 96),
            geometry: blank("local gas identities and contact geometry", cells * 16 * 64),
            faces: blank(
                "staggered phase flow and finite displacement",
                cells * 2 * 160,
            ),
            transport: blank("conservative phase flux", cells * 2 * 80),
            members: blank("connected transport member indices", cells * 8),
            next: blank("stress and transport capacity", cells * 16),
            compression: blank("conservative transport reconstruction", cells * 64),
            table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cross-gap resistance"),
                contents: &floats(table.iter().flatten().copied()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            terrain: blank("connected bed geometry", u64::from(nx) * 48),
            // Eight slices per cell height and a three-entry header per column.
            strata: blank("bed strata", u64::from(nx) * u64::from(3 + 8 * ny) * 8),
            arrival: blank("finite arrival source", cells * 96),
            labels: blank("gas components", labels::initial(n).len() as u64),
            schedule: blank("common transport schedule", 48),
            gaps: blank("slots between bubbles per cell", cells * 16),
            contact_list: blank("contact fit list", 4 * (1 + cells)),
            contact_groups: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("contact fit workgroups"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stream: stream::Fields::new(device),
        };
        buffers.fill(queue, scene);
        buffers
    }

    /// The eight buffers a checkpoint compares, by name.
    pub fn named(&self) -> [(&'static str, &wgpu::Buffer); 8] {
        [
            ("cells", &self.cells),
            ("geometry", &self.geometry),
            ("faces", &self.faces),
            ("stream", &self.stream.solution),
            ("transport", &self.transport),
            ("terrain", &self.terrain),
            ("members", &self.members),
            ("schedule", &self.schedule),
        ]
    }

    /// Writes the picture at rest in `scene` into buffers that hold zeros:
    /// the cells, each cell's gas by identity, and the labels of a picture
    /// whose films drained long ago.
    fn fill(&self, queue: &wgpu::Queue, scene: &Scene) {
        queue.write_buffer(&self.cells, 0, &floats(scene.cell_records()));
        // LAYER_ROW: each cell's gas by identity, as (volume, identity, 0, 0).
        let layers = scene
            .gas_layers
            .iter()
            .flat_map(|c| (0..GAS_SLOTS).flat_map(move |l| [c[l], (l + 1) as f32, 0., 0.]));
        queue.write_buffer(&self.geometry, u64::from(self.n) * 16, &floats(layers));
        queue.write_buffer(&self.labels, 0, &labels::initial(self.n));
    }

    /// Back to the picture at rest in `scene`, as `new` leaves the buffers:
    /// everything a run has written is cleared, the standing gravity is
    /// restored and the scene is written again. The pipelines and the
    /// materials' table stay.
    pub fn restore(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let mut encoder = device.create_command_encoder(&Default::default());
        for buffer in [
            &self.cells,
            &self.geometry,
            &self.faces,
            &self.transport,
            &self.members,
            &self.next,
            &self.compression,
            &self.terrain,
            &self.strata,
            &self.arrival,
            &self.labels,
            &self.schedule,
            &self.gaps,
            &self.contact_list,
            &self.contact_groups,
        ]
        .into_iter()
        .chain(self.stream.all())
        {
            encoder.clear_buffer(buffer, 0, None);
        }
        // The clears are submitted before the scene is written: a write
        // waits for the next submission and would be cleared by them.
        queue.submit([encoder.finish()]);
        queue.write_buffer(&self.config, 24, &STANDARD_GRAVITY.to_le_bytes());
        queue.write_buffer(&self.config, 32, &0f32.to_le_bytes());
        self.fill(queue, scene);
    }
}

/// A zeroed storage buffer that can be read back, written and cleared.
pub fn blank(device: &wgpu::Device, name: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
