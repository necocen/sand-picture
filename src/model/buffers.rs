//! Every buffer of the model and its exact GPU layout, created once from the
//! scene. Sizes are bytes; `n` is the number of cells.
use super::grid::{CELLS, cell_width, subdivisions};
use super::material::{self, Grains};
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
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, grains: &Grains) -> Self {
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
        let geometry = blank("local gas identities and contact geometry", cells * 16 * 64);
        // LAYER_ROW: each cell's gas by identity, as (volume, identity, 0, 0).
        let layers = scene
            .gas_layers
            .iter()
            .flat_map(|c| (0..GAS_SLOTS).flat_map(move |l| [c[l], (l + 1) as f32, 0., 0.]))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        queue.write_buffer(&geometry, cells * 16, &layers);
        Self {
            n,
            nx,
            ny,
            config,
            cells: filled(
                device,
                "phase volumes momenta and stress",
                &floats(scene.cell_records()),
            ),
            geometry,
            faces: blank(
                "staggered phase flow and finite displacement",
                cells * 2 * 160,
            ),
            transport: blank("conservative phase flux", cells * 2 * 80),
            members: blank("connected transport member indices", cells * 8),
            next: blank("stress and transport capacity", cells * 16),
            compression: blank("conservative transport reconstruction", cells * 64),
            table: filled(
                device,
                "cross-gap resistance",
                &floats(material::table(grains).into_iter().flatten().collect()),
            ),
            terrain: blank("connected bed geometry", u64::from(nx) * 48),
            // Eight slices per cell height and a three-entry header per column.
            strata: blank("bed strata", u64::from(nx) * u64::from(3 + 8 * ny) * 8),
            arrival: blank("finite arrival source", cells * 96),
            labels: filled(device, "gas components", &labels::initial(n)),
            schedule: blank("common transport schedule", 48),
            gaps: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("slots between bubbles per cell"),
                size: cells * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            contact_list: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("contact fit list"),
                size: 4 * (1 + cells),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            contact_groups: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("contact fit workgroups"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            stream: stream::Fields::new(device),
        }
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
}

/// A zeroed storage buffer that can be read back and written.
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

fn filled(device: &wgpu::Device, name: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(name),
        contents,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn floats(values: Vec<f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
