//! One dispatch of the update: a compute pipeline, the buffers it binds and
//! its workgroup count. A program is one shader module and a pipeline is
//! made once for an entry point; the kernels that run it differ in what they
//! bind. A kernel names its buffers as its program does: by the names of the
//! program's globals, whose `@binding` numbers stay in the WGSL alone.
use super::buffers::Buffers;
use super::material::Grains;
use super::shader::Program;
use super::uses::uses;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// A fixed workgroup count.
#[derive(Clone)]
pub enum Groups {
    Fixed(u32),
}

/// What a kernel is made from.
#[derive(Clone)]
pub struct Plan {
    pub program: Program,
    pub entry: &'static str,
    /// The words of the kernel's own `cfg`, for a kernel that has one in the
    /// place of the model's: the flow solver's level, colour and grid.
    pub config: Option<Vec<u32>>,
    pub groups: Groups,
}
impl Plan {
    /// A kernel that runs `entry` of `program` over the buffers the entry
    /// point uses (`uses`).
    pub fn new(program: Program, entry: &'static str, groups: u32) -> Self {
        Self {
            program,
            entry,
            config: None,
            groups: Groups::Fixed(groups),
        }
    }

    /// The kernel's `cfg` is its own, of these words.
    pub fn with_config(mut self, words: &[u32]) -> Self {
        self.config = Some(words.to_vec());
        self
    }
}

pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: u32,
}
impl Kernel {
    /// Records this dispatch in a compute pass.
    pub fn record(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
    }
}

/// A global of a program as its text declares it: its binding, and whether
/// it is a uniform (the others are storage buffers).
#[derive(Clone, Copy)]
pub struct Declared {
    pub binding: u32,
    pub uniform: bool,
}

/// The buffers a program declares, by name: every
/// `@group(0) @binding(N) var<..> name:` of its text.
pub fn declarations(source: &str) -> HashMap<String, Declared> {
    let mut declared = HashMap::new();
    let mut rest = source;
    while let Some(at) = rest.find("@binding(") {
        rest = &rest[at + "@binding(".len()..];
        let Some(close) = rest.find(')') else { break };
        let Ok(binding) = rest[..close].trim().parse() else {
            continue;
        };
        let Some(var) = rest.find("var<") else { break };
        let Some(end) = rest[var..].find('>') else {
            break;
        };
        let class = &rest[var + "var<".len()..var + end];
        let name: String = rest[var + end + 1..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        // Only buffers are bound by name: a texture or a sampler has no class.
        if !name.is_empty() && !rest[..var].contains(';') {
            let uniform = class.trim_start().starts_with("uniform");
            declared.insert(name, Declared { binding, uniform });
        }
    }
    declared
}

/// The programs' shader modules with what they declare, and the pipelines
/// made from them so far by entry point.
pub struct Pipelines {
    /// A program's text is parsed and checked once, here: doing that again
    /// for each kernel is what a slow start is made of. A pipeline takes from
    /// its module only what its entry point reaches (wgpu sees to that).
    programs: HashMap<Program, (wgpu::ShaderModule, HashMap<String, Declared>)>,
    made: HashMap<(Program, &'static str), (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}
impl Pipelines {
    pub fn new(device: &wgpu::Device, grains: &Grains) -> Self {
        Self {
            programs: Program::ALL
                .into_iter()
                .map(|program| {
                    let source = program.source(grains);
                    let declared = declarations(&source);
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("{program:?}")),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                    (program, (module, declared))
                })
                .collect(),
            made: HashMap::new(),
        }
    }

    /// The kernel of `plan` over the model's `buffers`; its pipeline is
    /// compiled if this is the first kernel of its entry point.
    pub fn kernel(&mut self, device: &wgpu::Device, buffers: &Buffers, plan: Plan) -> Kernel {
        let (module, declared) = &self.programs[&plan.program];
        let fields = uses(plan.program, plan.entry);
        let places: Vec<Declared> = fields
            .iter()
            .map(|name| {
                *declared.get(*name).unwrap_or_else(|| {
                    panic!(
                        "{:?} declares no buffer {name} ({})",
                        plan.program, plan.entry
                    )
                })
            })
            .collect();
        let (pipeline, layout) = self
            .made
            .entry((plan.program, plan.entry))
            .or_insert_with(|| compile(device, module, plan.entry, &places));
        // The kernel's own configuration, where it has one.
        let own = plan.config.as_ref().map(|words| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(plan.entry),
                contents: &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });
        let entries: Vec<wgpu::BindGroupEntry<'_>> = fields
            .iter()
            .zip(&places)
            .map(|(name, place)| wgpu::BindGroupEntry {
                binding: place.binding,
                resource: match &own {
                    Some(own) if place.uniform => own.as_entire_binding(),
                    _ => buffers.get(name).as_entire_binding(),
                },
            })
            .collect();
        Kernel {
            bindings: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(plan.entry),
                layout,
                entries: &entries,
            }),
            pipeline: pipeline.clone(),
            groups: match plan.groups {
                Groups::Fixed(groups) => groups,
            },
        }
    }
}

fn compile(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    places: &[Declared],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(entry),
        entries: &places
            .iter()
            .map(|place| wgpu::BindGroupLayoutEntry {
                binding: place.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if place.uniform {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            }),
        ),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}
