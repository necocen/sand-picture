//! One dispatch of the update: a compute pipeline, its bindings (binding 0 is
//! a uniform configuration, the others storage buffers) and its workgroup
//! count. A program is one shader module and a pipeline is made once for an
//! entry point; the kernels that run it differ in what they bind.
use super::material::Grains;
use super::shader::Program;
use std::collections::HashMap;

/// A fixed workgroup count, or one that an earlier kernel of the same update
/// writes into an indirect-argument buffer.
pub enum Groups {
    Fixed(u32),
    Indirect(wgpu::Buffer),
}

/// What a kernel is made from.
pub struct Plan {
    pub program: Program,
    pub entry: &'static str,
    pub bindings: Vec<(u32, wgpu::Buffer)>,
    pub groups: Groups,
}
impl Plan {
    pub fn new(
        program: Program,
        entry: &'static str,
        bindings: &[(u32, &wgpu::Buffer)],
        groups: u32,
    ) -> Self {
        Self {
            program,
            entry,
            bindings: bindings.iter().map(|(i, b)| (*i, (*b).clone())).collect(),
            groups: Groups::Fixed(groups),
        }
    }

    /// Dispatch the workgroup counts written in `arguments` instead.
    pub fn indirect(mut self, arguments: &wgpu::Buffer) -> Self {
        self.groups = Groups::Indirect(arguments.clone());
        self
    }
}

pub struct Kernel {
    entry: &'static str,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: Groups,
}
impl Kernel {
    /// Records this dispatch in a compute pass.
    pub fn record(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        match &self.groups {
            Groups::Fixed(groups) => pass.dispatch_workgroups(*groups, 1, 1),
            Groups::Indirect(arguments) => pass.dispatch_workgroups_indirect(arguments, 0),
        }
    }

    /// The shader entry point, for profiles.
    pub fn entry(&self) -> &'static str {
        self.entry
    }
}

/// The programs' shader modules, and the pipelines made from them so far by
/// entry point.
pub struct Pipelines {
    /// A program's text is parsed and checked once, here: doing that again
    /// for each kernel is what a slow start is made of. A pipeline takes from
    /// its module only what its entry point reaches (wgpu sees to that).
    modules: HashMap<Program, wgpu::ShaderModule>,
    made: HashMap<(Program, &'static str), (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}
impl Pipelines {
    pub fn new(device: &wgpu::Device, grains: &Grains) -> Self {
        Self {
            modules: Program::ALL
                .into_iter()
                .map(|program| {
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("{program:?}")),
                        source: wgpu::ShaderSource::Wgsl(program.source(grains).into()),
                    });
                    (program, module)
                })
                .collect(),
            made: HashMap::new(),
        }
    }

    /// The kernel of `plan`; its pipeline is compiled if this is the first
    /// kernel of its entry point.
    pub fn kernel(&mut self, device: &wgpu::Device, plan: Plan) -> Kernel {
        let module = &self.modules[&plan.program];
        let (pipeline, layout) = self
            .made
            .entry((plan.program, plan.entry))
            .or_insert_with(|| compile(device, module, &plan));
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(plan.entry),
            layout,
            entries: &plan
                .bindings
                .iter()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: *i,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Kernel {
            entry: plan.entry,
            pipeline: pipeline.clone(),
            bindings,
            groups: plan.groups,
        }
    }
}

fn compile(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    plan: &Plan,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(plan.entry),
        entries: &plan
            .bindings
            .iter()
            .map(|(i, _)| wgpu::BindGroupLayoutEntry {
                binding: *i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if *i == 0 {
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
        label: Some(plan.entry),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(plan.entry),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            }),
        ),
        module,
        entry_point: Some(plan.entry),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}
