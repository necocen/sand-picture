//! The sand picture's window.
//! Keys: F turns the picture over, Space stops and runs its time, R resets
//! it, V cycles the views, H hides or shows the panel, P moves the panel
//! between the window's top and bottom.
#[path = "sand_picture/app.rs"]
mod app;
#[path = "sand_picture/panel.rs"]
mod panel;
#[path = "sand_picture/product.rs"]
mod product;
#[path = "sand_picture/screen.rs"]
mod screen;

use app::SandPicture;
use mikage::{
    GpuDescriptor, GpuRequirements, PresentationPolicy, RedrawPolicy, RunConfig, SimulationPolicy,
};
use sand_picture::model::DT;
use std::time::Duration;

/// The window's first size in logical pixels, the tank's 3:2. The picture is
/// drawn at whatever physical size the display makes of it.
const WINDOW: [u32; 2] = [720, 480];

fn main() {
    let config = RunConfig::<()> {
        title: "Sand Picture".into(),
        width: WINDOW[0],
        height: WINDOW[1],
        // WebGPU's default limits and nothing above them: the same model
        // runs in a browser (eight storage buffers a stage, 256 threads a
        // workgroup).
        gpu: GpuDescriptor {
            requirements: GpuRequirements {
                required_limits: Some(wgpu::Limits::default()),
                ..Default::default()
            },
            ..Default::default()
        },
        // A frame is drawn when something changed, at the display's pace.
        redraw_policy: RedrawPolicy::Reactive,
        presentation_policy: PresentationPolicy::Vsync,
        // One update of DT for every DT of the clock, as far as the GPU
        // keeps up; time it cannot keep up with is dropped, not owed.
        simulation_policy: SimulationPolicy::Fixed {
            dt: Duration::from_secs_f64(f64::from(DT)),
            max_ticks_per_wake: 1,
            max_backlog: Duration::from_millis(250),
        },
        // One update in flight beside one frame: a frame then waits for one
        // update at most, and shows the newest.
        max_in_flight_submissions: 2,
        ..RunConfig::with_defaults(())
    };
    #[cfg(not(target_family = "wasm"))]
    if let Some(agent) = agent() {
        mikage::run_with_agent(
            |gpu, target, _size| SandPicture::new(gpu, target),
            config,
            agent,
        )
        .expect("the window could not run");
        return;
    }
    mikage::run(|gpu, target, _size| SandPicture::new(gpu, target), config)
        .expect("the window could not run");
}

/// The HTTP port through which tools read the window's picture and state:
/// `SP_AGENT=1` binds mikage's default address, any other value is the address.
#[cfg(not(target_family = "wasm"))]
fn agent() -> Option<mikage::AgentConfig> {
    let value = std::env::var("SP_AGENT").ok()?;
    let config = mikage::AgentConfig::default();
    if value == "1" {
        return Some(config);
    }
    let address = value
        .parse()
        .expect("SP_AGENT is 1 or an address such as 127.0.0.1:3939");
    Some(config.with_bind_addr(address))
}
