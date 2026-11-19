//! The sand picture's window.
//! Keys: V cycles the views, H hides or shows the panel, P moves the panel
//! between the window's top and bottom.
#[path = "sand_picture/panel.rs"]
mod panel;
#[path = "sand_picture/viewer.rs"]
mod viewer;

use mikage::{PresentationPolicy, RedrawPolicy, RunConfig, SimulationPolicy};
use viewer::Viewer;

/// The window's first size in logical pixels, the tank's 3:2. The picture is
/// drawn at whatever physical size the display makes of it.
const WINDOW: [u32; 2] = [720, 480];

fn main() {
    let config = RunConfig::<()> {
        title: "Sand Picture".into(),
        width: WINDOW[0],
        height: WINDOW[1],
        redraw_policy: RedrawPolicy::Reactive,
        presentation_policy: PresentationPolicy::Vsync,
        simulation_policy: SimulationPolicy::Manual,
        ..RunConfig::with_defaults(())
    };
    #[cfg(not(target_family = "wasm"))]
    if let Some(agent) = agent() {
        mikage::run_with_agent(|_gpu, _target, _size| Viewer::default(), config, agent)
            .expect("the window could not run");
        return;
    }
    mikage::run(|_gpu, _target, _size| Viewer::default(), config)
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
