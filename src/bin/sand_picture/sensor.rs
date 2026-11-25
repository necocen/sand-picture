//! Device motion / gravity sensor bridge.
//!
//! On browsers that expose `DeviceMotionEvent` in a secure context, subscribes
//! to `devicemotion` (`accelerationIncludingGravity`), normalizes iOS WebKit's
//! inverted axis signs to W3C proper acceleration, rotates into the current
//! screen orientation, and delivers smoothed screen-frame apparent gravity
//! `[gx (+right), gy (+down), gz (out of glass)]` in m/s^2.

/// Lifecycle state of the device gravity sensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
pub enum SensorState {
    /// No motion sensor API is available on this platform or context.
    Unavailable,
    /// Motion sensor API exists and is currently switched off.
    Off,
    /// Permission granted or not required; waiting for first non-null sample.
    Starting,
    /// iOS Safari requires a direct DOM tap to open its permission dialog.
    NeedsTap,
    /// Actively delivering live gravity / acceleration samples.
    Active,
    /// Permission was denied by the user or browser.
    Denied,
}

impl SensorState {
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Starting | Self::NeedsTap | Self::Active)
    }
}

/// Converts W3C-normalized proper acceleration `(ax, ay, az)` in device
/// portrait coordinates (+x right, +y up, +z out of screen) and clockwise
/// screen orientation `angle_deg` (`0`, `90`, `180`, `270`) into screen-frame
/// apparent gravity `[gx (+right), gy (+down), gz (out of glass)]` (m/s^2).
#[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
pub fn proper_to_apparent_gravity(proper: [f32; 3], angle_deg: i32) -> [f32; 3] {
    let [x, y, z] = proper;
    let (sx, sy) = match angle_deg.rem_euclid(360) {
        90 => (-y, x),
        180 => (-x, -y),
        270 => (y, -x),
        _ => (x, y),
    };
    [-sx, sy, z.abs()]
}

#[cfg(not(target_family = "wasm"))]
pub struct Sensor;

#[cfg(not(target_family = "wasm"))]
impl Sensor {
    pub fn new() -> Self {
        Self
    }

    pub fn state(&self) -> SensorState {
        SensorState::Unavailable
    }

    pub fn toggle(&mut self) {}

    pub fn gravity(&self) -> Option<[f32; 3]> {
        None
    }
}

#[cfg(target_family = "wasm")]
pub struct Sensor {
    shared: std::rc::Rc<std::cell::RefCell<WasmSensor>>,
}

#[cfg(target_family = "wasm")]
type MotionCallback = wasm_bindgen::closure::Closure<dyn FnMut(web_sys::DeviceMotionEvent)>;
#[cfg(target_family = "wasm")]
type PromiseCallback = wasm_bindgen::closure::Closure<dyn FnMut(wasm_bindgen::JsValue)>;

#[cfg(target_family = "wasm")]
struct WasmSensor {
    state: SensorState,
    smoothed: Option<[f32; 3]>,
    ios_sign: f32,
    listener: Option<MotionCallback>,
    prompt_click: Option<wasm_bindgen::closure::Closure<dyn FnMut()>>,
    permission_cbs: Option<(PromiseCallback, PromiseCallback)>,
}

#[cfg(target_family = "wasm")]
const PROMPT_BUTTON_ID: &str = "sand-picture-sensor-permission";

#[cfg(target_family = "wasm")]
impl Sensor {
    pub fn new() -> Self {
        let available = web_sys::window().is_some_and(|win| {
            win.is_secure_context()
                && js_sys::Reflect::has(&win, &wasm_bindgen::JsValue::from_str("DeviceMotionEvent"))
                    .unwrap_or(false)
        });
        let ios_sign = if detect_ios() { -1.0 } else { 1.0 };
        Self {
            shared: std::rc::Rc::new(std::cell::RefCell::new(WasmSensor {
                state: if available {
                    SensorState::Off
                } else {
                    SensorState::Unavailable
                },
                smoothed: None,
                ios_sign,
                listener: None,
                prompt_click: None,
                permission_cbs: None,
            })),
        }
    }

    pub fn state(&self) -> SensorState {
        self.shared.borrow().state
    }

    pub fn gravity(&self) -> Option<[f32; 3]> {
        let inner = self.shared.borrow();
        (inner.state == SensorState::Active)
            .then_some(inner.smoothed)
            .flatten()
    }

    pub fn toggle(&mut self) {
        let current = self.shared.borrow().state;
        match current {
            SensorState::Unavailable => {}
            SensorState::Starting | SensorState::NeedsTap | SensorState::Active => {
                stop_listening(&self.shared);
            }
            SensorState::Off | SensorState::Denied => {
                start_or_request(&self.shared);
            }
        }
    }
}

#[cfg(target_family = "wasm")]
impl Drop for Sensor {
    fn drop(&mut self) {
        stop_listening(&self.shared);
    }
}

#[cfg(target_family = "wasm")]
fn detect_ios() -> bool {
    let Some(win) = web_sys::window() else {
        return false;
    };
    if has_ios_request_permission(&win) {
        return true;
    }
    let nav = win.navigator();
    let ua = nav.user_agent().unwrap_or_default();
    if ua.contains("iPhone") || ua.contains("iPad") || ua.contains("iPod") {
        return true;
    }
    let platform = nav.platform().unwrap_or_default();
    platform == "MacIntel" && nav.max_touch_points() > 1
}

#[cfg(target_family = "wasm")]
fn has_ios_request_permission(win: &web_sys::Window) -> bool {
    let Ok(ctor) = js_sys::Reflect::get(win, &wasm_bindgen::JsValue::from_str("DeviceMotionEvent"))
    else {
        return false;
    };
    let Ok(req) =
        js_sys::Reflect::get(&ctor, &wasm_bindgen::JsValue::from_str("requestPermission"))
    else {
        return false;
    };
    req.is_function()
}

#[cfg(target_family = "wasm")]
fn screen_angle_deg(win: &web_sys::Window) -> i32 {
    if let Ok(screen) = win.screen() {
        let orientation = screen.orientation();
        if let Ok(angle) = orientation.angle() {
            return i32::from(angle);
        }
    }
    js_sys::Reflect::get(win, &wasm_bindgen::JsValue::from_str("orientation"))
        .ok()
        .and_then(|v| v.as_f64())
        .map_or(0, |a| a.round() as i32)
}

#[cfg(target_family = "wasm")]
fn remove_prompt_button(shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document())
        && let Some(el) = doc.get_element_by_id(PROMPT_BUTTON_ID)
    {
        el.remove();
    }
    shared.borrow_mut().prompt_click = None;
}

#[cfg(target_family = "wasm")]
fn stop_listening(shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>) {
    use wasm_bindgen::JsCast;
    remove_prompt_button(shared);
    let old = {
        let mut inner = shared.borrow_mut();
        inner.state = SensorState::Off;
        inner.smoothed = None;
        inner.permission_cbs = None;
        inner.listener.take()
    };
    if let (Some(win), Some(cb)) = (web_sys::window(), old) {
        let _ =
            win.remove_event_listener_with_callback("devicemotion", cb.as_ref().unchecked_ref());
    }
}

#[cfg(target_family = "wasm")]
fn attach_listener(shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>) {
    use wasm_bindgen::JsCast;
    let Some(win) = web_sys::window() else {
        return;
    };
    remove_prompt_button(shared);
    {
        let mut inner = shared.borrow_mut();
        if inner.listener.is_some() {
            inner.state = if inner.smoothed.is_some() {
                SensorState::Active
            } else {
                SensorState::Starting
            };
            return;
        }
        inner.state = SensorState::Starting;
        inner.smoothed = None;
    }

    let weak = std::rc::Rc::downgrade(shared);
    let closure = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::DeviceMotionEvent)>::new(
        move |event: web_sys::DeviceMotionEvent| {
            let Some(cell) = weak.upgrade() else {
                return;
            };
            let Some(acc) = event.acceleration_including_gravity() else {
                return;
            };
            let (Some(x), Some(y), Some(z)) = (acc.x(), acc.y(), acc.z()) else {
                return;
            };
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                return;
            }
            let mag2 = x * x + y * y + z * z;
            if mag2 < 0.04 {
                return;
            }
            let angle = web_sys::window().map_or(0, |w| screen_angle_deg(&w));
            let mut inner = cell.borrow_mut();
            let s = inner.ios_sign;
            let raw = proper_to_apparent_gravity([x as f32 * s, y as f32 * s, z as f32 * s], angle);
            let next = match inner.smoothed {
                Some(prev) => {
                    const ALPHA: f32 = 0.45;
                    [
                        prev[0] + ALPHA * (raw[0] - prev[0]),
                        prev[1] + ALPHA * (raw[1] - prev[1]),
                        prev[2] + ALPHA * (raw[2] - prev[2]),
                    ]
                }
                None => raw,
            };
            inner.smoothed = Some(next);
            inner.state = SensorState::Active;
        },
    );
    if win
        .add_event_listener_with_callback("devicemotion", closure.as_ref().unchecked_ref())
        .is_ok()
    {
        shared.borrow_mut().listener = Some(closure);
    } else {
        shared.borrow_mut().state = SensorState::Denied;
    }
}

#[cfg(target_family = "wasm")]
fn request_ios_permission(
    shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>,
    from_dom_tap: bool,
) {
    use wasm_bindgen::JsCast;
    let Some(win) = web_sys::window() else {
        return;
    };
    let Ok(ctor) =
        js_sys::Reflect::get(&win, &wasm_bindgen::JsValue::from_str("DeviceMotionEvent"))
    else {
        attach_listener(shared);
        return;
    };
    let Ok(req) =
        js_sys::Reflect::get(&ctor, &wasm_bindgen::JsValue::from_str("requestPermission"))
    else {
        attach_listener(shared);
        return;
    };
    let Ok(func) = req.dyn_into::<js_sys::Function>() else {
        attach_listener(shared);
        return;
    };
    let Ok(promise_val) = func.call0(&ctor) else {
        if !from_dom_tap {
            show_prompt_button(shared);
        } else {
            shared.borrow_mut().state = SensorState::Denied;
        }
        return;
    };
    let Ok(promise) = promise_val.dyn_into::<js_sys::Promise>() else {
        attach_listener(shared);
        return;
    };
    shared.borrow_mut().state = SensorState::Starting;

    let ok_shared = std::rc::Rc::clone(shared);
    let on_ok: PromiseCallback =
        wasm_bindgen::closure::Closure::once(move |val: wasm_bindgen::JsValue| {
            if val.as_string().as_deref() == Some("granted") {
                attach_listener(&ok_shared);
            } else {
                remove_prompt_button(&ok_shared);
                ok_shared.borrow_mut().state = SensorState::Denied;
            }
        });
    let err_shared = std::rc::Rc::clone(shared);
    let on_err: PromiseCallback =
        wasm_bindgen::closure::Closure::once(move |_err: wasm_bindgen::JsValue| {
            if !from_dom_tap {
                show_prompt_button(&err_shared);
            } else {
                remove_prompt_button(&err_shared);
                err_shared.borrow_mut().state = SensorState::Denied;
            }
        });
    let _ = promise.then2(&on_ok, &on_err);
    shared.borrow_mut().permission_cbs = Some((on_ok, on_err));
}

#[cfg(target_family = "wasm")]
fn show_prompt_button(shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>) {
    use wasm_bindgen::JsCast;
    remove_prompt_button(shared);
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        shared.borrow_mut().state = SensorState::Denied;
        return;
    };
    let Some(body) = doc.body() else {
        shared.borrow_mut().state = SensorState::Denied;
        return;
    };
    let Ok(el) = doc.create_element("button") else {
        shared.borrow_mut().state = SensorState::Denied;
        return;
    };
    let Ok(btn) = el.dyn_into::<web_sys::HtmlElement>() else {
        shared.borrow_mut().state = SensorState::Denied;
        return;
    };
    btn.set_id(PROMPT_BUTTON_ID);
    btn.set_text_content(Some("Tap to allow tilt sensor"));
    btn.style().set_css_text(
        "position:fixed;left:50%;bottom:24px;transform:translateX(-50%);\
         z-index:9999;padding:12px 20px;border-radius:999px;border:1px solid rgba(255,255,255,0.28);\
         background:rgba(16,22,34,0.92);color:#e6eefc;font:600 15px -apple-system,sans-serif;\
         box-shadow:0 6px 24px rgba(0,0,0,0.45);cursor:pointer;",
    );
    let shared_click = std::rc::Rc::clone(shared);
    let on_click = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(move || {
        if let Some(doc) = web_sys::window().and_then(|w| w.document())
            && let Some(el) = doc.get_element_by_id(PROMPT_BUTTON_ID)
        {
            el.remove();
        }
        request_ios_permission(&shared_click, true);
    });
    let _ = btn.add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref());
    if body.append_child(&btn).is_ok() {
        let mut inner = shared.borrow_mut();
        inner.prompt_click = Some(on_click);
        inner.state = SensorState::NeedsTap;
    } else {
        shared.borrow_mut().state = SensorState::Denied;
    }
}

#[cfg(target_family = "wasm")]
fn start_or_request(shared: &std::rc::Rc<std::cell::RefCell<WasmSensor>>) {
    let needs_permission = web_sys::window().is_some_and(|w| has_ios_request_permission(&w));
    if needs_permission {
        request_ios_permission(shared, false);
    } else {
        attach_listener(shared);
    }
}

#[cfg(test)]
mod tests {
    use super::proper_to_apparent_gravity;

    #[test]
    fn screen_orientation_maps_proper_acceleration_to_apparent_gravity() {
        let g = 9.81_f32;
        // Portrait upright: proper (0, +g, 0) -> apparent [0, +g (down), 0].
        let v = proper_to_apparent_gravity([0.0, g, 0.0], 0);
        assert!(v[0].abs() < 1e-5 && (v[1] - g).abs() < 1e-5 && v[2].abs() < 1e-5);

        // Portrait tilted 30 deg CW (right edge down): proper (-g sin 30, +g cos 30, 0)
        // -> apparent [+g sin 30 (right), +g cos 30 (down), 0].
        let rad = 30.0_f32.to_radians();
        let v = proper_to_apparent_gravity([-g * rad.sin(), g * rad.cos(), 0.0], 0);
        assert!((v[0] - g * rad.sin()).abs() < 1e-4 && (v[1] - g * rad.cos()).abs() < 1e-4);

        // Landscape-primary (90 deg CCW, top on left) upright: proper (+g, 0, 0) -> down.
        let v = proper_to_apparent_gravity([g, 0.0, 0.0], 90);
        assert!(v[0].abs() < 1e-5 && (v[1] - g).abs() < 1e-5);

        // Landscape-secondary (270 / -90 deg) upright: proper (-g, 0, 0) -> down.
        let v = proper_to_apparent_gravity([-g, 0.0, 0.0], -90);
        assert!(v[0].abs() < 1e-5 && (v[1] - g).abs() < 1e-5);

        // Lying flat screen-up: proper (0, 0, +g) -> plate gravity +g.
        let v = proper_to_apparent_gravity([0.0, 0.0, g], 0);
        assert!(v[0].abs() < 1e-5 && v[1].abs() < 1e-5 && (v[2] - g).abs() < 1e-5);
    }
}
