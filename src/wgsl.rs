//! WGSL validation helper for unit tests.

/// Panics unless `source` is valid WGSL according to naga.
#[cfg(test)]
pub fn assert_valid(name: &str, source: &str) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
    Validator::new(ValidationFlags::all(), Capabilities::default())
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
}
