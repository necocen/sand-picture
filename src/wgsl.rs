//! WGSL as it is handed to the compiler: the text of a program, with its
//! loops of constant bounds written out where that pays (`unroll`).
mod unroll;

/// Whether the loops of constant bounds are written out. wgpu hands Metal a
/// loop in a form its compiler does not unroll, and the regression's baseline
/// holds the results of the written-out loops (writing a loop out can change
/// the last bits), so Apple's native targets write them out. A browser
/// compiles the WGSL itself, and no other backend was measured: there the
/// loops stay as written.
const WRITE_OUT_LOOPS: bool = cfg!(target_vendor = "apple");

/// `source` as it is compiled on this target.
pub fn compiled(source: String) -> String {
    if WRITE_OUT_LOOPS {
        written_out(&source)
    } else {
        source
    }
}

/// `source` without comments and with its constant loops written out.
pub fn written_out(source: &str) -> String {
    unroll::unroll(source)
}

/// Panics unless `source` is valid WGSL both as written and with its loops
/// written out: a browser gets the one form, Metal the other.
#[cfg(test)]
pub fn assert_valid(name: &str, source: &str) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    for text in [written_out(source), source.to_owned()] {
        let module = naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&text)));
        Validator::new(ValidationFlags::all(), Capabilities::default())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}
