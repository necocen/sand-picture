//! A shader cut down to what one entry point reaches.
//!
//! wgpu's Metal backend writes every function of a module to Metal's language
//! and compiles the whole of it for each pipeline, and one entry point of the
//! model reaches a few hundredths of the model's functions. So each kernel
//! gets a module of its own: the declarations its entry point reaches, in
//! their order and in their own text. A declaration is never rewritten: the
//! Metal compiler works with fast math, and another form of the same
//! expression may round differently.

use std::collections::{HashMap, HashSet};

/// A top-level declaration of comment-free WGSL: its text and, unless it is a
/// directive, the name it declares.
struct Declaration<'a> {
    name: Option<&'a str>,
    text: &'a str,
}

/// The source's declarations that `entry` reaches, with every directive.
/// `source` holds no comments (the loop writer has removed them).
pub fn reach(source: &str, entry: &str) -> String {
    let declarations = declarations(source);
    let index: HashMap<&str, usize> = declarations
        .iter()
        .enumerate()
        .filter_map(|(i, d)| d.name.map(|name| (name, i)))
        .collect();
    let start = *index
        .get(entry)
        .unwrap_or_else(|| panic!("the shader has no entry point {entry}"));
    let mut kept = HashSet::from([start]);
    let mut pending = vec![start];
    while let Some(i) = pending.pop() {
        for word in identifiers(declarations[i].text) {
            if let Some(&j) = index.get(word)
                && kept.insert(j)
            {
                pending.push(j);
            }
        }
    }
    declarations
        .iter()
        .enumerate()
        .filter(|(i, d)| d.name.is_none() || kept.contains(i))
        .map(|(_, d)| d.text)
        .collect::<Vec<_>>()
        .join("\n")
}

fn identifier(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The identifiers of a text, in order.
fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut at = 0;
    std::iter::from_fn(move || {
        loop {
            match bytes.get(at) {
                None => return None,
                // A number's digits and suffix are no identifier.
                Some(c) if c.is_ascii_digit() => {
                    while at < bytes.len() && (identifier(bytes[at]) || bytes[at] == b'.') {
                        at += 1;
                    }
                }
                Some(&c) if identifier(c) => break,
                Some(_) => at += 1,
            }
        }
        let start = at;
        while at < bytes.len() && identifier(bytes[at]) {
            at += 1;
        }
        Some(&text[start..at])
    })
}

/// The position after the bracket that closes the one at `open`.
fn closing(bytes: &[u8], open: usize) -> usize {
    let mut depth = 0_usize;
    for (i, &c) in bytes.iter().enumerate().skip(open) {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("an unclosed bracket in the shader");
}

/// The source's top-level declarations, in order.
fn declarations(source: &str) -> Vec<Declaration<'_>> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut at = 0;
    loop {
        while at < bytes.len() && (bytes[at].is_ascii_whitespace() || bytes[at] == b';') {
            at += 1;
        }
        if at >= bytes.len() {
            return found;
        }
        let start = at;
        // Attributes before the keyword: `@name` or `@name(...)`.
        while bytes[at] == b'@' {
            at += 1;
            while at < bytes.len() && identifier(bytes[at]) {
                at += 1;
            }
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
            if at < bytes.len() && bytes[at] == b'(' {
                at = closing(bytes, at);
            }
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
        }
        let word_start = at;
        while at < bytes.len() && identifier(bytes[at]) {
            at += 1;
        }
        let keyword = &source[word_start..at];
        let named = matches!(
            keyword,
            "fn" | "struct" | "const" | "var" | "alias" | "override"
        );
        let name = named.then(|| {
            let mut cursor = at;
            while bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            // `var<storage,read_write> name`
            if bytes[cursor] == b'<' {
                while bytes[cursor] != b'>' {
                    cursor += 1;
                }
                cursor += 1;
                while bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
            }
            let from = cursor;
            while cursor < bytes.len() && identifier(bytes[cursor]) {
                cursor += 1;
            }
            &source[from..cursor]
        });
        // A function or a structure ends with its braces, anything else with
        // the next semicolon outside brackets.
        let end = if matches!(keyword, "fn" | "struct") {
            let open = at
                + source[at..]
                    .find('{')
                    .expect("a declaration without a body");
            closing(bytes, open)
        } else {
            let mut depth = 0_usize;
            let mut cursor = at;
            loop {
                match bytes.get(cursor) {
                    Some(b'(' | b'[' | b'{') => depth += 1,
                    Some(b')' | b']' | b'}') => depth -= 1,
                    Some(b';') if depth == 0 => break cursor + 1,
                    Some(_) => {}
                    None => panic!("a declaration without an end in the shader"),
                }
                cursor += 1;
            }
        };
        found.push(Declaration {
            name,
            text: &source[start..end],
        });
        at = end;
    }
}

#[cfg(test)]
mod tests {
    use super::reach;

    const SOURCE: &str = "enable f16;
struct Cell {a:f32,b:vec2<f32>}
struct Unused {a:f32};
const SIDE:u32=4u;
const SPARE:f32=1.5;
@group(0) @binding(1) var<storage,read_write> cells:array<Cell>;
@group(0) @binding(2) var<storage,read_write> other:array<f32>;
alias Pair=vec2<f32>;
fn weight(c:Cell)->f32 {return c.a*f32(SIDE);}
fn idle(x:f32)->f32 {return x*SPARE;}
fn pair(c:Cell)->Pair {return c.b;}
@compute @workgroup_size(64) fn sweep(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=cells[id.x];cells[id.x].a=weight(c)+pair(c).x;
}
@compute @workgroup_size(64)
fn rest(@builtin(global_invocation_id) id:vec3<u32>) {other[id.x]=idle(2.);}";

    #[test]
    fn an_entry_keeps_what_it_reaches_in_its_own_text() {
        let kept = reach(SOURCE, "sweep");
        for text in [
            "enable f16;",
            "struct Cell {a:f32,b:vec2<f32>}",
            "const SIDE:u32=4u;",
            "@group(0) @binding(1) var<storage,read_write> cells:array<Cell>;",
            "alias Pair=vec2<f32>;",
            "fn weight(c:Cell)->f32 {return c.a*f32(SIDE);}",
            "fn pair(c:Cell)->Pair {return c.b;}",
            "let c=cells[id.x];cells[id.x].a=weight(c)+pair(c).x;",
        ] {
            assert!(kept.contains(text), "missing {text:?} in\n{kept}");
        }
        for text in ["Unused", "SPARE", "other", "idle", "rest"] {
            assert!(!kept.contains(text), "kept {text:?} in\n{kept}");
        }
    }

    #[test]
    fn the_kept_declarations_stay_in_order() {
        let kept = reach(SOURCE, "rest");
        let places: Vec<usize> = ["enable", "SPARE", "other:", "fn idle", "fn rest"]
            .iter()
            .map(|text| kept.find(text).unwrap_or_else(|| panic!("missing {text}")))
            .collect();
        assert!(places.windows(2).all(|pair| pair[0] < pair[1]), "{kept}");
        assert!(!kept.contains("cells"), "{kept}");
    }

    #[test]
    #[should_panic(expected = "no entry point")]
    fn an_unknown_entry_is_refused() {
        reach(SOURCE, "missing");
    }
}
