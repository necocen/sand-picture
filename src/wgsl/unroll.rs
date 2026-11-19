//! Loops with constant bounds, written out before a shader is compiled.
//!
//! wgpu writes every WGSL loop to Metal as `while(true)` guarded by a
//! 64-bit bound counter and a first-turn flag, a form in which the Metal
//! compiler does not see a counted loop, and on the GPUs measured each turn
//! then costs more than the few loads and products of a small body (the same
//! 25-term product: 15 us as a nested loop, 7 us written out). So every
//! `for` whose start, end and step are integer constants, and whose body
//! neither breaks out nor writes its counter, becomes one block a turn,
//! `{let k=0u;BODY}{let k=1u;BODY}...`: the same statements in the same
//! order. A body's `if C {.. continue;}` becomes `if C {..} else {the rest}`.
//! The source keeps its loops.
//!
//! A loop marked `/*loop: why*/for(...)` stays a loop. Some are faster so: a
//! long body many times over (a 24-turn bisection, nine identities' fluxes)
//! outgrows what the compiler handles well, and a kernel that ends at once
//! for most of its threads pays for its size.
use std::collections::HashMap;

/// The most turns written out, and the most bytes one loop may become.
const MAX_TURNS: usize = 32;
const MAX_BYTES: usize = 16_384;
/// Stands in the comment-free text where a `/*loop ..*/` mark stood.
const KEPT: char = '\u{1}';

#[derive(Clone, Copy)]
struct Integer {
    value: i64,
    /// Whether the source says so (a `u` suffix, a `u32` constant).
    unsigned: bool,
}

/// `source` without comments (line breaks kept) and with its constant loops
/// written out. Line numbers do not change.
pub fn unroll(source: &str) -> String {
    let source = without_comments(source);
    let constants = constants(&source);
    expand(&source, &constants).replace(KEPT, "")
}

fn without_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(at) = rest.find('/') {
        out.push_str(&rest[..at]);
        if rest[at..].starts_with("//") {
            let end = rest[at..].find('\n').map_or(rest.len(), |n| at + n);
            rest = &rest[end..];
        } else if rest[at..].starts_with("/*") {
            let end = rest[at + 2..].find("*/").map_or(rest.len(), |n| at + n + 4);
            if rest[at + 2..end].trim_start().starts_with("loop") {
                out.push(KEPT);
            }
            out.extend(rest[at..end].chars().filter(|c| *c == '\n'));
            rest = &rest[end..];
        } else {
            out.push('/');
            rest = &rest[at + 1..];
        }
    }
    out.push_str(rest);
    out
}

fn identifier(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The next `word` at or after `from` that stands alone.
fn keyword(text: &str, from: usize, word: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut at = from;
    while let Some(found) = text[at..].find(word).map(|n| at + n) {
        let end = found + word.len();
        if (found == 0 || !identifier(bytes[found - 1]))
            && bytes.get(end).is_none_or(|c| !identifier(*c))
        {
            return Some(found);
        }
        at = end;
    }
    None
}

/// The index after the bracket that closes the one at `open`.
fn closing(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let (left, right) = match bytes[open] {
        b'(' => (b'(', b')'),
        b'{' => (b'{', b'}'),
        _ => return None,
    };
    let mut depth = 0usize;
    for (k, c) in bytes.iter().enumerate().skip(open) {
        if *c == left {
            depth += 1;
        } else if *c == right {
            depth -= 1;
            if depth == 0 {
                return Some(k + 1);
            }
        }
    }
    None
}

fn literal(text: &str) -> Option<Integer> {
    let (digits, unsigned) = match text.strip_suffix('u') {
        Some(digits) => (digits, true),
        None => (text.strip_suffix('i').unwrap_or(text), false),
    };
    let value = digits.parse::<i64>().ok()?;
    (digits.bytes().all(|c| c.is_ascii_digit() || c == b'-')).then_some(Integer { value, unsigned })
}

/// The module's integer constants, `const NAME[:u32|:i32]=LITERAL;`.
fn constants(source: &str) -> HashMap<String, Integer> {
    let mut found = HashMap::new();
    let mut at = 0;
    while let Some(start) = keyword(source, at, "const") {
        at = start + 5;
        let Some(end) = source[at..].find(';').map(|n| at + n) else {
            break;
        };
        let Some((name, value)) = source[at..end].split_once('=') else {
            continue;
        };
        let (name, kind) = match name.split_once(':') {
            Some((name, kind)) => (name.trim(), Some(kind.trim())),
            None => (name.trim(), None),
        };
        if let Some(mut integer) = literal(value.trim())
            && matches!(kind, None | Some("u32" | "i32"))
            && !name.is_empty()
            && name.bytes().all(identifier)
        {
            integer.unsigned |= kind == Some("u32");
            found.insert(name.to_owned(), integer);
        }
    }
    found
}

struct Loop<'a> {
    name: &'a str,
    /// `:u32` or `:i32` as the counter declares it, or nothing.
    kind: &'a str,
    unsigned: bool,
    /// The counter's values, turn by turn; `None` when the header is not constant.
    turns: Option<Vec<i64>>,
    /// The text between the body's braces, and the index after its closing one.
    body: &'a str,
    body_start: usize,
    end: usize,
}

fn value(text: &str, constants: &HashMap<String, Integer>) -> Option<Integer> {
    let text = text.trim();
    literal(text).or_else(|| constants.get(text).copied())
}

/// The `for` at `start`: `for(var NAME[:TYPE]=A;NAME<B;NAME++){BODY}` and its
/// variants with `<=`, `>`, `>=`, `--`, `+=C`, `-=C`, `*=C` and `/=C`.
fn parse<'a>(
    text: &'a str,
    start: usize,
    constants: &HashMap<String, Integer>,
) -> Option<Loop<'a>> {
    let open = start + 3 + text[start + 3..].find(|c: char| !c.is_whitespace())?;
    let header_end = closing(text, open)?;
    let brace = header_end + text[header_end..].find(|c: char| !c.is_whitespace())?;
    let end = closing(text, brace)?;
    let mut loop_ = Loop {
        name: "",
        kind: "",
        unsigned: false,
        turns: None,
        body: &text[brace + 1..end - 1],
        body_start: brace + 1,
        end,
    };
    let header: Vec<&str> = text[open + 1..header_end - 1].split(';').collect();
    let [init, condition, step] = header[..] else {
        return Some(loop_);
    };
    let Some((declared, first)) = init
        .trim()
        .strip_prefix("var ")
        .and_then(|s| s.split_once('='))
    else {
        return Some(loop_);
    };
    let (name, kind) = match declared.find(':') {
        Some(colon) => (declared[..colon].trim(), declared[colon..].trim()),
        None => (declared.trim(), ""),
    };
    if name.is_empty() || !name.bytes().all(identifier) || !matches!(kind, "" | ":u32" | ":i32") {
        return Some(loop_);
    }
    loop_.name = name;
    loop_.kind = kind;
    let turns = || -> Option<(Vec<i64>, bool)> {
        let first = value(first, constants)?;
        let condition = condition.trim().strip_prefix(name)?.trim_start();
        let (compare, bound) = ["<=", ">=", "<", ">"]
            .iter()
            .find_map(|op| Some((*op, condition.strip_prefix(op)?)))?;
        let bound = value(bound, constants)?;
        let step = step.trim().strip_prefix(name)?.trim();
        let by = |text: &str| Some(value(text, constants)?.value);
        let next: Box<dyn Fn(i64) -> Option<i64>> = match step {
            "++" => Box::new(|k| Some(k + 1)),
            "--" => Box::new(|k| Some(k - 1)),
            _ => {
                let (op, operand) = step.split_at_checked(2)?;
                let operand = by(operand)?;
                match op {
                    "+=" if operand != 0 => Box::new(move |k| Some(k + operand)),
                    "-=" if operand != 0 => Box::new(move |k| Some(k - operand)),
                    "*=" if operand > 1 => Box::new(move |k| (k != 0).then_some(k * operand)),
                    "/=" if operand > 1 => Box::new(move |k| Some(k / operand)),
                    _ => return None,
                }
            }
        };
        let holds = |k: i64| match compare {
            "<=" => k <= bound.value,
            ">=" => k >= bound.value,
            "<" => k < bound.value,
            _ => k > bound.value,
        };
        let mut values = Vec::new();
        let mut k = first.value;
        while holds(k) {
            if values.len() == MAX_TURNS {
                return None;
            }
            values.push(k);
            k = next(k)?;
        }
        Some((values, first.unsigned || bound.unsigned || kind == ":u32"))
    };
    if let Some((values, unsigned)) = turns() {
        loop_.unsigned = unsigned && kind != ":i32";
        loop_.turns = Some(values);
    }
    Some(loop_)
}

/// Whether `word` (`break` or `continue`) stands in `text` outside the loops
/// that `text` still holds, whose own it would be.
fn leaves_loop(text: &str, word: &str) -> bool {
    let mut own = String::with_capacity(text.len());
    let mut at = 0;
    loop {
        let next = ["for", "while", "loop"]
            .iter()
            .filter_map(|word| keyword(text, at, word))
            .min();
        let Some(start) = next else {
            own.push_str(&text[at..]);
            break;
        };
        own.push_str(&text[at..start]);
        let Some(end) = text[start..]
            .find('{')
            .and_then(|brace| closing(text, start + brace))
        else {
            return true;
        };
        at = end;
    }
    keyword(&own, 0, word).is_some()
}

/// Whether `body` may write the counter `name`.
fn writes(body: &str, name: &str) -> bool {
    let bytes = body.as_bytes();
    let mut at = 0;
    while let Some(found) = keyword(body, at, name) {
        at = found + name.len();
        if found > 0 && bytes[found - 1] == b'.' {
            // A member of that name is another thing.
            continue;
        }
        if found > 0 && bytes[found - 1] == b'&' && !(found > 1 && bytes[found - 2] == b'&') {
            // A pointer to the counter.
            return true;
        }
        let rest = body[at..].trim_start();
        let assigns = [
            "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
        ]
        .iter()
        .any(|op| rest.starts_with(op))
            || (rest.starts_with('=') && !rest.starts_with("=="));
        if assigns {
            return true;
        }
    }
    false
}

/// The end of the statement that starts at `start` in `body`: a block, an
/// `if` with its `else` chain, a loop, or anything up to its `;`.
fn statement_end(body: &str, start: usize) -> Option<usize> {
    let bytes = body.as_bytes();
    let block_after = |from: usize| -> Option<usize> {
        // The first `{` outside parentheses, then its closing brace.
        let mut depth = 0usize;
        for (k, c) in bytes.iter().enumerate().skip(from) {
            match c {
                b'(' => depth += 1,
                b')' => depth = depth.checked_sub(1)?,
                b'{' if depth == 0 => return closing(body, k),
                b';' if depth == 0 => return None,
                _ => {}
            }
        }
        None
    };
    let starts = |word: &str| keyword(body, start, word) == Some(start);
    if bytes.get(start) == Some(&b'{') {
        return closing(body, start);
    }
    if starts("if") {
        let mut end = block_after(start + 2)?;
        loop {
            let next = end + body[end..].find(|c: char| !c.is_whitespace()).unwrap_or(0);
            if keyword(body, next, "else") != Some(next) {
                return Some(end);
            }
            end = block_after(next + 4)?;
        }
    }
    if ["for", "while", "loop", "switch"]
        .iter()
        .any(|word| starts(word))
    {
        return block_after(start);
    }
    let mut depth = 0usize;
    for (k, c) in bytes.iter().enumerate().skip(start) {
        match c {
            b'(' | b'{' | b'[' => depth += 1,
            b')' | b'}' | b']' => depth = depth.checked_sub(1)?,
            b';' if depth == 0 => return Some(k + 1),
            _ => {}
        }
    }
    None
}

/// `body` with each `if C {.. continue;}` among its statements turned into
/// `if C {..} else {the statements after it}`: the same turn of the loop
/// without leaving it. `None` when a `continue` stands anywhere else.
fn without_continue(body: &str) -> Option<String> {
    let mut out = String::with_capacity(body.len() + 16);
    let mut at = 0;
    loop {
        let Some(start) = body[at..]
            .find(|c: char| !c.is_whitespace() && c != KEPT)
            .map(|n| at + n)
        else {
            out.push_str(&body[at..]);
            return Some(out);
        };
        let end = statement_end(body, start)?;
        let statement = &body[start..end];
        if !leaves_loop(statement, "continue") {
            out.push_str(&body[at..end]);
            at = end;
            continue;
        }
        // Only `if C {.. continue;}` without an else, the continue last.
        if keyword(statement, 0, "if") != Some(0) {
            return None;
        }
        let brace = statement.find('{')?;
        if closing(statement, brace)? != statement.len() {
            return None;
        }
        let inner = statement[brace + 1..statement.len() - 1].trim_end();
        let kept = inner.strip_suffix("continue;")?;
        if leaves_loop(kept, "continue") {
            return None;
        }
        let rest = without_continue(&body[end..])?;
        out.push_str(&body[at..start]);
        out.push_str(&statement[..brace + 1]);
        out.push_str(kept);
        out.push_str("}else{");
        out.push_str(&rest);
        out.push('}');
        return Some(out);
    }
}

fn expand(text: &str, constants: &HashMap<String, Integer>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(start) = keyword(text, at, "for") {
        out.push_str(&text[at..start]);
        let Some(found) = parse(text, start, constants) else {
            out.push_str("for");
            at = start + 3;
            continue;
        };
        let body = expand(found.body, constants);
        let marked = text[..start].trim_end().ends_with(KEPT);
        let written = found.turns.as_ref().filter(|_| !marked).and_then(|turns| {
            let rewritten = without_continue(&body)?;
            (turns.len() * (rewritten.len() + found.name.len() + 16) <= MAX_BYTES
                && !leaves_loop(&rewritten, "break")
                && !writes(&rewritten, found.name))
            .then_some((turns, rewritten))
        });
        match written {
            Some((turns, body)) => {
                // Later turns stay on the first one's lines.
                let flat = body.replace('\n', " ");
                let suffix = if found.unsigned { "u" } else { "" };
                for (turn, k) in turns.iter().enumerate() {
                    out.push_str(&format!("{{let {}{}={k}{suffix};", found.name, found.kind));
                    out.push_str(if turn == 0 { &body } else { &flat });
                    out.push('}');
                }
                // A loop of no turns still holds its lines.
                if turns.is_empty() {
                    out.extend(body.chars().filter(|c| *c == '\n'));
                }
            }
            None => {
                out.push_str(&text[start..found.body_start]);
                out.push_str(&body);
                out.push('}');
            }
        }
        at = found.end;
    }
    out.push_str(&text[at..]);
    out
}

#[cfg(test)]
mod tests {
    use super::unroll;

    #[test]
    fn constant_loops_are_written_out_in_order() {
        assert_eq!(
            unroll("const N:u32=3u;fn f(){for(var k=0u;k<N;k++){a[k]=k;}}"),
            "const N:u32=3u;fn f(){{let k=0u;a[k]=k;}{let k=1u;a[k]=k;}{let k=2u;a[k]=k;}}"
        );
        assert_eq!(
            unroll("for(var y=-1;y<=1;y+=2){for(var x:i32=1;x>=0;x--){s+=g(x,y);}}"),
            "{let y=-1;{let x:i32=1;s+=g(x,y);}{let x:i32=0;s+=g(x,y);}}\
             {let y=1;{let x:i32=1;s+=g(x,y);}{let x:i32=0;s+=g(x,y);}}"
        );
        assert_eq!(
            unroll("for(var k=4u;k>0u;k/=2u){s+=a[k];}"),
            "{let k=4u;s+=a[k];}{let k=2u;s+=a[k];}{let k=1u;s+=a[k];}"
        );
    }

    #[test]
    fn a_continue_becomes_an_else() {
        assert_eq!(
            unroll("for(var k=0u;k<2u;k++){let v=a[k];if v>0.{n+=1u;continue;}b+=v;c+=1.;}"),
            "{let k=0u;let v=a[k];if v>0.{n+=1u;}else{b+=v;c+=1.;}}\
             {let k=1u;let v=a[k];if v>0.{n+=1u;}else{b+=v;c+=1.;}}"
        );
        // Two in a row nest; one inside another statement holds the loop.
        assert_eq!(
            unroll("for(var k=0u;k<1u;k++){if p{continue;}if q{continue;}b+=1.;}"),
            "{let k=0u;if p{}else{if q{}else{b+=1.;}}}"
        );
        let held = "for(var k=0u;k<2u;k++){if p{b+=1.;}else{continue;}c+=1.;}";
        assert_eq!(unroll(held), held);
    }

    #[test]
    fn other_loops_stay() {
        for source in [
            "for(var k=0u;k<cfg.n;k++){a[k]=k;}",
            "for(var k=0u;k<4u;k++){if a[k]>0.{break;}}",
            "for(var k=0u;k<4u;k++){k+=1u;}",
            "for(var k=0u;k<100u;k++){b+=1.;}",
        ] {
            assert_eq!(unroll(source), source);
        }
        // A marked loop stays, and the loops inside it are still written out.
        assert_eq!(
            unroll(
                "/*loop: slower written out*/for(var k=0u;k<2u;k++){for(var j=0u;j<2u;j++){b+=1.;}}"
            ),
            "for(var k=0u;k<2u;k++){{let j=0u;b+=1.;}{let j=1u;b+=1.;}}"
        );
        // An inner loop's own break does not hold the outer one.
        assert_eq!(
            unroll("for(var k=0u;k<2u;k++){for(var j=0u;j<n;j++){if j>k{break;}}}"),
            "{let k=0u;for(var j=0u;j<n;j++){if j>k{break;}}}{let k=1u;for(var j=0u;j<n;j++){if j>k{break;}}}"
        );
        // A member called like the counter is not the counter.
        assert_eq!(
            unroll("for(var x=0;x<2;x++){p.x=x;}"),
            "{let x=0;p.x=x;}{let x=1;p.x=x;}"
        );
    }

    #[test]
    fn lines_keep_their_numbers() {
        let source = "a; // one\nfor(var k=0u;k<2u;k++){\n  b[k]=1.; /* two\n three */\n}\nc;";
        let written = unroll(source);
        assert_eq!(written.lines().count(), source.lines().count());
        assert_eq!(written.lines().last(), Some("c;"));
        assert!(!written.contains("one") && !written.contains("three"));
    }
}
