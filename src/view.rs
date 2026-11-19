//! The picture's views.

/// How the picture is drawn: the cells' plain colours, the optical picture,
/// or the flow's direction and speed as colours.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    Simple,
    #[default]
    Photoreal,
    Flow,
}
impl Mode {
    pub const ALL: [Self; 3] = [Self::Simple, Self::Photoreal, Self::Flow];

    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Photoreal => "Photoreal",
            Self::Flow => "Flow",
        }
    }

    /// The view after this one, around the list.
    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::Mode;

    #[test]
    fn next_goes_around_every_view() {
        let mut mode = Mode::default();
        let mut seen = Vec::new();
        for _ in Mode::ALL {
            seen.push(mode);
            mode = mode.next();
        }
        assert_eq!(mode, Mode::default());
        for view in Mode::ALL {
            assert!(seen.contains(&view), "{view:?} is never reached");
        }
    }
}
