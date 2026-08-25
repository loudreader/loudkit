//! Splitting text that is longer than one window — the port of
//! `loudkit.frontend.chunking`.
//!
//! A window carries about 255 speech tokens, roughly ten seconds. Anything
//! longer has to be split, generated in pieces and joined, and *where* the
//! splits fall is audible: a break at a full stop is inaudible, a break
//! mid-clause is not. That makes it an algorithm-layer decision rather than a
//! caller's convenience, and it must be identical in every port — a different
//! split is a different set of joins and therefore a different reading.
//! Python reference: `loudkit/frontend/chunking.py`.

/// Characters of prepared text per speech token.
///
/// Measured on the reference voice across English, Polish (after the
/// respelling funnel) and German: 0.53–0.64. The constant is the low end with
/// margin (0.5 < the 0.53 measured minimum) because it is used to *stay under*
/// a limit, never to predict a length — the middle of the range would let the
/// worst case overflow the window, and an overflow is a hard failure.
///
/// Must equal `loudkit.frontend.chunking.CHARS_PER_TOKEN`.
pub const CHARS_PER_TOKEN: f64 = 0.5;

/// How text longer than one window is split.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub enabled: bool,
    pub max_tokens: usize,
    pub prefix_tokens: usize,
    pub split_on: Vec<String>,
}

impl ChunkConfig {
    /// Validate the recipe, the way `loudkit.config.ChunkConfig.__post_init__` does.
    ///
    /// Python refuses four configurations here, and the ports were plain structs that
    /// read `max_tokens` straight from the manifest and accepted all of them. The
    /// second is the one that matters: `d8742aa` fixed "split_text hangs forever on a
    /// config the validator accepts" on the Python side only — a `max_tokens` small
    /// enough that `int(max_tokens * CHARS_PER_TOKEN)` is zero makes the splitter cut
    /// nothing and loop forever, which on a server is a wedged request holding the
    /// single-flight engine.
    ///
    /// # Errors
    /// Returns the same refusal Python raises, so a user who hits it in two
    /// languages reads the same sentence twice.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("chunking.max_tokens must be positive: 0".to_string());
        }
        let budget = (self.max_tokens as f64 * CHARS_PER_TOKEN) as usize;
        if budget < 1 {
            return Err(format!(
                "chunking.max_tokens={} leaves no character budget to split on \
                 (int({} * {CHARS_PER_TOKEN}) == 0); needs at least {}",
                self.max_tokens,
                self.max_tokens,
                (1.0 / CHARS_PER_TOKEN).ceil() as usize
            ));
        }
        if self.prefix_tokens >= self.max_tokens {
            return Err(format!(
                "chunking.prefix_tokens must be in [0, max_tokens): {}",
                self.prefix_tokens
            ));
        }
        if self.split_on.is_empty() {
            return Err(
                "chunking.split_on cannot be empty: there would be nowhere to break".to_string(),
            );
        }
        Ok(())
    }
}

impl Default for ChunkConfig {
    /// The shipping recipe.
    fn default() -> Self {
        Self {
            enabled: true,
            max_tokens: 255,
            prefix_tokens: 6,
            split_on: [". ", "! ", "? ", "; ", ", "]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }
}

/// A conservative upper estimate of the speech tokens `text` will produce.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() as f64 / CHARS_PER_TOKEN) as usize + 1
}

/// Split `text` into pieces that each fit one window, in order, together
/// covering the input. Never empty for non-empty input.
///
/// Indexed by `char`, not by byte: a byte-indexed cut lands inside a multi-byte
/// character and produces invalid UTF-8 — the shape of bug the ports have had
/// before.
#[must_use]
pub fn split_text(text: &str, cfg: &ChunkConfig) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if !cfg.enabled || estimate_tokens(trimmed) <= cfg.max_tokens {
        return vec![trimmed.to_string()];
    }

    let budget = (cfg.max_tokens as f64 * CHARS_PER_TOKEN) as usize;
    let mut chunks: Vec<String> = Vec::new();
    let mut rest: Vec<char> = trimmed.chars().collect();

    while !rest.is_empty() {
        if rest.len() <= budget {
            chunks.push(rest.iter().collect::<String>().trim().to_string());
            break;
        }
        let head: String = rest[..(budget + 1).min(rest.len())].iter().collect();
        let mut cut: i64 = -1;
        // Strongest separator first, and within a separator the LATEST break,
        // so chunks run as long as they may rather than as short as they can.
        for sep in &cfg.split_on {
            if let Some(at) = head.rfind(sep.as_str()) {
                if at > 0 {
                    cut = (head[..at].chars().count() + sep.chars().count()) as i64;
                    break;
                }
            }
        }
        if cut <= 0 {
            // No punctuation in a whole window's worth of text. Break at the
            // last word boundary; it will be heard, and that is the point.
            if let Some(at) = head.rfind(' ') {
                cut = head[..at].chars().count() as i64;
            }
        }
        if cut <= 0 {
            cut = budget as i64; // one unbroken token longer than a window
        }
        // Never zero: a cut of 0 leaves `rest` unchanged and the loop spins.
        let cut = cut.max(1) as usize;

        chunks.push(rest[..cut].iter().collect::<String>().trim().to_string());
        rest = rest[cut..]
            .iter()
            .copied()
            .collect::<String>()
            .trim_start()
            .chars()
            .collect();
    }

    chunks.retain(|c| !c.is_empty());
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The splitter must cut where the shared fixture says.
    ///
    /// Where the splits fall is audible, so a different split is a different
    /// reading — not a formatting choice.
    /// `src/**` ships in the published crate and `tests/data/` does not, so a
    /// consumer running `cargo test` on the crate has no fixture to read. This
    /// returns instead of failing there, and `LOUDKIT_REQUIRE_ASSETS=1` turns
    /// that back into a failure for the runs that are supposed to have it -
    /// the same switch the integration tests use, so there is one rule.
    #[test]
    fn split_text_matches_the_shared_fixture() {
        let dir = std::env::var("LOUDKIT_FIXTURE_DIR")
            .unwrap_or_else(|_| "../tests/data/conformance".to_string());
        let raw = match std::fs::read_to_string(format!("{dir}/speechtext.json")) {
            Ok(raw) => raw,
            Err(e) => {
                assert!(
                    !std::env::var("LOUDKIT_REQUIRE_ASSETS")
                        .is_ok_and(|v| !v.is_empty() && v != "0"),
                    "LOUDKIT_REQUIRE_ASSETS is set and {dir}/speechtext.json is unreadable: {e}"
                );
                return;
            }
        };
        let payload: Value = serde_json::from_str(&raw).unwrap();
        let cases = payload["chunking"].as_array().expect("no chunking cases");
        assert!(!cases.is_empty(), "the fixture carries no chunking cases");

        for case in cases {
            let cfg = ChunkConfig {
                enabled: true,
                max_tokens: case["max_tokens"].as_u64().unwrap() as usize,
                prefix_tokens: case["prefix_tokens"].as_u64().unwrap() as usize,
                split_on: case["split_on"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
            };
            let want: Vec<String> = case["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let got = split_text(case["text"].as_str().unwrap(), &cfg);
            assert_eq!(got, want, "config {}", case["config"]);
        }
    }

    /// Shared arithmetic, not a tuning knob: a port that picks a different
    /// value splits in different places and reads the text differently.
    #[test]
    fn chars_per_token_matches_python() {
        assert!((CHARS_PER_TOKEN - 0.5).abs() < f64::EPSILON);
    }
}
