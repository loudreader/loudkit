//! Splitting text that is longer than one window: the port of
//! `loudkit.frontend.chunking`.
//!
//! A window carries about 255 speech tokens, roughly ten seconds. Anything
//! longer has to be split, generated in pieces and joined, and *where* the
//! splits fall is audible: a break at a full stop is inaudible, a break
//! mid-clause is not. That makes it an algorithm-layer decision rather than a
//! caller's convenience, and it must be identical in every port: a different
//! split is a different set of joins and therefore a different reading.
//!
//! Python reference: `loudkit/frontend/chunking.py`.

/// Characters of prepared text per speech token.
///
/// Measured on the reference voice across English, Polish (after the
/// respelling funnel) and German: 0.53–0.64. The constant is the low end with
/// margin (0.5 < the 0.53 measured minimum) because it is used to *stay under*
/// a limit, never to predict a length: the middle of the range would let the
/// worst case overflow the window more often still.
///
/// It is a budget, not a guarantee. Measured over 9920 rendered chunks in
/// ten languages, 54 overran the window anyway: a mean cannot bound a
/// variance, and most of those are chunks that should have fitted and did
/// not because the model emitted no stop token in time. An overflow is not
/// an error either: the generator stops at the cap mid-word and the
/// remainder is never spoken, which is what `cap_resplit` and `split_in_half`
/// exist for.
///
/// Must equal `loudkit.frontend.chunking.CHARS_PER_TOKEN`.
pub const CHARS_PER_TOKEN: f64 = 0.5;

/// The two spellings of [`ChunkConfig::cap_resplit`].
///
/// [`split_text`] budgets characters against a constant, and a speaker slower
/// than it fills the window before the text runs out; the generator then stops
/// at the cap mid-word and the remainder is lost, because chunk texts are fixed
/// before any of them renders. [`WORD_CAP_RESPLIT`] halves such a chunk and
/// generates both halves in its place. [`OFF_CAP_RESPLIT`] ships the truncated
/// window, which is what every checkpoint built before this field did.
pub const WORD_CAP_RESPLIT: &str = "word";
/// See [`WORD_CAP_RESPLIT`].
pub const OFF_CAP_RESPLIT: &str = "off";

/// The two spellings of [`ChunkConfig::mid_sentence_period`].
///
/// A period that does not end a sentence is not a boundary. Breaking on one
/// cuts `"But Mr. Smith went home"` after the title and hands the renderer a
/// seven-character chunk with its own derived seed and a token ceiling
/// proportional to seven characters. [`HOLD_MID_SENTENCE_PERIOD`] is the law;
/// [`BREAK_MID_SENTENCE_PERIOD`] is the other spelling, kept namable so a pack
/// can say what it was measured under.
pub const HOLD_MID_SENTENCE_PERIOD: &str = "hold";
/// The law before [`HOLD_MID_SENTENCE_PERIOD`]. See it for the whole story.
pub const BREAK_MID_SENTENCE_PERIOD: &str = "break";

/// How text longer than one window is split.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub enabled: bool,
    pub max_tokens: usize,
    pub prefix_tokens: usize,
    pub split_on: Vec<String>,
    /// Written forms whose following period does not end a sentence, given
    /// without that period: the period is the separator's.
    ///
    /// One union list for every language: surveyed over 1200 passages in ten,
    /// a language-blind union re-chunks the corpus identically to ten
    /// per-language lists. Data, not code: replacing the vector is the whole of
    /// adding a language. Not the funnel's list, which maps a written
    /// abbreviation to spoken words and carries only the unambiguous ones; what
    /// reaches here is the residue the funnel refuses to touch.
    pub abbreviations: Vec<String>,
    /// [`HOLD_MID_SENTENCE_PERIOD`] or [`BREAK_MID_SENTENCE_PERIOD`].
    pub mid_sentence_period: String,

    /// [`WORD_CAP_RESPLIT`] or [`OFF_CAP_RESPLIT`].
    pub cap_resplit: String,
}

impl ChunkConfig {
    /// How many characters one chunk may hold, `int(max_tokens *
    /// CHARS_PER_TOKEN)` as Python writes it.
    ///
    /// The truncation is the arithmetic, not a rounding choice: the validator
    /// refuses a `max_tokens` small enough to make this zero, and it can only
    /// refuse the same number the splitter later uses.
    fn budget(&self) -> usize {
        (self.max_tokens as f64 * CHARS_PER_TOKEN) as usize
    }

    /// Validate the recipe, the way `loudkit.config.ChunkConfig.__post_init__` does.
    ///
    /// Four configurations, the same four Python refuses. The second is the one
    /// that matters: a `max_tokens` small enough that
    /// `int(max_tokens * CHARS_PER_TOKEN)` is zero makes the splitter cut
    /// nothing and loop forever, which on a server is a wedged request holding
    /// the single-flight engine.
    ///
    /// # Errors
    /// Returns the same refusal Python raises, so a user who hits it in two
    /// languages reads the same sentence twice.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("chunking.max_tokens must be positive: 0".to_string());
        }
        let budget = self.budget();
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
        if self.mid_sentence_period != HOLD_MID_SENTENCE_PERIOD
            && self.mid_sentence_period != BREAK_MID_SENTENCE_PERIOD
        {
            return Err(format!(
                "unknown mid_sentence_period {:?}: expected \
                 {HOLD_MID_SENTENCE_PERIOD:?} or {BREAK_MID_SENTENCE_PERIOD:?}",
                self.mid_sentence_period
            ));
        }
        if self.cap_resplit != WORD_CAP_RESPLIT && self.cap_resplit != OFF_CAP_RESPLIT {
            return Err(format!(
                "unknown cap_resplit {:?}: expected {WORD_CAP_RESPLIT:?} or {OFF_CAP_RESPLIT:?}",
                self.cap_resplit
            ));
        }
        // An empty entry is a suffix of everything, so it would hold every
        // candidate and drive every split down to a word boundary. Silent, and
        // audible on every long passage.
        if self.abbreviations.iter().any(String::is_empty) {
            return Err("chunking.abbreviations cannot contain an empty string".to_string());
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
            // The surveyed union, sorted. It must equal
            // `loudkit.config.ChunkConfig.abbreviations`.
            abbreviations: [
                "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M", "Mr", "Mrs", "R",
                "S", "St", "T", "V", "Vors", "dr", "mrs", "prof", "\u{15b}w",
            ]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
            mid_sentence_period: HOLD_MID_SENTENCE_PERIOD.to_string(),
            cap_resplit: WORD_CAP_RESPLIT.to_string(),
        }
    }
}

/// Whether `b` is an ASCII letter or digit.
///
/// A byte, not a `char`, and not `char::is_alphanumeric`: the five
/// implementations have to answer this identically, and every language's idea
/// of "letter" is its own. ASCII is the part they cannot disagree on, and a
/// UTF-8 continuation byte is never mistaken for one.
fn is_ascii_word(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// Whether the candidate at byte offset `at` is a period inside a sentence
/// rather than the end of one.
///
/// `look` is the search window plus one character, because the test below reads
/// the character *after* the separator and the latest candidate can end the
/// window exactly.
///
/// Gated on the period: `"! "` and `"? "` end sentences and `"; "` and `", "` do
/// not end them at all, so neither is ever in doubt. The whole question is about
/// the one mark that is written for two jobs.
fn holds(look: &str, at: usize, sep: &str, cfg: &ChunkConfig) -> bool {
    if cfg.mid_sentence_period != HOLD_MID_SENTENCE_PERIOD || !sep.starts_with('.') {
        return false;
    }
    // A sentence does not resume in lower case. This is what catches the
    // ellipsis the funnel folds to a single period, which no abbreviation list
    // reaches. ASCII only, and measured rather than assumed: over 2253 periods
    // in ten languages, four are followed by a word starting with a non-ASCII
    // lowercase letter, and reading the whole Unicode Lowercase property
    // instead moves one passage in 1200.
    let after = at + sep.len();
    let bytes = look.as_bytes();
    if after < bytes.len() && bytes[after].is_ascii_lowercase() {
        return true;
    }
    // Or the token in front of the period is a listed abbreviation. Entries
    // carry no period of their own: the period belongs to the separator.
    let boundary = &look[..at];
    for abbreviation in &cfg.abbreviations {
        if !boundary.ends_with(abbreviation.as_str()) {
            continue;
        }
        let before = boundary.len() - abbreviation.len();
        if before > 0 && is_ascii_word(boundary.as_bytes()[before - 1]) {
            continue; // the tail of a longer word, not a word of its own
        }
        return true;
    }
    false
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
/// character and produces invalid UTF-8: the shape of bug the ports have had
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

    let budget = cfg.budget();
    let mut chunks: Vec<String> = Vec::new();
    let mut rest: Vec<char> = trimmed.chars().collect();

    while !rest.is_empty() {
        if rest.len() <= budget {
            chunks.push(rest.iter().collect::<String>().trim().to_string());
            break;
        }
        let head: String = rest[..(budget + 1).min(rest.len())].iter().collect();
        // One character past the window, and used only by `holds`: the latest
        // candidate can end the window exactly, and the test reads the
        // character after it. The search itself stays inside the budget.
        let look: String = rest[..(budget + 2).min(rest.len())].iter().collect();
        // `None` is no cut found, and so is `Some(0)`: a cut of zero leaves
        // `rest` unchanged and spins the loop, so the fallbacks treat the two
        // the same way.
        let mut cut: Option<usize> = None;
        // Strongest separator first, and within a separator the LATEST break,
        // so chunks run as long as they may rather than as short as they can.
        for sep in &cfg.split_on {
            let mut found = head.rfind(sep.as_str());
            // A period inside a sentence is not a boundary, so the search keeps
            // walking back through this separator's own occurrences before it
            // gives up and tries a weaker one. Searching `head[..at]` skips an
            // occurrence overlapping the held one, which no separator here can
            // have.
            while let Some(at) = found {
                if at == 0 || !holds(&look, at, sep, cfg) {
                    break;
                }
                found = head[..at].rfind(sep.as_str());
            }
            if let Some(at) = found {
                if at > 0 {
                    cut = Some(head[..at].chars().count() + sep.chars().count());
                    break;
                }
            }
        }
        if cut.is_none() {
            // No punctuation in a whole window's worth of text. Break at the
            // last word boundary; it will be heard, and that is the point.
            // WORD_BOUNDARIES, not U+0020: NBSP survives the funnel and is
            // ordinary in real prose, so text whose every space is
            // non-breaking found no boundary here and got cut mid-word.
            if let Some(at) = head.rfind(WORD_BOUNDARIES) {
                cut = Some(head[..at].chars().count());
            }
        }
        // No usable cut is one unbroken token longer than a window.
        let cut = cut.filter(|&at| at > 0).unwrap_or(budget).max(1);

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

/// Characters [`split_in_half`] may cut on, written out rather than tested for.
///
/// A predicate would be shorter and the five ports do not agree on one:
/// measured, Python's `str.isspace()` treats U+001C-U+001F as whitespace
/// where the other four do not, and JS alone KEEPS U+0085 where the other
/// four strip it. Swift alone strips U+200B, JS alone strips U+FEFF; the
/// funnel removes both before the splitter sees them, but U+0085 and
/// U+001C-U+001F survive it. A disagreement there is a different split point,
/// which is different audio for the same text and seed. A hand-written table
/// cannot drift.
///
/// The funnel does not remove these. NBSP in particular is ordinary in real
/// prose ("10 000", French punctuation, typeset copy), so a capped chunk whose
/// only boundaries are NBSP comes back unsplittable and ships its truncation
/// unless NBSP is on this list.
pub const WORD_BOUNDARIES: [char; 7] = [
    '\u{0020}', '\u{0009}', '\u{000a}', '\u{000d}', '\u{00a0}', '\u{2007}', '\u{202f}',
];

/// Halve a chunk the window could not hold, at a word boundary.
///
/// [`split_text`]'s estimate is conservative but not a guarantee: it budgets
/// characters against a constant, and a speaker slower than that constant
/// fills the window before the text runs out. The generator then stops at the
/// cap mid-word, and the words that did not fit are *lost* rather than
/// deferred, because chunk texts are fixed before any of them is rendered.
/// Measured across ten languages, 54 of 9920 chunks reached a cap and 30
/// were still speaking when it closed, over five voices; gating on the
/// window leaves 51 and 27, on two.
///
/// The boundary is the nearest WORD break, and punctuation is not sought. The
/// reason is mechanical rather than comparative: a comma is an instruction to
/// pause, this model has no pause-duration prior, and fed one it overshoots.
/// No measurement compares seeking punctuation against this law; see
/// `docs/design/text-funnel.md`. Not sought is not avoided: the nearest word
/// break can follow a comma.
///
/// `None` for a single unbroken run. Splitting it would have to cut a word,
/// which is worse than the truncation it would be repairing.
pub fn split_in_half(text: &str) -> Option<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let middle = chars.len() as f64 / 2.0;
    let mut best: Option<(f64, usize)> = None;
    for (i, c) in chars.iter().enumerate() {
        // Interior only: a boundary at either end yields an empty half.
        if !WORD_BOUNDARIES.contains(c) || i == 0 || i + 1 >= chars.len() {
            continue;
        }
        let d = (i as f64 - middle).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i + 1));
        }
    }
    let (_, at) = best?;
    let first: String = chars[..at].iter().collect();
    let second: String = chars[at..].iter().collect();
    let (first, second) = (first.trim().to_string(), second.trim().to_string());
    // Trimming can empty a half the scan thought was interior, on input whose
    // boundary run is all whitespace. `split_text` trims before this is ever
    // called, so the engine cannot reach it, but this is public and a caller
    // handed an empty half would render silence and call it speech.
    if first.is_empty() || second.is_empty() {
        return None;
    }
    Some((first, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The splitter must cut where the shared fixture says.
    ///
    /// Where the splits fall is audible, so a different split is a different
    /// reading, not a formatting choice.
    /// Declines when the fixture is absent, by the rule in
    /// [`crate::shared_fixture`]: the published crate ships `src/**` and not
    /// `tests/data/`.
    #[test]
    fn split_text_matches_the_shared_fixture() {
        let Some(payload) = crate::shared_fixture("speechtext.json") else {
            return;
        };
        let cases = payload["chunking"].as_array().expect("no chunking cases");
        assert!(!cases.is_empty(), "the fixture carries no chunking cases");

        for case in cases {
            let cfg = ChunkConfig {
                cap_resplit: WORD_CAP_RESPLIT.to_string(),
                enabled: true,
                max_tokens: case["max_tokens"].as_u64().unwrap() as usize,
                prefix_tokens: case["prefix_tokens"].as_u64().unwrap() as usize,
                split_on: case["split_on"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
                abbreviations: case["abbreviations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
                mid_sentence_period: case["mid_sentence_period"].as_str().unwrap().to_string(),
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

    /// A period that does not end a sentence must not end a chunk.
    ///
    /// Breaking on one cuts `"But Mr. Smith went home"` after the title and
    /// hands the renderer a seven-character chunk: its own utterance, its own
    /// derived seed, and a token ceiling proportional to seven characters. It
    /// is the only chunk in a 9920-row rendered census that hit that ceiling.
    /// Surveyed over 1200 passages in ten languages, 59 of 3773 cuts landed on
    /// a period inside a sentence; under this law, one.
    #[test]
    fn mid_sentence_periods_are_not_boundaries() {
        const TITLE: &str = "But Mr. Smith went home to the house on the hill where he had \
                             lived for forty years without ever once complaining about any \
                             of it at all.";
        assert_ne!(
            split_text(TITLE, &ChunkConfig::default())[0],
            "But Mr.",
            "the hold did not fire"
        );

        // The old law, kept namable so a pack can say what it was measured
        // under.
        let old = ChunkConfig {
            mid_sentence_period: BREAK_MID_SENTENCE_PERIOD.to_string(),
            ..ChunkConfig::default()
        };
        assert_eq!(split_text(TITLE, &old)[0], "But Mr.");

        // The half of the law that no list could do: `speech_text` folds a
        // mid-sentence ellipsis to a single period, and the next word is lower
        // case. It is the dominant cause in Polish, which has no abbreviation
        // cuts at all.
        const ELLIPSIS: &str = "Grzeja sie i swieca. ciepłem ktore pamietaja z lata i z \
                                kazdej innej pory roku na swiecie, a potem gasna powoli i \
                                nikt juz nie pamieta.";
        assert!(!split_text(ELLIPSIS, &ChunkConfig::default())[0].ends_with("swieca."));

        // Gated on the period. A comma is followed by a lower-case word almost
        // every time it is written, so a rule that did not gate would veto
        // every comma in the language.
        const COMMAS: &str = "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa \
                              lambda mu, nu xi omicron pi rho, sigma tau upsilon phi chi \
                              psi omega at the end.";
        assert_eq!(
            split_text(COMMAS, &ChunkConfig::default()),
            split_text(COMMAS, &old)
        );

        // "NASA" ends in "A", and "A" is a listed initial; the guard on the
        // character in front of the match is what keeps this one breaking.
        const NASA: &str = "The rocket that carried them up there was built by NASA. And \
                            the rest of the afternoon went by without anybody saying much \
                            about it to anyone.";
        assert!(split_text(NASA, &ChunkConfig::default())[0].ends_with("by NASA."));

        // Every sentence end in the window is held, so the split falls through
        // to the latest comma. A comma break is heard; a chunk of "Mr." is
        // heard worse.
        const NORRELL: &str = "Mr. Norrell, who had been waiting in the hall for the better \
                               part of an hour, said nothing at all to either of them about \
                               what he had seen there.";
        assert!(split_text(NORRELL, &ChunkConfig::default())[0].ends_with("hour,"));
    }
}
