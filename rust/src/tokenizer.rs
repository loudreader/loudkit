//! The grapheme BPE tokenizer: a bit-parity port of the JS/Go tokenizers
//! over the HF tokenizers.json format. Splits on added-token contents (trie),
//! Whitespace-pretokenizes (`\w+|[^\w\s]+`, Unicode-aware), then BPE-merges.

use std::collections::HashMap;
use std::fs;

use serde_json::Value;

#[derive(Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    end: Option<String>,
}

struct DictSplitter {
    root: TrieNode,
}

impl DictSplitter {
    fn new(words: &[String]) -> Self {
        let mut root = TrieNode::default();
        for w in words {
            let mut node = &mut root;
            for c in w.chars() {
                node = node.children.entry(c).or_default();
            }
            node.end = Some(w.clone());
        }
        DictSplitter { root }
    }

    fn split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut result = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            let mut node = &self.root;
            let mut match_str: Option<String> = None;
            let mut j = i;
            while j < chars.len() {
                match node.children.get(&chars[j]) {
                    Some(next) => {
                        node = next;
                        if let Some(end) = &node.end {
                            match_str = Some(end.clone());
                        }
                        j += 1;
                    }
                    None => break,
                }
            }
            if let Some(m) = match_str {
                if i > start {
                    result.push(chars[start..i].iter().collect());
                }
                result.push(m.clone());
                i += m.chars().count();
                start = i;
            } else {
                i += 1;
            }
        }
        if start < chars.len() {
            result.push(chars[start..].iter().collect());
        }
        result
    }
}

/// Which of the pre-tokenizer's two alternations a character belongs to.
///
/// `\w+|[^\w\s]+` is two runs and a separator, and this is the separator named
/// rather than left as the 0, 1 and 2 the loop below used to compare.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Whitespace: matched by neither arm, so it ends the run it follows.
    Space,
    /// `\w`: the first arm.
    Word,
    /// `[^\w\s]`: the second arm, which runs together with its own kind only.
    Symbol,
}

/// Whitespace pre-tokenizer: `\w+|[^\w\s]+`, Unicode-aware.
fn whitespace_regex(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    // Never read while `cur` is empty, so the starting value stands for
    // nothing in particular.
    let mut cur_kind = Kind::Space;
    for c in text.chars() {
        let kind = if is_word(c) {
            Kind::Word
        } else if c.is_whitespace() {
            Kind::Space
        } else {
            Kind::Symbol
        };
        if kind == Kind::Space {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !cur.is_empty() && cur_kind != kind {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        cur_kind = kind;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn is_word(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory};
    c == '_'
        || c.is_alphabetic()
        || c.is_numeric()
        || matches!(
            get_general_category(c),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
}

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    unk_id: usize,
    /// Merge rank, indexed first by the left half of the pair and then by the
    /// right. Nested rather than keyed on a `(String, String)` tuple so that
    /// looking a pair up borrows the two pieces the caller already holds; the
    /// flat map could only be asked with a tuple it owned, which cost two
    /// `String` allocations per adjacent pair per merge round.
    merges: HashMap<String, HashMap<String, usize>>,
    added: HashMap<String, usize>,
    splitter: DictSplitter,
    end_of_word: String,
}

/// The vocabulary entry every space in the text becomes before the merges run.
pub(crate) const SPACE: &str = "[SPACE]";

/// The tokens the funnel writes by name rather than by BPE.
///
/// [`SPACE`] is substituted for every space, and the two others bracket the
/// row. A vocabulary without them still parses and still encodes, but a
/// `[SPACE]` that no entry names comes back as its own characters merged as
/// ordinary text, which is a different token stream under the same
/// fingerprint. `text.py` refuses these three by name, and so does this.
const REQUIRED_TOKENS: [&str; 3] = ["[START]", "[STOP]", SPACE];

/// Parse a tokenizer.json file.
///
/// # Errors
///
/// For an unreadable or non-JSON file, for a vocabulary entry whose id is not
/// a whole number, and for a vocabulary missing any of `REQUIRED_TOKENS`.
pub fn parse(path: &str) -> Result<Tokenizer, String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| format!("{path}: bad JSON: {e}"))?;
    let model = &v["model"];
    let mut vocab = HashMap::new();
    if let Some(vocab_obj) = model["vocab"].as_object() {
        for (k, id) in vocab_obj {
            // Refused, not folded to 0: an id this reader cannot read is a
            // token silently aliased onto whatever token 0 is.
            let id = id
                .as_u64()
                .ok_or_else(|| format!("{path}: vocabulary id for {k:?} is not a whole number"))?;
            vocab.insert(k.clone(), id as usize);
        }
    }
    for required in REQUIRED_TOKENS {
        if !vocab.contains_key(required) {
            return Err(format!("{path}: vocabulary is missing {required:?}"));
        }
    }
    let mut merges: HashMap<String, HashMap<String, usize>> = HashMap::new();
    if let Some(merges_arr) = model["merges"].as_array() {
        for (i, m) in merges_arr.iter().enumerate() {
            if let Some(s) = m.as_str() {
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    // A pair listed twice keeps the later rank, which is what
                    // the flat map did when the second `insert` overwrote the
                    // first.
                    merges
                        .entry(parts[0].to_string())
                        .or_default()
                        .insert(parts[1].to_string(), i);
                }
            }
        }
    }
    let unk = model["unk_token"].as_str().unwrap_or("[UNK]").to_string();
    let unk_id = vocab.get(&unk).copied().unwrap_or(1);
    let mut added = HashMap::new();
    let mut added_contents = Vec::new();
    if let Some(arr) = v["added_tokens"].as_array() {
        for a in arr {
            let content = a["content"].as_str().unwrap_or("").to_string();
            // Refused like a vocabulary id, and for the same reason.
            let id = a["id"]
                .as_u64()
                .ok_or_else(|| format!("{path}: added token {content:?} has no whole-number id"))?;
            added.insert(content.clone(), id as usize);
            added_contents.push(content);
        }
    }
    let end_of_word = model["end_of_word_suffix"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(Tokenizer {
        vocab,
        unk_id,
        merges,
        added,
        splitter: DictSplitter::new(&added_contents),
        end_of_word,
    })
}

impl Tokenizer {
    /// The largest id this vocabulary can emit, added tokens included.
    ///
    /// Every id from `encode` indexes the checkpoint's text embedding table
    /// directly, so this is the number the engine checks the table against.
    #[must_use]
    pub fn max_id(&self) -> usize {
        self.vocab
            .values()
            .chain(self.added.values())
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Encode a pre-tagged, normalised string into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for section in self.splitter.split(text) {
            if let Some(id) = self.added.get(&section) {
                ids.push(*id);
                continue;
            }
            for pretoken in whitespace_regex(&section) {
                for sub in self.bpe(&pretoken) {
                    match self.vocab.get(&sub) {
                        Some(id) => ids.push(*id),
                        None => ids.push(self.unk_id),
                    }
                }
            }
        }
        ids
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }
        let mut cur: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if !self.end_of_word.is_empty() {
            if let Some(last) = cur.last_mut() {
                last.push_str(&self.end_of_word);
            }
        }
        if cur.len() == 1 {
            return cur;
        }
        loop {
            let mut best_rank = None;
            let mut best_idx = None;
            for i in 0..cur.len() - 1 {
                let rank = self
                    .merges
                    .get(cur[i].as_str())
                    .and_then(|right| right.get(cur[i + 1].as_str()));
                if let Some(rank) = rank {
                    if best_rank.is_none_or(|br| *rank < br) {
                        best_rank = Some(*rank);
                        best_idx = Some(i);
                    }
                }
            }
            let Some(i) = best_idx else { break };
            // Merged in place. Rebuilding the row instead cloned every piece
            // that was not merging, once per merge round.
            let right = cur.remove(i + 1);
            cur[i].push_str(&right);
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer.json written to a temporary path, for the shapes `parse`
    /// must refuse. Small enough to write inline, which is the point: no
    /// fixture, so this runs on the published tarball too.
    fn write(name: &str, body: &str) -> String {
        let dir = std::env::temp_dir().join(format!("loudkit-tok-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        fs::write(&path, body).expect("write");
        path.to_string_lossy().into_owned()
    }

    /// A vocabulary missing a control token is refused by name.
    ///
    /// It parses and it encodes; what it produces is a `[SPACE]` merged as
    /// ordinary characters, which is a different token stream under the same
    /// fingerprint. `LOUDKIT_TOKENIZER` is the path that reaches this, since
    /// it lets a caller pair a checkpoint with a file by hand.
    #[test]
    fn a_vocabulary_missing_a_control_token_is_refused() {
        for missing in REQUIRED_TOKENS {
            let kept: Vec<String> = REQUIRED_TOKENS
                .iter()
                .filter(|t| **t != missing)
                .enumerate()
                .map(|(i, t)| format!("{}: {i}", serde_json::to_string(t).unwrap()))
                .collect();
            let path = write(
                &format!("missing-{}.json", missing.trim_matches(['[', ']'])),
                &format!(r#"{{"model":{{"vocab":{{{}}}}}}}"#, kept.join(",")),
            );
            let err = parse(&path)
                .err()
                .expect("a missing control token must be refused");
            assert!(err.contains(missing), "the error must name it: {err}");
        }
    }

    /// An id this reader cannot read is refused, not folded to token 0.
    ///
    /// Folded, it aliases that entry onto whatever token 0 is, silently.
    #[test]
    fn a_vocabulary_id_that_is_not_a_number_is_refused() {
        let path = write(
            "bad-id.json",
            r#"{"model":{"vocab":{"[START]":0,"[STOP]":1,"[SPACE]":"two"}}}"#,
        );
        let err = parse(&path)
            .err()
            .expect("a non-numeric id must be refused");
        assert!(err.contains("[SPACE]"), "{err}");
    }

    /// The three together are enough to load, so the refusals cost nothing a
    /// real tokenizer carries.
    #[test]
    fn the_three_control_tokens_are_enough_to_parse() {
        let path = write(
            "minimal.json",
            r#"{"model":{"vocab":{"[START]":0,"[STOP]":1,"[SPACE]":2}}}"#,
        );
        let tok = parse(&path).expect("a vocabulary with all three loads");
        assert_eq!(tok.max_id(), 2);
    }
}
