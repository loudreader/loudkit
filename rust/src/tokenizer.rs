//! The grapheme BPE tokenizer — a bit-parity port of the JS/Go tokenizers
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

/// Whitespace pre-tokenizer: `\w+|[^\w\s]+`, Unicode-aware.
fn whitespace_regex(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_kind = 0; // 0 none, 1 word, 2 non-word-non-space
    for c in text.chars() {
        let kind = if is_word(c) {
            1
        } else if c.is_whitespace() {
            0
        } else {
            2
        };
        if kind == 0 {
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
    merges: HashMap<(String, String), usize>,
    added: HashMap<String, usize>,
    splitter: DictSplitter,
    end_of_word: String,
}

/// Parse a tokenizer.json file.
pub fn parse(path: &str) -> Result<Tokenizer, String> {
    let buf = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| format!("{path}: bad JSON: {e}"))?;
    let model = &v["model"];
    let mut vocab = HashMap::new();
    if let Some(vocab_obj) = model["vocab"].as_object() {
        for (k, id) in vocab_obj {
            vocab.insert(k.clone(), id.as_u64().unwrap_or(0) as usize);
        }
    }
    let mut merges = HashMap::new();
    if let Some(merges_arr) = model["merges"].as_array() {
        for (i, m) in merges_arr.iter().enumerate() {
            if let Some(s) = m.as_str() {
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    merges.insert((parts[0].to_string(), parts[1].to_string()), i);
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
            let id = a["id"].as_u64().unwrap_or(0) as usize;
            added.insert(content.clone(), id);
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
                if let Some(rank) = self.merges.get(&(cur[i].clone(), cur[i + 1].clone())) {
                    if best_rank.is_none_or(|br| *rank < br) {
                        best_rank = Some(*rank);
                        best_idx = Some(i);
                    }
                }
            }
            let i = match best_idx {
                Some(i) => i,
                None => break,
            };
            let merged = cur[i].clone() + &cur[i + 1];
            let mut next = Vec::with_capacity(cur.len() - 1);
            next.extend_from_slice(&cur[..i]);
            next.push(merged);
            next.extend_from_slice(&cur[i + 2..]);
            cur = next;
        }
        cur
    }
}
