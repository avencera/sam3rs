//! CLIP byte-level BPE tokenizer with SAM 3's 32-token context
//!
//! Text cleaning is whitespace collapsing and lowercasing. Upstream also runs
//! `ftfy.fix_text` and `html.unescape`, which only change mojibake and HTML entities and
//! are not ported. `<start_of_text>` and `<end_of_text>` in the text become their special
//! ids, as upstream's BPE cache makes them

use std::collections::HashMap;
use std::path::Path;

use eyre::{Result, WrapErr, eyre};
use regex::Regex;

pub const CONTEXT: usize = 32;
const SOT: u32 = 49406;
const EOT: u32 = 49407;
/// Upstream's spellings; Hugging Face's vocab.json names the same ids `<|startoftext|>`
/// and `<|endoftext|>`
const SPECIAL: [(&str, u32); 2] = [("<start_of_text>", SOT), ("<end_of_text>", EOT)];

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    pattern: Regex,
    whitespace: Regex,
}

/// GPT-2's reversible byte to printable-unicode map
fn bytes_to_unicode() -> [char; 256] {
    let mut printable: Vec<u32> =
        ('!' as u32..='~' as u32).chain('¡' as u32..='¬' as u32).chain('®' as u32..='ÿ' as u32).collect();
    let mut chars = printable.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    for (b, c) in printable.iter().zip(chars) {
        map[*b as usize] = char::from_u32(c).unwrap();
    }
    map
}

impl Tokenizer {
    /// Loads Hugging Face `vocab.json` and `merges.txt` from `dir`
    pub fn open(dir: &Path) -> Result<Self> {
        let vocab = std::fs::read_to_string(dir.join("vocab.json")).wrap_err("read vocab.json")?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&vocab)?;
        let merges = std::fs::read_to_string(dir.join("merges.txt")).wrap_err("read merges.txt")?;
        let ranks = merges
            .lines()
            .filter(|l| !l.starts_with("#version") && !l.is_empty())
            .enumerate()
            .map(|(i, l)| {
                let (a, b) = l.split_once(' ').ok_or_else(|| eyre!("bad merge line {l:?}"))?;
                Ok(((a.to_string(), b.to_string()), i))
            })
            .collect::<Result<_>>()?;
        let pattern =
            Regex::new(r"(?i)<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")?;
        Ok(Self { encoder, ranks, byte_encoder: bytes_to_unicode(), pattern, whitespace: Regex::new(r"\s+")? })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|p| self.ranks.get(&(p[0].clone(), p[1].clone())).map(|r| (*r, p[0].clone(), p[1].clone())))
                .min();
            let Some((_, first, second)) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let clean = self.whitespace.replace_all(text.trim(), " ").trim().to_lowercase();
        let mut ids = Vec::new();
        for m in self.pattern.find_iter(&clean) {
            if let Some((_, id)) = SPECIAL.iter().find(|(text, _)| *text == m.as_str()) {
                ids.push(*id);
                continue;
            }
            let mapped: String = m.as_str().bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                ids.push(*self.encoder.get(&piece).ok_or_else(|| eyre!("token {piece:?} not in vocab"))?);
            }
        }
        Ok(ids)
    }

    /// `[SOT] tokens [EOT]` padded with zeros to 32, truncated with EOT kept last
    pub fn tokenize(&self, text: &str) -> Result<[u32; CONTEXT]> {
        let mut tokens = vec![SOT];
        tokens.extend(self.encode(text)?);
        tokens.push(EOT);
        if tokens.len() > CONTEXT {
            tokens.truncate(CONTEXT);
            tokens[CONTEXT - 1] = EOT;
        }
        let mut out = [0u32; CONTEXT];
        out[..tokens.len()].copy_from_slice(&tokens);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::Tokenizer;

    /// Compares against upstream's tokenizer on the cases from scripts/tokenizer_cases.py.
    /// Reads vocab.json and merges.txt from `SAM3RS_TOKENIZER_DIR`, default models/sam3;
    /// openai/clip-vit-base-patch32 ships the same vocab and merges
    #[test]
    fn matches_upstream() {
        let dir = std::env::var_os("SAM3RS_TOKENIZER_DIR").map_or_else(|| PathBuf::from("models/sam3"), PathBuf::from);
        let tokenizer = Tokenizer::open(&dir).unwrap_or_else(|e| panic!("{e:#}: run scripts/fetch-models.sh"));
        let cases: Value = serde_json::from_str(include_str!("../tests/tokenizer_cases.json")).unwrap();
        let mut failures = Vec::new();
        for case in cases.as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let got = tokenizer.tokenize(text).unwrap();
            if got.as_slice() != want {
                failures.push(format!("{text:?}\n  got  {got:?}\n  want {want:?}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
