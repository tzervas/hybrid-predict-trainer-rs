//! Text tokenization for language model training.
//!
//! This module provides tokenization utilities for converting raw text into
//! token sequences suitable for language model training. Supports both simple
//! byte-level tokenization and integration with advanced tokenizers.

/// Simple byte-level tokenizer (vocab_size=256).
/// Can be swapped for GPT-2 BPE tokenizer when tokenizers crate is wired in.
#[derive(Debug, Clone)]
pub struct ByteTokenizer {
    /// Sequence length for model inputs
    pub seq_length: usize,
}

impl ByteTokenizer {
    /// Creates a new byte tokenizer with specified sequence length.
    #[must_use]
    pub fn new(seq_length: usize) -> Self {
        Self { seq_length }
    }

    /// Returns the vocabulary size (256 for byte tokenization).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        256
    }

    /// Encode text to byte token IDs (0-255).
    ///
    /// Each UTF-8 byte is mapped to a token ID from 0 to 255.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Create (input, target) pair for language modeling.
    ///
    /// Returns None if text is too short (less than 2 tokens).
    /// The input is text[0:seq_length] and targets are text[1:seq_length+1].
    #[must_use]
    pub fn create_lm_pair(&self, text: &str) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut ids = self.encode(text);
        ids.truncate(self.seq_length + 1);
        if ids.len() < 2 {
            return None;
        }
        let input_ids = ids[..ids.len() - 1].to_vec();
        let targets = ids[1..].to_vec();
        Some((input_ids, targets))
    }

    /// Decode token IDs back to bytes (lossy for invalid UTF-8).
    ///
    /// Invalid UTF-8 sequences are replaced with the replacement character (U+FFFD).
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids.iter().filter_map(|&id| {
            if id < 256 {
                Some(id as u8)
            } else {
                None
            }
        }).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokenizer_new() {
        let tokenizer = ByteTokenizer::new(256);
        assert_eq!(tokenizer.seq_length, 256);
    }

    #[test]
    fn test_byte_tokenizer_vocab_size() {
        let tokenizer = ByteTokenizer::new(256);
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn test_encode() {
        let tokenizer = ByteTokenizer::new(256);
        let text = "hello";
        let tokens = tokenizer.encode(text);
        // 'h' = 104, 'e' = 101, 'l' = 108, 'l' = 108, 'o' = 111
        assert_eq!(tokens, vec![104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_create_lm_pair() {
        let tokenizer = ByteTokenizer::new(5);
        let text = "hello";
        let (input_ids, targets) = tokenizer.create_lm_pair(text).unwrap();
        // input: "hell", targets: "ello"
        assert_eq!(input_ids, vec![104, 101, 108, 108]);
        assert_eq!(targets, vec![101, 108, 108, 111]);
    }

    #[test]
    fn test_create_lm_pair_too_short() {
        let tokenizer = ByteTokenizer::new(256);
        let text = "x";
        let result = tokenizer.create_lm_pair(text);
        assert!(result.is_none());
    }

    #[test]
    fn test_create_lm_pair_truncation() {
        let tokenizer = ByteTokenizer::new(3);
        let text = "hello world";
        let (input_ids, targets) = tokenizer.create_lm_pair(text).unwrap();
        // Should truncate to 3+1=4 bytes: "hell"
        assert_eq!(input_ids.len(), 3);
        assert_eq!(targets.len(), 3);
    }

    #[test]
    fn test_decode() {
        let tokenizer = ByteTokenizer::new(256);
        let tokens = vec![104, 101, 108, 108, 111]; // "hello"
        let text = tokenizer.decode(&tokens);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = ByteTokenizer::new(256);
        let original = "hello world";
        let tokens = tokenizer.encode(original);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, original);
    }
}
