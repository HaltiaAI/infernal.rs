use std::collections::HashMap;

use crate::{unescape_whitespace, Token, TokenData, TokenId, TokenText, TokenType};

pub mod spm;
// pub mod bpe;

#[derive(Clone, Copy, Debug)]
pub struct SpecialTokens {
    pub bos: i32,
    pub eos: i32,
    pub unk: i32,
    pub sep: i32,
    pub pad: i32,
    pub linefeed: i32,
}

impl SpecialTokens {
    // Default LLaMA special tokens
    const LLAMA_BOS_ID: TokenId = 1;
    const LLAMA_EOS_ID: TokenId = 2;
    const LLAMA_UNK_ID: TokenId = 0;
    const LLAMA_SEP_ID: TokenId = -1;
    const LLAMA_PAD_ID: TokenId = -1;
    const LINEFEED_ID: TokenId = 13;

    pub fn llama_with_overrides(
        bos_id: Option<TokenId>,
        eos_id: Option<TokenId>,
        unk_id: Option<TokenId>,
        sep_id: Option<TokenId>,
        pad_id: Option<TokenId>,
    ) -> Self {
        Self {
            bos: bos_id.unwrap_or(Self::LLAMA_BOS_ID),
            eos: eos_id.unwrap_or(Self::LLAMA_EOS_ID),
            unk: unk_id.unwrap_or(Self::LLAMA_UNK_ID),
            sep: sep_id.unwrap_or(Self::LLAMA_SEP_ID),
            pad: pad_id.unwrap_or(Self::LLAMA_PAD_ID),
            // TODO: is this correct? llama.cpp tokenizes it from \n directly but doesn't seem
            // necessary
            linefeed: Self::LINEFEED_ID,
        }
    }
}

pub trait Tokenizer {
    /// Returns the token to ID mapping.
    fn token_to_id(&self) -> &HashMap<Token, TokenId>;

    /// Returns the ID to token mapping.
    ///
    /// This is just an array, where the array indices are the IDs.
    fn id_to_token(&self) -> &[TokenData];

    /// Returns special tokens of this vocab.
    fn special(&self) -> &SpecialTokens;

    /// Returns the vocabulary length.
    fn vocab_len(&self) -> usize;

    /// Tokenizes the input into the output buffer, which is cleared first.
    fn tokenize(&mut self, input: &str, output: &mut Vec<TokenId>);

    /// Returns token metadata for the token ID.
    fn get_token_data(&self, id: TokenId) -> Option<&TokenData>;

    /// Converts a token id to the token itself, pretty formatted (i.e. may differ from canonical
    /// representation, depending on the tokenizer).
    ///
    /// Returns None if the token falls outside the model's vocabulary.
    #[inline(always)]
    fn token_to_string(&self, id: TokenId) -> Option<TokenText> {
        let token = self.get_token_data(id)?;
        match token.typ {
            TokenType::Normal => {
                let bytes = unescape_whitespace(&token.text);
                match std::str::from_utf8(&bytes) {
                    Ok(s) => Some(TokenText::String(s.to_string())),
                    Err(_) => Some(TokenText::Bytes(bytes)),
                }
            }
            // This is the following character in llama.cpp for some reason: 㖅
            TokenType::Unknown => String::from_utf8(vec![0xe2, 0x96, 0x85])
                .ok()
                .map(TokenText::String),
            TokenType::Byte => {
                let token = &token.text[3..5];
                // TODO: maybe move this into SPM as this is specific to that
                if token == [0x30, 0x0] {
                    // FIXME: Hacky, but it checks for "/0" and replaces it with "/n".
                    return Some(TokenText::String("\n".to_string()));
                }
                let byte = u8::from_str_radix(
                    std::str::from_utf8(token).expect("byte token not a valid utf-8"),
                    16,
                )
                .expect("byte token not a valid hex value");
                Some(TokenText::Byte(byte))
            }
            _ => None,
        }
    }
}
