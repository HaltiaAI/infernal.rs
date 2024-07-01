use std::{
    collections::{BinaryHeap, HashMap},
    io::Write,
};

use crate::{tokenizer::Tokenizer, SpecialTokens, Token, TokenData, TokenId, Vocab, VocabType};

#[derive(Debug)]
pub struct Spm {
    token_to_id: HashMap<Token, TokenId>,
    id_to_token: Vec<TokenData>,
    special: SpecialTokens,

    // These are used as temporary buffers to avoid reallocations. The data is only relevant for
    // a single tokenization round and so the buffers are cleared before each call to
    // `Self::tokenize`.
    symbols: Vec<Utf8Symbol>,
    input: String,
    work_queue: BinaryHeap<Bigram>,
    rev_merge: HashMap<String, (usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
struct Utf8Symbol {
    /// Byte offset of this utf-8 symbol.
    offset: usize,
    /// Byte length of this utf-8 symbol.
    len: usize,
    /// The index of the previous symbol in the `symbols` vec.
    prev: Option<usize>,
    /// The index of the next symbol in the `symbols` vec.
    next: Option<usize>,
}

#[derive(Debug)]
struct Bigram {
    left: usize,
    right: usize,
    score: f32,
    len: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order so that the highest score is at the top of the heap.
        self.score.partial_cmp(&other.score).unwrap()
    }
}

impl Spm {
    pub fn new(vocab: Vocab) -> Self {
        assert!(
            matches!(vocab.typ, VocabType::Spm),
            "tokenizer only accepts SPM vocab"
        );
        Self {
            token_to_id: vocab.token_to_id,
            id_to_token: vocab.id_to_token,
            special: vocab.special,

            symbols: Vec::new(),
            input: String::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    pub fn id_to_token(&self, id: TokenId) -> Option<&Token> {
        if (id as usize) < self.id_to_token.len() {
            Some(&self.id_to_token[id as usize].text)
        } else {
            None
        }
    }

    pub fn byte_to_token(&self, byte: u8) -> Option<TokenId> {
        // Note that in the original code a buffer of size 7 is used due to the C null terminator
        // being housed there, which we don't need in Rust.
        let mut buf = [0u8; 6];
        write!(&mut buf[..], "<0x{byte:02X}>").unwrap();

        // Convert to str for printing or further usage
        let tok = std::str::from_utf8(&buf).unwrap();
        self.token_to_id.get(tok.as_bytes()).copied()
    }
}

impl Tokenizer for Spm {
    #[inline(always)]
    fn token_to_id(&self) -> &HashMap<Token, TokenId> {
        &self.token_to_id
    }

    #[inline(always)]
    fn id_to_token(&self) -> &[TokenData] {
        &self.id_to_token
    }

    #[inline(always)]
    fn special(&self) -> &SpecialTokens {
        &self.special
    }

    #[inline(always)]
    fn vocab_len(&self) -> usize {
        self.id_to_token.len()
    }

    fn tokenize(&mut self, input: &str, output: &mut Vec<TokenId>) {
        output.clear();
        self.clear();

        if input.is_empty() {
            return;
        }

        // Without adding this leading whitespace, we do not get the same results as
        // the original tokenizer.
        output.push(self.special.bos);

        self.escape_input(input);
        // Split string into utf-8 chars.
        self.split_utf8_symbols();

        // Seed the work queue with all possible 2-char tokens.
        for i in 1..self.symbols.len() {
            self.add_bigram(i - 1, i);
        }

        // Keep substituting the highest frequency pairs for as long as we can.
        while let Some(b) = self.work_queue.pop() {
            // If one of the symbols already got merged, skip it.
            let is_merged = self.symbols[b.left].len == 0
                || self.symbols[b.right].len == 0
                || self.symbols[b.left].len + self.symbols[b.right].len != b.len;
            if is_merged {
                continue;
            }

            // Merge the right symbol into the left one.
            self.symbols[b.left].len += self.symbols[b.right].len;
            self.symbols[b.right].len = 0;

            // Remove the right symbol from the chain.
            self.symbols[b.left].next = self.symbols[b.right].next;
            if let Some(right_next) = self.symbols[b.right].next {
                self.symbols[right_next].prev = Some(b.left);
            }

            // Find more substitutions.
            if let Some(left_prev) = self.symbols[b.left].prev {
                self.add_bigram(left_prev, b.left);
            }
            if let Some(left_next) = self.symbols[b.left].next {
                self.add_bigram(b.left, left_next);
            }
        }

        // Collect all substituted symbols (bigrams or otherwise) into output.
        let mut i = Some(0);

        while let Some(next) = i {
            let symbol = self.symbols[next];
            i = symbol.next;
            self.resegment(symbol, output);
        }
    }

    fn get_token_data(&self, id: TokenId) -> Option<&TokenData> {
        self.id_to_token.get(id as usize)
    }
}

pub(crate) const SPACE_RPLCMNT: &[u8] = &[0xe2, 0x96, 0x81];

// methods for `tokenize`
impl Spm {
    /// Reserves space in the input buffer, prepends the string with a whitespace char, and escapes
    /// the whitespaces with a special unicode sequence.
    fn escape_input(&mut self, input: &str) -> &str {
        let space_rplcmnt = std::str::from_utf8(SPACE_RPLCMNT).unwrap();

        let n_spaces = self.input.matches(' ').count();
        let new_length = input.len() + n_spaces * (space_rplcmnt.len() - 1);
        self.input
            .reserve(1 + new_length - self.input.len().min(new_length));
        self.input += space_rplcmnt;
        for ch in input.chars() {
            if ch == ' ' {
                self.input += space_rplcmnt;
            } else {
                self.input.push(ch);
            }
        }
        &self.input
    }

    fn clear(&mut self) {
        self.symbols.clear();
        self.input.clear();
        self.work_queue.clear();
        self.rev_merge.clear();
    }

    /// Splits escaped input string into utf-8 chars.
    fn split_utf8_symbols(&mut self) {
        let mut chars = self.input.chars().peekable();
        let mut offset = 0;
        let mut idx = 0;
        while let Some(ch) = chars.next() {
            let len = ch.len_utf8();
            self.symbols.push(Utf8Symbol {
                offset,
                prev: if offset == 0 { None } else { Some(idx - 1) },
                next: if chars.peek().is_some() {
                    Some(idx + 1)
                } else {
                    None
                },
                len,
            });
            offset += len;
            idx += 1;
        }
    }

    fn add_bigram(&mut self, left: usize, right: usize) {
        let left_sym = &self.symbols[left];
        let right_sym = &self.symbols[right];
        let bigram = &self.input[left_sym.offset..right_sym.offset + right_sym.len];
        if let Some(tid) = self.token_to_id.get(bigram.as_bytes()) {
            if let Some(token) = self.id_to_token.get(*tid as usize) {
                self.work_queue.push(Bigram {
                    left,
                    right,
                    score: token.score,
                    len: bigram.len(),
                });

                // TODO: avoid realloc if possible
                self.rev_merge.insert(bigram.to_owned(), (left, right));
            }
        }
    }

    // TODO#perf: unfortunately can't take symbol as ref here due to borrowing self mutably. check
    // if this shows up in profiles, thrashing cache lines or something.
    fn resegment(&mut self, symbol: Utf8Symbol, output: &mut Vec<TokenId>) {
        let text = &self.input[symbol.offset..symbol.offset + symbol.len];
        let bytes = text.as_bytes();
        if let Some(token) = self.token_to_id.get(bytes) {
            output.push(*token);
        } else if let Some((left, right)) = self.rev_merge.get(text).copied() {
            self.resegment(self.symbols[left], output);
            self.resegment(self.symbols[right], output);
        } else {
            // Output any symbols that did not form bigrams as bytes.
            for byte in bytes {
                let token = self
                    .byte_to_token(*byte)
                    .expect("no token mapping exists for byte '{byte}'");
                output.push(token);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::{CStr, CString},
        os::unix::prelude::OsStrExt,
    };

    use crate::{loader, Llama2, Params};

    use super::*;

    #[test]
    fn matches_llamacpp() {
        let path = dirs::home_dir()
            .expect("$HOME must be set")
            .join(".config/haltia/default.gguf");
        let params = Params::default();
        let (_, mut tokenizer, _) = loader::load::<Llama2>(&path, &params).unwrap();

        let mut our_tokens = Vec::new();
        tokenizer.tokenize("hello world\nhow are you?", &mut our_tokens);
        println!("our tokens: {:?}", our_tokens);

        unsafe {
            let path_cstr = CString::new(path.as_os_str().as_bytes()).unwrap();
            let params = llama_cpp_sys::llama_context_default_params();
            let their_model = llama_cpp_sys::llama_load_model_from_file(path_cstr.as_ptr(), params);
            let mut their_tokens = vec![0; 100];
            let n_tokens = llama_cpp_sys::llama_tokenize_with_model(
                their_model,
                "hello world\nhow are you?\0".as_ptr().cast(),
                their_tokens.as_mut_ptr(),
                their_tokens.len() as i32,
                true,
            );

            println!("n_tokens: {}", n_tokens);
            println!("llama.cpp tokens: {:?}", &their_tokens[..n_tokens as usize]);

            assert_eq!(&their_tokens[..n_tokens as usize], our_tokens);
        }
    }

    /// 7 consecutive spaces not correctly tokenized into token 5390.
    // FIXME
    #[test]
    fn regression_long_spaces() {
        const SYSTEM_NEGATIVE_PROMPT: &str = "Always assist with care, respect, and truth. Respond
                        with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or 
                        negative content. Ensure replies promote fairness and positivity.";

        let path = dirs::home_dir()
            .expect("$HOME must be set")
            .join(".config/haltia/default.gguf");
        let params = Params::default();
        let (_, mut infernal_tokenizer, _) = loader::load::<Llama2>(&path, &params).unwrap();

        let mut infernal_tokens = Vec::new();
        infernal_tokenizer.tokenize(SYSTEM_NEGATIVE_PROMPT, &mut infernal_tokens);
        println!("infernal tokens:  {:?}", infernal_tokens);

        println!(
            "infernal id=5390 to token={:?} (n={})",
            infernal_tokenizer.token_to_string(5390),
            infernal_tokenizer
                .token_to_string(5390)
                .unwrap()
                .as_bytes()
                .len(),
        );

        unsafe {
            let path_cstr = CString::new(path.as_os_str().as_bytes()).unwrap();
            let ggml_params = llama_cpp_sys::llama_context_default_params();
            let ggml_model =
                llama_cpp_sys::llama_load_model_from_file(path_cstr.as_ptr(), ggml_params);
            let mut ggml_tokens = vec![0; 256];
            let prompt_cstring = CString::new(SYSTEM_NEGATIVE_PROMPT).unwrap();
            let n_tokens = llama_cpp_sys::llama_tokenize_with_model(
                ggml_model,
                prompt_cstring.as_ptr().cast(),
                ggml_tokens.as_mut_ptr(),
                ggml_tokens.len() as i32,
                true,
            );

            let mut buf = [0u8; 256];
            let len = llama_cpp_sys::llama_token_to_piece_with_model(
                ggml_model,
                5390,
                buf.as_mut_ptr().cast(),
                buf.len() as i32,
            );
            buf[len as usize] = 0;
            let tok_string = CStr::from_ptr(buf.as_ptr().cast()).to_str().unwrap();
            println!("llama.cpp id=5390 to token={tok_string} (n={len})");

            println!("n_tokens: {}", n_tokens);
            println!("llama.cpp tokens: {:?}", &ggml_tokens[..n_tokens as usize]);

            assert_eq!(&ggml_tokens[..n_tokens as usize], infernal_tokens);
        }
    }
}
