#[derive(Debug, Clone)]
pub struct Template {
    prefix: String,
    suffix: String,
    system: String,
    end_of_turn: String,
    user: String,
    assistant: String,
    new_line: bool,
}

impl Template {
    pub fn new_llama2_chat() -> Self {
        Self {
            prefix: "".to_string(),
            suffix: "".to_string(),
            end_of_turn: String::new(),
            system: "system:".to_string(),
            user: "user:".to_string(),
            assistant: "assistant:".to_string(),
            new_line: false,
        }
    }

    pub fn new_mistral_openorca() -> Self {
        Self::new_llama2_chat()
    }

    pub fn new_mistral_instruct() -> Self {
        Self::new_llama2_chat()
    }

    pub fn new_zephyr_alpha() -> Self {
        Self {
            prefix: String::new(),
            suffix: "</s>".to_string(),
            end_of_turn: "<|end_of_turn|>".to_string(),
            system: "<|system|>".to_string(),
            user: "<|user|>".to_string(),
            assistant: "<|assistant|>".to_string(),
            new_line: true,
        }
    }

    pub fn prompt(&self, msg: &str) -> String {
        let new_line = if self.new_line { "\n" } else { " " };
        format!(
            "{}{}{}{}{}{}{}{}{}{}",
            new_line,
            self.prefix,
            self.user,
            new_line,
            msg,
            self.suffix,
            new_line,
            self.end_of_turn,
            self.prefix,
            self.assistant,
        )
    }

    pub fn system_prompt(&self, message: &str) -> String {
        let new_line = if self.new_line { "\n" } else { " " };
        format!(
            "{}{}{}{}{}",
            &self.prefix, &self.system, new_line, message, &self.suffix,
        )
    }

    pub fn user_prompt(&self, message: &str) -> String {
        let new_line = if self.new_line { "\n" } else { " " };
        format!(
            "{}{}{}{}{}",
            &self.prefix, &self.user, new_line, message, &self.suffix,
        )
    }

    pub fn assistant_prompt(&self, message: &str) -> String {
        let new_line = if self.new_line { "\n" } else { " " };
        format!(
            "{}{}{}{}{}",
            &self.prefix, &self.assistant, new_line, message, &self.suffix,
        )
    }

    pub fn custom_prompt(&self, role: &str, message: &str) -> String {
        let new_line = if self.new_line { "\n" } else { " " };
        format!(
            "{prefix}{role}{new_line}{message}{suffix}",
            prefix = &self.prefix,
            suffix = &self.suffix,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_mistral_system() {
        let system_message = "You are a helpful AI.";
        let template = Template::new_mistral_openorca();
        let result = template.system_prompt(system_message);
        let expected = "system:\nYou are a helpful AI.".to_string();
        assert_eq!(result, expected);
    }

    #[test]
    pub fn test_mistral_prompt() {
        let prompt_message = "Why is the sky blue?";
        let template = Template::new_mistral_openorca();
        let result = template.prompt(prompt_message);
        let expected = "\n<|im_start|>user\n\
            Why is the sky blue?<|im_end|>\n\
            <|im_start|>assistant:"
            .to_string();
        assert_eq!(result, expected);
    }
}
