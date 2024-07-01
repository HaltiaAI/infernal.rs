use infernal_llm::{backend, Config, Llama2, Llm, Template, Utf8Constructor, DEFAULT_SYS_PROMPT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = backend::Metal::new()?;

    let mut llm: Llm<Llama2> = Llm::new(
        Config::new("~/.config/models/mistral-7b"),
        Template::new_llama2_chat(),
        DEFAULT_SYS_PROMPT,
        &mut backend,
    )?;

    llm.feed_prompt("Why is 42 the answer?")?;

    // Some returned tokens may be partial UTF-8 glyphs and so this is needed.
    let mut utf8_ctor = Utf8Constructor::default();
    while let Some(tok) = llm.infer_next(&mut backend)? {
        if let Some(s) = utf8_ctor.get(tok) {
            println!("{s}");
        }
    }

    // Clear logit,s context window, embeddings, etc.
    llm.clear();

    // ---------------

    // Alternate, simpler way:
    llm.feed_prompt("Why is 42 the answer?")?;
    llm.infer_loop(&mut backend, |s| println!("{s}"))?;

    Ok(())
}
