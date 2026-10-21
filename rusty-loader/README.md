# rusty-loader

Model file loading for [Rusty](../README.md): safetensors (single file and
sharded through `model.safetensors.index.json`), Llama-style `config.json`
parsing, an affine-quantised weight upload path, and a `Tokenizer` with
chat-template encoding and a `Sampler`.

```rust,ignore
use rusty_loader::Tokenizer;

let tok = Tokenizer::from_file("./models/tinyllama/tokenizer.json")?;
let ids = tok.encode("Hello, world");
println!("{}", tok.decode(&ids));
```

The snippet needs a model directory on disk, so it is not run in CI.
