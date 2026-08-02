# Maykeye/TinyLLama-v0 tokenizer fixture

Source: https://huggingface.co/Maykeye/TinyLLama-v0

Tokenizer / config only (no vendored weights). Download weights from Hub:

```bash
hf download Maykeye/TinyLLama-v0
dyninfer generate --hf Maykeye/TinyLLama-v0 --prompt "Once upon a time"
```
