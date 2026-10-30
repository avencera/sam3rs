"""Write tests/tokenizer_cases.json: token ids from the upstream SAM 3 tokenizer.

    _scratch/venv-torch/bin/python -I scripts/tokenizer_cases.py

The Rust tokenizer test compares against these. HTML entities and mojibake are left
out: sam3rs doesn't port upstream's `ftfy.fix_text` and `html.unescape` cleaning
"""

import json
import pathlib

import sam3
from sam3.model.tokenizer_ve import SimpleTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXTS = [
    "dog",
    "a dog",
    "Dog",
    "  two   dogs  ",
    "the red car's wheel",
    "it's we'll they'd I'm you've they're don't",
    "person in a yellow shirt",
    "1234",
    "3.5 inch screen",
    "hello, world!",
    "e-mail x86_64 C++",
    "café naïve façade",
    "Ünïcödé",
    "日本語のテキスト",
    "emoji 🐶 dog",
    "tab\tseparated\nlines",
    "",
    "<start_of_text>",
    "<end_of_text>",
    "dog<end_of_text>",
    "<START_OF_TEXT> dog",
    "a<start_of_text>b",
    "<|endoftext|>",
    "fish & chips",
    "a very long prompt that keeps going with many many words so that it runs past the "
    "thirty two token context and has to be truncated with the end token kept last",
]


def main() -> None:
    bpe = pathlib.Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    tokenizer = SimpleTokenizer(bpe_path=str(bpe))
    # VETextEncoder tokenizes with SAM 3's 32-token context, not the default 77
    cases = [{"text": t, "ids": tokenizer(t, context_length=32)[0].tolist()} for t in TEXTS]
    out = ROOT / "tests" / "tokenizer_cases.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
