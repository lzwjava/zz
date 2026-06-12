#!/usr/bin/env python3
"""
Extract SFT dataset from lzwjava's Jekyll blog (en + zh).

Input:  ~/projects/jekyll-ai-blog/_posts/{en,zh}/*.md
Output: notes_sft.jsonl          (all examples)
        notes_sft_train.jsonl     (train split)
        notes_sft_eval.jsonl      (eval split, 200 examples)

Each line: {"conversations": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""

import json, re, random
from pathlib import Path

try:
    import frontmatter  # pip install python-frontmatter
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-frontmatter"])
    import frontmatter

POSTS_ROOT = Path.home() / "projects/jekyll-ai-blog/_posts"
OUT_DIR = Path(__file__).parent

EVAL_SIZE = 200
SEED = 42
MIN_BODY_LEN = 100  # chars — drop very short stubs

# Strip Jekyll/Liquid tags: {{ ... }}, {% ... %}
LIQUID = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
# Strip kramdown attribute lists like {: .centered } or {: .responsive }
KRAMDOWN = re.compile(r"\{:\s*\.[^}]*\}")
# Strip image refs like ![](assets/images/...)
IMG_REF = re.compile(r"!\[.*?\]\(.*?\)")
# Strip bare image markdown lines
CAPTION = re.compile(r"^\*Source:.*$", re.MULTILINE)
# Strip horizontal rules
HR = re.compile(r"^---\s*$", re.MULTILINE)
# Collapse multiple blank lines
BLANKS = re.compile(r"\n{3,}")


def clean_body(text: str) -> str:
    """Clean markdown body for SFT — remove liquid tags, kramdown, excessive whitespace."""
    text = LIQUID.sub("", text)
    text = KRAMDOWN.sub("", text)
    text = IMG_REF.sub("", text)
    text = CAPTION.sub("", text)
    text = BLANKS.sub("\n\n", text)
    return text.strip()


def extract_posts(lang_dir: Path, lang: str):
    """Yield (title, body) from all .md files in a language directory."""
    count = 0
    skipped_short = 0
    skipped_no_title = 0

    for f in sorted(lang_dir.rglob("*.md")):
        post = frontmatter.load(f)
        title = post.get("title", "").strip()
        if not title:
            skipped_no_title += 1
            continue

        body = clean_body(post.content)
        if len(body) < MIN_BODY_LEN:
            skipped_short += 1
            continue

        count += 1
        yield {
            "conversations": [
                {"role": "user", "content": title},
                {"role": "assistant", "content": body},
            ],
            "meta": {
                "file": str(f.relative_to(POSTS_ROOT.parent)),
                "lang": lang,
                "type": post.get("type", "post"),
                "generated": post.get("generated", False),
            },
        }

    print(f"  [{lang}] extracted {count}, skipped {skipped_no_title} (no title), {skipped_short} (too short)")


def main():
    random.seed(SEED)
    all_examples = []

    for lang in ["en", "zh"]:
        lang_dir = POSTS_ROOT / lang
        if not lang_dir.exists():
            print(f"  [{lang}] directory not found, skipping")
            continue
        for ex in extract_posts(lang_dir, lang):
            all_examples.append(ex)

    print(f"\nTotal examples: {len(all_examples)}")

    # Shuffle and split
    random.shuffle(all_examples)
    eval_set = all_examples[:EVAL_SIZE]
    train_set = all_examples[EVAL_SIZE:]

    # Write full dataset (without meta for training compatibility)
    def write_jsonl(path, examples, include_meta=False):
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {"conversations": ex["conversations"]}
                if include_meta:
                    record["meta"] = ex["meta"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "notes_sft.jsonl", all_examples)
    write_jsonl(OUT_DIR / "notes_sft_train.jsonl", train_set)
    write_jsonl(OUT_DIR / "notes_sft_eval.jsonl", eval_set, include_meta=True)

    print(f"  train: {len(train_set)}")
    print(f"  eval:  {len(eval_set)}")
    print(f"\nWrote to {OUT_DIR}/")

    # Quick token estimate
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        for ex in all_examples:
            for msg in ex["conversations"]:
                total_tokens += len(enc.encode(msg["content"], disallowed_special=()))
        print(f"\nEstimated total tokens: {total_tokens / 1e6:.1f}M")
    except ImportError:
        # Rough estimate: ~4 chars per token
        total_chars = sum(
            len(msg["content"])
            for ex in all_examples
            for msg in ex["conversations"]
        )
        print(f"\nEstimated total chars: {total_chars / 1e6:.1f}M (~{total_chars / 4 / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
