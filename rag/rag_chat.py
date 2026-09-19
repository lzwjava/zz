#!/usr/bin/env python3
"""
Retrieval-augmented chat over lzwjava's notes.

Retrieves top-k chunks (hybrid dense + BM25) and answers with a grounded
prompt that cites sources as [1], [2], ...

Backends:
    llama  (default) llama.cpp + the fine-tuned GGUF from ../finetune
    hf               transformers (merged Safetensors dir or any HF model)
    vllm             vLLM (fastest on big GPUs)
    none             retrieval only, no generation

Usage:
    python rag_chat.py --question "How do I export a Qwen3 LoRA to GGUF?"
    python rag_chat.py --interactive
    python rag_chat.py -q "正则表达式" --backend none --show-context
    python rag_chat.py -q "vllm 部署" --k 6 --max-tokens 768

The default model is the LoRA-merged Qwen3-4B fine-tuned on these notes
(finetune/lzw-notes-merged_gguf/lzw-notes-merged.Q4_K_M.gguf), so answers
keep the notes' voice while staying grounded in retrieved text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from retriever import Retriever  # noqa: E402

FINETUNE_DIR = DIR.parent / "finetune"
DEFAULT_GGUF = FINETUNE_DIR / "lzw-notes-merged_gguf" / "lzw-notes-merged.Q4_K_M.gguf"
DEFAULT_HF_MODEL = FINETUNE_DIR / "lzw-notes-merged"

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant for lzwjava's technical notes.\n"
    "Answer the question using ONLY the numbered sources provided by the user.\n"
    "Cite the sources you use inline as [1], [2], etc. Never invent citations.\n"
    "If the sources do not contain the answer, say so plainly.\n"
    "Prefer concrete commands, code and numbers found in the sources.\n"
    "Answer in the language of the question."
)


def parse_args():
    p = argparse.ArgumentParser(description="RAG chat over the notes index")
    p.add_argument("--question", "-q", default=None)
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--index-dir", default=str(DIR / "rag_index"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.5, help="RRF weight for dense results")
    p.add_argument("--lang", default=None)
    p.add_argument("--context-chars", type=int, default=1200, help="Max chars per source")
    p.add_argument("--backend", choices=["llama", "hf", "vllm", "none"], default="llama")
    p.add_argument("--gguf", default=str(DEFAULT_GGUF))
    p.add_argument("--model", default=str(DEFAULT_HF_MODEL), help="HF model path for --backend hf/vllm")
    p.add_argument("--device", default=None, help="Embedding device (cuda / cpu)")
    p.add_argument("--ctx", type=int, default=8192, help="llama.cpp context size")
    p.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp GPU offload (-1 = all)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--think", action="store_true", help="Allow Qwen3 thinking mode")
    p.add_argument("--show-context", action="store_true")
    p.add_argument("--no-stream", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_messages(question: str, context: str, think: bool):
    user = f"Sources:\n\n{context}\n\nQuestion: {question}"
    if not think:
        user += " /no_think"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


LT, GT = "\u003c", "\u003e"
THINK_OPEN = f"{LT}think{GT}"
THINK_CLOSE = (
    f"{LT}/think{GT}",
    f"{LT}\uff5cend\u2581of\u2581thinking\uff5c{GT}",
)


class ThinkFilter:
    """Drop Qwen3 thinking blocks (`<think>...</think>`) from a streamed response.

    Tags are matched across token boundaries by buffering a short tail, and the
    buffer is dropped (not emitted) while inside a thinking block.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.buf = ""
        self.inside = False

    def _find_close(self) -> tuple[int, int]:
        best, size = -1, 0
        for tag in THINK_CLOSE:
            idx = self.buf.find(tag)
            if idx != -1 and (best == -1 or idx < best):
                best, size = idx, len(tag)
        return best, size

    def feed(self, text: str) -> str:
        if not self.enabled:
            return text
        self.buf += text
        out: list[str] = []
        while True:
            if self.inside:
                idx, size = self._find_close()
                if idx == -1:
                    break  # keep buffering; content inside a think block is dropped
                self.buf = self.buf[idx + size :]
                self.inside = False
            else:
                idx = self.buf.find(THINK_OPEN)
                if idx == -1:
                    # hold back a tail that could be the start of an open tag
                    safe = max(0, len(self.buf) - (len(THINK_OPEN) - 1))
                    out.append(self.buf[:safe])
                    self.buf = self.buf[safe:]
                    break
                out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(THINK_OPEN) :]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        out = "" if self.inside else self.buf
        self.buf = ""
        return out


# ---------------------------------------------------------------------------
# Generator backends
# ---------------------------------------------------------------------------

class LlamaBackend:
    def __init__(self, args):
        if not Path(args.gguf).exists():
            raise SystemExit(
                f"GGUF not found: {args.gguf}\n"
                "Build it with finetune/export_gguf.py, or use --backend hf/none."
            )
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=args.gguf,
            n_ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )

    def stream(self, messages, args):
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )
        for chunk in out:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def complete(self, messages, args) -> str:
        res = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return res["choices"][0]["message"]["content"]


class HFBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _inputs(self, messages):
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tok(text, return_tensors="pt").to(self.model.device)

    def complete(self, messages, args) -> str:
        inputs = self._inputs(messages)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
            )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def stream(self, messages, args):
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            streamer=streamer,
        )
        Thread(target=self.model.generate, kwargs=kwargs).start()
        yield from streamer


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.LLM, self.SamplingParams = LLM, SamplingParams
        self.llm = LLM(model=args.model, max_model_len=args.ctx, gpu_memory_utilization=0.85)

    def complete(self, messages, args) -> str:
        params = self.SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        out = self.llm.chat(messages, params)
        return out[0].outputs[0].text

    def stream(self, messages, args):
        yield self.complete(messages, args)


def make_backend(args):
    if args.backend == "llama":
        return LlamaBackend(args)
    if args.backend == "hf":
        return HFBackend(args)
    if args.backend == "vllm":
        return VLLMBackend(args)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def answer(retriever: Retriever, backend, args, question: str) -> None:
    hits = retriever.hybrid(question, k=args.k, alpha=args.alpha, lang=args.lang)
    if not hits:
        print("No sources retrieved.")
        return

    context = retriever.format_context(hits, max_chars=args.context_chars)

    if args.show_context or backend is None:
        for i, hit in enumerate(hits, 1):
            label = hit["title"] + (f" › {hit['heading']}" if hit["heading"] else "")
            print(f"\n[{i}] {label}  ({hit['score']:.4f})  {hit['file']}")
            print(hit["text"][: args.context_chars])
        print("\n" + "-" * 78)
        if backend is None:
            return

    messages = build_messages(question, context, args.think)
    print("\nAnswer:\n", flush=True)

    if args.no_stream:
        text = backend.complete(messages, args)
        filt = ThinkFilter(not args.think)
        print(filt.feed(text) + filt.flush())
    else:
        filt = ThinkFilter(not args.think)
        for delta in backend.stream(messages, args):
            piece = filt.feed(delta)
            if piece:
                print(piece, end="", flush=True)
        tail = filt.flush()
        if tail:
            print(tail, end="", flush=True)
        print()

    print("\nSources:")
    for i, hit in enumerate(hits, 1):
        loc = f"{hit['file']}"
        if hit.get("url"):
            loc += f"  ->  {hit['url']}"
        print(f"  [{i}] {hit['title']}" + (f" › {hit['heading']}" if hit["heading"] else ""))
        print(f"      {loc}")


def main():
    args = parse_args()
    retriever = Retriever(
        Path(args.index_dir), device=args.device, use_bm25=True
    )
    print(
        f"Index: {retriever.meta['count']:,} chunks | embed: {retriever.model_name} | "
        f"backend: {args.backend}",
        file=sys.stderr,
    )
    backend = make_backend(args)

    if args.interactive or not args.question:
        print("RAG chat — empty line or Ctrl-D to quit.", file=sys.stderr)
        while True:
            try:
                question = input("\nquestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            answer(retriever, backend, args, question)
    else:
        answer(retriever, backend, args, args.question)


if __name__ == "__main__":
    main()
