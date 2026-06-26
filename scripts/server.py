#!/usr/bin/env python3
"""OpenAI-compatible API server for GPT-2 using transformers + FastAPI."""
import torch, json, time, uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/workspace/model/hf-model"
app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to("cuda")
model.eval()
print(f"Model loaded on GPU. Vocab: {tokenizer.vocab_size}")

class CompletionRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "sec-edgar-gpt-124m"
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
        )
    gen_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    elapsed = time.time() - t0
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(gen_ids),
            "total_tokens": input_ids.shape[1] + len(gen_ids),
        },
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Simple: concatenate messages as prompt
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
        )
    gen_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"message": {"role": "assistant", "content": text}, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(gen_ids),
            "total_tokens": input_ids.shape[1] + len(gen_ids),
        },
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def models():
    return {"data": [{"id": "sec-edgar-gpt-124m", "object": "model", "owned_by": "lzwjava"}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
