"""
Shared generation helpers: prompts, context budget, answer cleanup.
"""

import re
from typing import List


def format_passages(chunks: List[str]) -> str:
    parts = []
    for i, t in enumerate(chunks, start=1):
        parts.append(f"[Passage {i}]\n{t.strip()}")
    return "\n\n".join(parts)


def max_context_tokens_for_prompt(
    tokenizer,
    question: str,
    instruction_reserve: int = 220,
    safety: int = 48,
) -> int:
    """Reserve space for question + instruction so the full prompt fits model context."""
    max_len = getattr(tokenizer, "model_max_length", 1024) or 1024
    if max_len > 100_000:
        max_len = 1024
    q_tokens = len(tokenizer.encode(question, add_special_tokens=False))
    budget = max_len - q_tokens - instruction_reserve - safety
    return max(320, min(budget, 3072))


_RAG_INSTRUCTION_GPT2 = (
    "Read the passages below. Answer the question using only that information. "
    "Be direct and concise. If the passages do not contain the answer, say: "
    "The context does not contain enough information to answer. "
    "Do not repeat the question. Do not repeat the same sentence."
)


def get_prompt_gpt2(context: str, question: str) -> str:
    return (
        f"{_RAG_INSTRUCTION_GPT2}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def get_prompt_chat(context: str, question: str, model_name: str) -> str:
    instruction = (
        "Use ONLY the passages below to answer. Give a short, factual answer. "
        "If the passages do not contain the answer, reply exactly: "
        "The context does not contain enough information to answer. "
        "Do not repeat the question. Do not repeat phrases."
        f"\n\n{context}\n\nQuestion: {question}"
    )

    if "TinyLlama" in model_name:
        return (
            "<|system|>\nYou are a helpful assistant. Follow the user's instructions.\n"
            "<|user|>\n"
            f"{instruction}\n"
            "<|assistant|>\n"
        )

    if "phi" in model_name.lower():
        return f"Instruct: {instruction}\nOutput:"

    if "gemma" in model_name.lower():
        return (
            f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    return get_prompt_gpt2(context, question)


_NO_CTX_INSTRUCTION = (
    "Answer the question briefly and factually in one or two sentences. "
    "If you do not know, say you do not know. Do not invent names or facts. "
    "Do not repeat the question."
)


def get_prompt_no_retrieval(question: str, generator_model: str, use_chat_format: bool) -> str:
    if not use_chat_format:
        return f"{_NO_CTX_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"

    if "TinyLlama" in generator_model:
        return (
            "<|system|>\nYou are a helpful assistant.\n<|user|>\n"
            f"{_NO_CTX_INSTRUCTION}\n\nQuestion: {question}\n"
            "<|assistant|>\n"
        )

    if "phi" in generator_model.lower():
        return f"Instruct: {_NO_CTX_INSTRUCTION}\n\nQuestion: {question}\nOutput:"

    if "gemma" in generator_model.lower():
        return (
            f"<start_of_turn>user\n{_NO_CTX_INSTRUCTION}\n\nQuestion: {question}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

    return f"{_NO_CTX_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


def extract_chat_answer(text: str, model_name: str) -> str:
    if "TinyLlama" in model_name:
        for stop in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
            if stop in text:
                text = text.split(stop)[0]

    if "phi" in model_name.lower():
        for stop in ["\nInstruct:", "\nOutput:"]:
            if stop in text:
                text = text.split(stop)[0]

    if "gemma" in model_name.lower():
        for stop in ["<end_of_turn>", "<start_of_turn>"]:
            if stop in text:
                text = text.split(stop)[0]

    return text.strip()


def postprocess_generated_answer(text: str) -> str:
    """Remove prompt echoing and collapse obvious repetition."""
    if not text:
        return text

    t = text.strip()

    for marker in ("\nQuestion:", "\nQ:", "\nPassage 1]", "\nAnswer:"):
        idx = t.find(marker)
        if idx > 20:
            t = t[:idx].strip()
            break

    if "Question:" in t and t.index("Question:") > 30:
        t = t[: t.index("Question:")].strip()

    parts = re.split(r"(?<=[.!?])\s+", t)
    deduped = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if deduped and p.lower() == deduped[-1].lower():
            continue
        deduped.append(p)
    t = " ".join(deduped)

    words = t.split()
    if len(words) > 24:
        for n in range(min(16, len(words) // 2), 5, -1):
            span = words[:n]
            rest = words[n : n + n]
            if len(rest) == n and span == rest:
                t = " ".join(words[:n])
                break

    return t.strip()
