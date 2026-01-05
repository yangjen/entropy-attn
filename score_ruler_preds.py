# score_ruler_preds.py
'''
# Re-score baseline with no judge (fast):
# python score_ruler_preds.py \
#   --pred_path /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/predictions.jsonl \
#   --out_path  /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/predictions.baseline.rescored.jsonl

# Re-score baseline with inline judge only for needs_judge (recommended):
# CUDA_VISIBLE_DEVICES=3 python score_ruler_preds.py \
#   --pred_path /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/predictions.jsonl \
#   --out_path  /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/predictions.baseline.judged.rescored.jsonl \
#   --use_judge \
#   --judge_model meta-llama/Llama-3.1-8B-Instruct

Judge every non-EM case (slowest, but most thorough):
CUDA_VISIBLE_DEVICES=1 python score_ruler_preds.py \
  --pred_path /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/baseline_pred/qa_1.jsonl \
  --out_path  /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/baseline_pred/qa_1.baseline.rescored.jsonl \
  --use_judge \
  --judge_on_all_non_em

'''
# score_ruler_preds.py
import os
import json
import re
import string
import argparse
from typing import Any, Dict, Iterator, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# JSONL utils
# ---------------------------
def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def fmt_ratio(k: int, n: int) -> str:
    return f"{k}/{n} ({(k/max(n,1))*100:.1f}%)"


# ---------------------------
# Normalization + heuristics
# ---------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_PUNCT_NO_COMMA = str.maketrans("", "", string.punctuation.replace(",", ""))

def normalize_strict(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)      # remove leaked special tokens
    s = s.translate(_PUNCT_TABLE)         # remove punctuation (incl commas)
    s = " ".join(s.split())
    s = re.sub(r"^(the|a|an)\s+", "", s)  # drop leading articles
    return s

def normalize_list(s: str) -> str:
    """
    For list-y answers: keep commas so we can split reliably.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)
    s = s.translate(_PUNCT_NO_COMMA)      # keep commas
    s = " ".join(s.split())
    s = s.replace(" and ", ",")           # unify separators
    return s

def tokenize_items_list(s: str) -> List[str]:
    s = normalize_list(s)
    return [p.strip() for p in s.split(",") if p.strip()]

def strict_em(pred: str, gold: str) -> bool:
    return normalize_strict(pred) == normalize_strict(gold)

def list_set_match(pred: str, gold: str) -> bool:
    g = (gold or "").lower()
    if ("," in g) or (" and " in g):
        p_set = set(tokenize_items_list(pred))
        g_set = set(tokenize_items_list(gold))
        return (len(g_set) >= 2) and (p_set == g_set)
    return False

def soft_contains(pred: str, gold: str) -> bool:
    p = normalize_strict(pred)
    g = normalize_strict(gold)
    if not p or not g:
        return False
    return re.search(rf"\b{re.escape(g)}\b", p) is not None


def pick_pred_field(ex: Dict[str, Any]) -> str:
    for k in ["prediction", "pred", "output", "model_output", "answer"]:
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    return ""

def pick_prompt_field(ex: Dict[str, Any]) -> str:
    for k in ["input", "prompt", "query"]:
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    return ""

def pick_golds(ex: Dict[str, Any]) -> List[str]:
    golds = ex.get("outputs", [])
    if isinstance(golds, list) and all(isinstance(x, str) for x in golds):
        return golds
    golds2 = ex.get("output", [])
    if isinstance(golds2, list) and all(isinstance(x, str) for x in golds2):
        return golds2
    return []


# ---------------------------
# Judge prompt + parsing
# ---------------------------
def make_judge_prompt(context_prompt: str, golds: List[str], pred: str) -> str:
    gold_text = "\n".join([f"- {g}" for g in golds[:5]])
    return (
        "You are a strict evaluator. Decide whether the model answer is correct.\n"
        "Rules:\n"
        "1) Accept paraphrases and extra correct details.\n"
        "2) Reject answers that add incorrect specifics (e.g., wrong country/title) even if partly correct.\n"
        "3) If the answer is ambiguous or not supported, reject.\n"
        "Return ONLY a single character: 1 (correct) or 0 (incorrect).\n\n"
        f"ORIGINAL PROMPT (for context):\n{context_prompt}\n\n"
        f"GOLD ANSWERS:\n{gold_text}\n\n"
        f"MODEL ANSWER:\n{pred}\n\n"
        "OUTPUT (1 or 0):"
    )

def parse_judge_output(text: str) -> Optional[bool]:
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"[01]", t)
    if not m:
        return None
    return (m.group(0) == "1")


# ---------------------------
# Judge runner (SDPA default)
# ---------------------------
class SdpaJudgeRunner:
    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        self.model.eval()

    @torch.no_grad()
    def generate_one(self, prompt: str, max_new_tokens: int = 4) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.splitlines()[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True)
    ap.add_argument("--out_path", default="")
    ap.add_argument("--summary_path", default="")
    ap.add_argument("--log_every", type=int, default=10, help="Print progress every N examples")

    ap.add_argument("--use_judge", action="store_true")
    ap.add_argument("--judge_on_all_non_em", action="store_true")
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--judge_max_new_tokens", type=int, default=4)
    ap.add_argument("--judge_use_entropy_attn", action="store_true")
    args = ap.parse_args()

    pred_path = args.pred_path
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    summary_path = args.summary_path or os.path.join(os.path.dirname(pred_path), "score_summary.json")

    judge = None
    if args.use_judge:
        if args.judge_use_entropy_attn:
            from entropy_attn_llama import EntropyAttnLlama
            judge = EntropyAttnLlama(args.judge_model)
        else:
            judge = SdpaJudgeRunner(args.judge_model)

    total = 0
    em_strict_ok = 0
    em_soft_ok = 0
    final_ok = 0
    judge_calls = 0
    judge_parse_fail = 0

    out_f = None
    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        out_f = open(args.out_path, "w", encoding="utf-8")

    try:
        for ex in iter_jsonl(pred_path):
            pred = pick_pred_field(ex)
            prompt = pick_prompt_field(ex)
            golds = pick_golds(ex)

            answer_prefix = ex.get("answer_prefix", "")
            if answer_prefix and prompt and not prompt.endswith(answer_prefix):
                prompt_for_judge = prompt + answer_prefix
            else:
                prompt_for_judge = prompt

            em = any(strict_em(pred, g) for g in golds)
            lst = any(list_set_match(pred, g) for g in golds)
            cnt = any(soft_contains(pred, g) for g in golds)
            soft = em or lst or cnt

            if "_needs_judge" in ex and isinstance(ex["_needs_judge"], bool):
                needs_judge = ex["_needs_judge"]
            else:
                if args.use_judge:
                    needs_judge = (not em) if args.judge_on_all_non_em else (soft and not em)
                else:
                    needs_judge = False

            judge_ok = None
            if args.use_judge and needs_judge:
                if judge is None:
                    raise RuntimeError("use_judge=True but judge runner not initialized.")
                judge_calls += 1
                jprompt = make_judge_prompt(prompt_for_judge, golds, pred)
                jout = judge.generate_one(jprompt, max_new_tokens=args.judge_max_new_tokens)
                judge_ok = parse_judge_output(jout)
                if judge_ok is None:
                    judge_parse_fail += 1

            if em:
                ok = True
            elif (args.use_judge and needs_judge):
                ok = (judge_ok if judge_ok is not None else soft)
            else:
                ok = soft

            total += 1
            em_strict_ok += int(em)
            em_soft_ok += int(soft)
            final_ok += int(ok)

            if args.log_every > 0 and (total % args.log_every == 0):
                print(
                    f"[score] {total} done | "
                    f"strict={fmt_ratio(em_strict_ok, total)} | "
                    f"soft={fmt_ratio(em_soft_ok, total)} | "
                    f"final={fmt_ratio(final_ok, total)} | "
                    f"judge_calls={judge_calls}"
                )

            if out_f is not None:
                ex2 = dict(ex)
                ex2["_em_strict_ok"] = em
                ex2["_em_soft_ok"] = soft
                ex2["_needs_judge"] = needs_judge
                ex2["_judge_ok"] = judge_ok
                ex2["_final_ok"] = ok
                out_f.write(json.dumps(ex2, ensure_ascii=False) + "\n")
                out_f.flush()

        summary = {
            "pred_path": pred_path,
            "out_path": args.out_path or None,
            "n": total,
            "em_strict_acc": em_strict_ok / max(total, 1),
            "em_soft_acc": em_soft_ok / max(total, 1),
            "final_acc": final_ok / max(total, 1),
            "judge_calls": judge_calls,
            "judge_parse_fail": judge_parse_fail,
            "judge_model": args.judge_model if args.use_judge else None,
            "judge_use_entropy_attn": bool(args.judge_use_entropy_attn) if args.use_judge else None,
            "judge_on_all_non_em": bool(args.judge_on_all_non_em) if args.use_judge else None,
        }

        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))

    finally:
        if out_f is not None:
            out_f.close()

if __name__ == "__main__":
    main()
