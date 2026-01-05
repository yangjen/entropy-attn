# run_ruler_eval.py
"""
CUDA_VISIBLE_DEVICES=2 python run_ruler_eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data \
  --tasks qa_1 \
  --max_new_tokens 64 \
  --pred_name predictions_entropy_attn_judged.jsonl \
  --compact \
  --task_name qa_2 \
  --use_judge \
  --judge_on_all_non_em
  --log_every 10

"""

import os
import json
import re
import string
import argparse
from typing import Dict, Any, List, Optional

from entropy_attn_llama import EntropyAttnLlama

def fmt_ratio(k: int, n: int) -> str:
    return f"{k}/{n} ({(k/max(n,1))*100:.1f}%)"

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_PUNCT_NO_COMMA = str.maketrans("", "", string.punctuation.replace(",", ""))

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def normalize_strict(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)      # remove leaked special tokens
    s = s.translate(_PUNCT_TABLE)         # remove punctuation
    s = " ".join(s.split())
    s = re.sub(r"^(the|a|an)\s+", "", s)  # drop leading articles
    return s

def normalize_list(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)
    # keep commas so we can split reliably
    s = s.translate(_PUNCT_NO_COMMA)
    s = " ".join(s.split())
    # unify separators
    s = s.replace(" and ", ",")
    return s

def tokenize_items_list(s: str) -> List[str]:
    s = normalize_list(s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

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
    # "King Charles III of West Francia" should contain "King Charles III"
    p = normalize_strict(pred)
    g = normalize_strict(gold)
    if not p or not g:
        return False
    return re.search(rf"\b{re.escape(g)}\b", p) is not None

def make_judge_prompt(question_block: str, golds: List[str], pred: str) -> str:
    gold_text = "\n".join([f"- {g}" for g in golds[:5]])  # cap
    return (
        "You are a strict evaluator. Decide whether the model answer is correct.\n"
        "Rules:\n"
        "1) Accept paraphrases and extra correct details.\n"
        "2) Reject answers that add incorrect specifics (e.g., wrong country/title) even if partly correct.\n"
        "3) If the answer is ambiguous or not supported, reject.\n"
        "Return ONLY a single character: 1 (correct) or 0 (incorrect).\n\n"
        f"ORIGINAL PROMPT (for context):\n{question_block}\n\n"
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

def extract_question(prompt: str) -> str:
    if not prompt:
        return ""
    idx = prompt.rfind("Question:")
    if idx == -1:
        q = prompt.strip()
    else:
        q = prompt[idx + len("Question:"):].strip()

    q = q.splitlines()[0].strip()
    q = re.sub(r"<\|.*?\|>", "", q).strip()
    q = re.sub(r"\bassistant\b\s*$", "", q).strip()
    return q[:300]

def compact_row(
    ex: Dict[str, Any],
    task: str,
    prompt: str,
    golds: List[str],
    pred: str,
    em: bool,
    soft: bool,
    needs_judge: bool,
    judge_ok: Optional[bool],
    final_ok: bool
) -> Dict[str, Any]:
    out = {
        "task": task,
        "question": extract_question(prompt),
        "outputs": golds,
        "prediction": pred,
        "_em_strict_ok": em,
        "_em_soft_ok": soft,
        "_needs_judge": needs_judge,
        "_judge_ok": judge_ok,
        "_final_ok": final_ok,
    }
    for k in ["id", "qid", "example_id", "idx"]:
        if k in ex:
            out[k] = ex[k]
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--multiline", action="store_true")
    ap.add_argument("--pred_name", default="predictions.jsonl")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)

    # Inline judge controls
    ap.add_argument("--use_judge", action="store_true")
    ap.add_argument("--judge_max_new_tokens", type=int, default=4)
    ap.add_argument("--judge_on_all_non_em", action="store_true")
    args = ap.parse_args()

    runner = EntropyAttnLlama(args.model)

    all_tasks = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    tasks = all_tasks if args.tasks == "ALL" else [t.strip() for t in args.tasks.split(",") if t.strip()]

    summary = {}

    for task in tasks:
        task_dir = os.path.join(args.data_root, task)
        val_path = os.path.join(task_dir, "validation.jsonl")
        if not os.path.exists(val_path):
            print(f"[skip] {task}: missing {val_path}")
            continue

        # Always write into the correct task folder.
        # pred_path = os.path.join(task_dir, args.pred_name)

        # OPTIONAL safety: prevent accidental overwrite if you reuse the same pred_name across runs.
        # Uncomment if you want each task to auto-get its own filename:
        pred_path = os.path.join(task_dir, f"{task}.{args.pred_name}")

        os.makedirs(os.path.dirname(pred_path), exist_ok=True)

        total = 0
        em_strict_ok = 0
        em_soft_ok = 0
        final_ok = 0
        judge_calls = 0
        judge_parse_fail = 0

        with open(pred_path, "w", encoding="utf-8") as out_f:
            for ex in iter_jsonl(val_path):
                prompt = ex["input"]
                answer_prefix = ex.get("answer_prefix", "")
                if answer_prefix and not prompt.endswith(answer_prefix):
                    prompt = prompt + answer_prefix

                pred = runner.generate_one(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    stop_on_newline=(not args.multiline),
                )

                golds = ex.get("outputs", [])

                em = any(strict_em(pred, g) for g in golds)
                lst = any(list_set_match(pred, g) for g in golds)
                cnt = any(soft_contains(pred, g) for g in golds)
                soft = em or lst or cnt

                needs_judge = False
                if args.use_judge:
                    if args.judge_on_all_non_em:
                        needs_judge = (not em)
                    else:
                        needs_judge = (soft and not em)

                judge_ok = None
                if needs_judge:
                    judge_calls += 1
                    jprompt = make_judge_prompt(question_block=prompt, golds=golds, pred=pred)
                    jout = runner.generate_one(jprompt, max_new_tokens=args.judge_max_new_tokens, stop_on_newline=True)
                    judge_ok = parse_judge_output(jout)
                    if judge_ok is None:
                        judge_parse_fail += 1

                try:
                    jout = runner.generate_one(jprompt, max_new_tokens=args.judge_max_new_tokens, stop_on_newline=True)
                except Exception as e:
                    print("[JUDGE FAIL]", task, "ex_id=", ex.get("id", None), "err=", repr(e))
                    print("JPROMPT head:", jprompt[:400])
                    jout = ""  # fail-closed

                # Final decision logic:
                # - If strict EM true: accept.
                # - Else if judged: use judge result if parsed; otherwise fall back to soft heuristic.
                # - Else: use soft heuristic.
                if em:
                    ok = True
                elif needs_judge:
                    ok = (judge_ok if judge_ok is not None else soft)
                else:
                    ok = soft

                total += 1
                em_strict_ok += int(em)
                em_soft_ok += int(soft)
                final_ok += int(ok)

                if args.log_every > 0 and (total % args.log_every == 0):
                    print(
                        f"[{task}] {total} done | "
                        f"strict={fmt_ratio(em_strict_ok, total)} | "
                        f"soft={fmt_ratio(em_soft_ok, total)} | "
                        f"final={fmt_ratio(final_ok, total)} | "
                        f"judge_calls={judge_calls}"
                    )

                if args.compact:
                    ex_out = compact_row(
                        ex=ex,
                        task=task,          # <-- ALWAYS the folder task
                        prompt=prompt,
                        golds=golds,
                        pred=pred,
                        em=em,
                        soft=soft,
                        needs_judge=needs_judge,
                        judge_ok=judge_ok,
                        final_ok=ok,
                    )
                else:
                    ex_out = dict(ex)
                    ex_out["prediction"] = pred
                    ex_out["_em_strict_ok"] = em
                    ex_out["_em_soft_ok"] = soft
                    ex_out["_needs_judge"] = needs_judge
                    ex_out["_judge_ok"] = judge_ok
                    ex_out["_final_ok"] = ok

                out_f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                out_f.flush()

        summary[task] = {
            "n": total,
            "em_strict_acc": em_strict_ok / max(total, 1),
            "em_soft_acc": em_soft_ok / max(total, 1),
            "final_acc": final_ok / max(total, 1),
            "judge_calls": judge_calls,
            "judge_parse_fail": judge_parse_fail,
            "pred_path": pred_path,
        }

        print(
            f"[done] {task}: "
            f"strict={summary[task]['em_strict_acc']:.3f}  "
            f"soft={summary[task]['em_soft_acc']:.3f}  "
            f"final={summary[task]['final_acc']:.3f}  "
            f"judge_calls={judge_calls}  -> {pred_path}"
        )

    summ_path = os.path.join(args.data_root, "entropy_attn_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {summ_path}")

if __name__ == "__main__":
    main()

