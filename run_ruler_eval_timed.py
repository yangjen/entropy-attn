'''
CUDA_VISIBLE_DEVICES=1 python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data \
  --tasks qa_1 \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl flash_attention_2 \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4


  CUDA_VISIBLE_DEVICES=4 python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data \
  --tasks qa_2 \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4


'''


import os
import json
import re
import string
import argparse
from typing import Dict, Any, List, Optional

import torch
from attention_llama import LlamaRunner


def fmt_ratio(k: int, n: int) -> str:
    return f"{k}/{n} ({(k / max(n, 1)) * 100:.1f}%)"

def _summarize_times(times: List[float]) -> Dict[str, float]:
    """Summarize a list of durations in seconds."""
    if not times:
        return {"n": 0}
    ts = sorted(times)
    n = len(ts)
    def pct(p: float) -> float:
        if n == 1:
            return ts[0]
        k = int(round((p/100.0) * (n - 1)))
        k = max(0, min(n - 1, k))
        return ts[k]
    return {
        "n": n,
        "mean_s": sum(ts) / n,
        "median_s": pct(50),
        "p90_s": pct(90),
        "p95_s": pct(95),
        "min_s": ts[0],
        "max_s": ts[-1],
    }

def _cuda_time_call(fn, enabled: bool, state: Dict[str, int], skip: int, times: List[float]):
    """Time a single CUDA call with cuda.Events. Skips the first `skip` calls globally per-state."""
    if not enabled or (not torch.cuda.is_available()):
        return fn()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    out = fn()
    end_event.record()
    torch.cuda.synchronize()
    state["n"] = state.get("n", 0) + 1
    if state["n"] > skip:
        times.append(start_event.elapsed_time(end_event) / 1000.0)  # seconds
    return out


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
    s = re.sub(r"<\|.*?\|>", " ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s


def normalize_list(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)
    s = s.translate(_PUNCT_NO_COMMA)
    s = " ".join(s.split())
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
        # Only apply if gold is actually a list-like answer
        return (len(g_set) >= 2) and (p_set == g_set)
    return False


def soft_contains(pred: str, gold: str) -> bool:
    p = normalize_strict(pred)
    g = normalize_strict(gold)
    if not p or not g:
        return False
    return re.search(rf"\b{re.escape(g)}\b", p) is not None


def string_match_part(preds: List[str], refs: List[List[str]]) -> float:
    score = sum(
        [
            max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
            for pred, ref in zip(preds, refs)
        ]
    ) / max(len(preds), 1) * 100
    return round(score, 2)


def ruler_hit(pred: str, ref_list: List[str]) -> int:
    pred_l = (pred or "").lower()
    for r in ref_list:
        if (r or "").lower() in pred_l:
            return 1
    return 0


def make_judge_prompt(question_block: str, golds: List[str], pred: str) -> str:
    gold_text = "\n".join([f"- {g}" for g in golds[:5]])
    return (
        "You are a strict evaluator. Decide whether the model answer is correct.\n"
        "Rules:\n"
        "1) Accept paraphrases and extra correct details.\n"
        "2) Reject answers that add incorrect specifics even if partly correct.\n"
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
        q = prompt[idx + len("Question:") :].strip()

    q = q.splitlines()[0].strip()
    q = re.sub(r"<\|.*?\|>", "", q).strip()
    q = re.sub(r"\bassistant\b\s*$", "", q).strip()
    return q[:300]


def compact_row_custom(
    ex: Dict[str, Any],
    task: str,
    prompt: str,
    golds: List[str],
    pred: str,
    em: bool,
    soft: bool,
    needs_judge: bool,
    judge_ok: Optional[bool],
    final_ok: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
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


def compact_row_ruler(
    ex: Dict[str, Any],
    task: str,
    prompt: str,
    golds: List[str],
    pred: str,
    hit: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task": task,
        "question": extract_question(prompt),
        "outputs": golds,
        "prediction": pred,
        "_ruler_part_hit": bool(hit),
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
    ap.add_argument("--pred_name", default="predictions_timed.jsonl")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)

    # Inline judge controls
    ap.add_argument("--use_judge", action="store_true")
    ap.add_argument("--judge_max_new_tokens", type=int, default=4)
    ap.add_argument("--judge_on_all_non_em", action="store_true")

    ap.add_argument(
        "--eval_mode",
        default="custom",
        choices=["custom", "ruler_part"],
        help="custom: strict/soft EM + optional LLM judge; ruler_part: RULER string_match_part only",
    )

    ap.add_argument(
        "--attn_impl",
        default="entropy_attn",
        choices=["entropy_attn", "sdpa", "flash_attention_2", "eager"],
    )
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--time", action="store_true", help="Measure CUDA time for generation calls (pred and judge)")
    ap.add_argument("--time_skip", type=int, default=4, help="Skip first N timed calls per stream (warmup/autotune)")

    args = ap.parse_args()

    if args.eval_mode == "ruler_part":
        # Force disable judge to avoid accidental jprompt paths
        args.use_judge = False

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = LlamaRunner(
        args.model,
        attn_impl=args.attn_impl,
        dtype=dtype_map[args.dtype],
        deterministic=args.deterministic,
    )

    all_tasks = sorted(
        [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    )
    tasks = (
        all_tasks
        if args.tasks == "ALL"
        else [t.strip() for t in args.tasks.split(",") if t.strip()]
    )

    summary: Dict[str, Any] = {}
    # Optional CUDA timing
    pred_times_s: List[float] = []
    judge_times_s: List[float] = []
    pred_time_state: Dict[str, int] = {'n': 0}
    judge_time_state: Dict[str, int] = {'n': 0}


    for task in tasks:
        task_dir = os.path.join(args.data_root, task)
        val_path = os.path.join(task_dir, "validation.jsonl")
        if not os.path.exists(val_path):
            print(f"[skip] {task}: missing {val_path}")
            continue

        # Unique per attn_impl (and optionally dtype) to avoid collisions.
        pred_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_{args.pred_name}")
        # pred_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_{args.dtype}_{args.pred_name}")

        total = 0
        em_strict_ok = 0
        em_soft_ok = 0
        final_ok = 0
        judge_calls = 0
        judge_parse_fail = 0

        ruler_hits = 0

        with open(pred_path, "w", encoding="utf-8") as out_f:
            for ex in iter_jsonl(val_path):
                prompt = ex["input"]
                answer_prefix = ex.get("answer_prefix", "")
                if answer_prefix and not prompt.endswith(answer_prefix):
                    prompt = prompt + answer_prefix

                pred = _cuda_time_call(
                    lambda: runner.generate_one(
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_newline=(not args.multiline),
                    ),
                    enabled=args.time, state=pred_time_state, skip=args.time_skip, times=pred_times_s,
                )
                golds = ex.get("outputs", [])

                total += 1

                # ---------------- ruler_part ----------------
                if args.eval_mode == "ruler_part":
                    hit = ruler_hit(pred, list(golds))
                    ruler_hits += hit

                    if args.log_every > 0 and (total % args.log_every == 0):
                        running = round(ruler_hits / max(total, 1) * 100, 2)
                        print(
                            f"[{task}] {total} done | "
                            f"ruler_part={running:.2f}% ({ruler_hits}/{total})"
                        )

                    if args.compact:
                        ex_out = compact_row_ruler(
                            ex=ex,
                            task=task,
                            prompt=prompt,
                            golds=list(golds),
                            pred=pred,
                            hit=hit,
                        )
                    else:
                        ex_out = dict(ex)
                        ex_out["prediction"] = pred
                        ex_out["_ruler_part_hit"] = bool(hit)

                    out_f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                    out_f.flush()
                    continue

                # ---------------- custom eval ----------------
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

                judge_ok: Optional[bool] = None
                if needs_judge:
                    judge_calls += 1
                    jprompt = make_judge_prompt(question_block=prompt, golds=list(golds), pred=pred)
                    jout = _cuda_time_call(
                        lambda: runner.generate_one(
                            jprompt,
                            max_new_tokens=args.judge_max_new_tokens,
                            stop_on_newline=True,
                        ),
                        enabled=args.time, state=judge_time_state, skip=args.time_skip, times=judge_times_s,
                    )
                    judge_ok = parse_judge_output(jout)
                    if judge_ok is None:
                        judge_parse_fail += 1

                if em:
                    ok = True
                elif needs_judge:
                    ok = (judge_ok if judge_ok is not None else soft)
                else:
                    ok = soft

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
                    ex_out = compact_row_custom(
                        ex=ex,
                        task=task,
                        prompt=prompt,
                        golds=list(golds),
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

        if args.eval_mode == "ruler_part":
            ruler_score = round(ruler_hits / max(total, 1) * 100, 2)
            summary[task] = {
                "metric": "ruler_string_match_part",
                "score": ruler_score,
                "n": total,
                "hits": ruler_hits,
                "pred_path": pred_path,
            }
            print(f"[done] {task}: RULER string_match_part = {ruler_score:.2f} -> {pred_path}")
        else:
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

    def _safe_name(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)

    # Timing report (CUDA events) for generate calls
    if args.time and torch.cuda.is_available():
        t_pred = _summarize_times(pred_times_s)
        t_judge = _summarize_times(judge_times_s)
        print("\n[TIMING] pred_generate:", t_pred)
        print("[TIMING] judge_generate:", t_judge)
        summary["_timing"] = {
            "pred_generate": t_pred,
            "judge_generate": t_judge,
            "time_skip": args.time_skip,
            "attn_impl": args.attn_impl,
            "dtype": args.dtype,
        }

    tasks_tag = _safe_name(args.tasks)
    summ_path = os.path.join(args.data_root, f"summary_{tasks_tag}_{args.attn_impl}_timed.json")
    # summ_path = os.path.join(args.data_root, f"summary_{tasks_tag}_{args.attn_impl}_{args.dtype}.json")

    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {summ_path}")


if __name__ == "__main__":
    main()
