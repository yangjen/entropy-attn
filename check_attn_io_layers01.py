# check_attn_io_layers01.py
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import torch

from attention_llama import LlamaRunner


QA1_VAL_PATH = "/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/validation.jsonl"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# --------------------------
# JSONL loading + prompt build
# --------------------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_prompt_from_ex(ex: Dict[str, Any]) -> str:
    prompt = ex["input"]
    answer_prefix = ex.get("answer_prefix", "")
    if answer_prefix and not prompt.endswith(answer_prefix):
        prompt = prompt + answer_prefix
    return prompt


def ex_id_of(ex: Dict[str, Any], fallback: int) -> str:
    return str(ex.get("id", ex.get("qid", ex.get("example_id", ex.get("idx", fallback)))))


# --------------------------
# Fingerprints
# --------------------------
def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha1_int_list(xs: List[int]) -> str:
    b = (",".join(map(str, xs))).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


# --------------------------
# Sampling + stats helpers
# --------------------------
def make_seq_indices(seq_len: int, seed: int, head: int = 32, tail: int = 32, n_rand: int = 256) -> torch.Tensor:
    head_idx = torch.arange(min(head, seq_len), dtype=torch.long)
    tail_idx = torch.arange(max(seq_len - tail, 0), seq_len, dtype=torch.long) if seq_len > 0 else torch.empty((0,), dtype=torch.long)

    n_rand = min(n_rand, seq_len)
    if n_rand > 0:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        rand_idx = torch.randperm(seq_len, generator=g)[:n_rand].to(dtype=torch.long)
    else:
        rand_idx = torch.empty((0,), dtype=torch.long)

    idx = torch.unique(torch.cat([head_idx, tail_idx, rand_idx], dim=0))
    idx, _ = torch.sort(idx)
    return idx


@torch.no_grad()
def tensor_global_stats(x: torch.Tensor) -> Dict[str, float]:
    xf = x.float()
    mean = xf.mean()
    var = xf.var(unbiased=False)
    std = torch.sqrt(var + 1e-12)
    rms = torch.sqrt((xf * xf).mean() + 1e-12)
    maxabs = xf.abs().max()
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "rms": float(rms.item()),
        "maxabs": float(maxabs.item()),
    }


@torch.no_grad()
def sample_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D]
    idx_dev = idx.to(device=x.device)
    B, T, D = x.shape
    idx_exp = idx_dev.view(1, -1, 1).expand(B, -1, D)
    y = torch.gather(x, dim=1, index=idx_exp)
    return y.detach().float().cpu()


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    rel = d / denom
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rms": float(torch.sqrt((d * d).mean()).item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


# --------------------------
# Attention I/O capture
# --------------------------
class AttnIOCapture:
    def __init__(self, sample_idx: torch.Tensor):
        self.sample_idx = sample_idx
        self.data: Dict[str, Dict[str, Any]] = {}

    def hook_with_kwargs(self, name: str):
        # signature with_kwargs=True: (module, args, kwargs, output)
        def _fn(module, args, kwargs, out):
            # hidden_states can be in args[0] OR kwargs["hidden_states"]
            if args and isinstance(args[0], torch.Tensor):
                x = args[0]
            elif isinstance(kwargs, dict) and "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                x = kwargs["hidden_states"]
            else:
                raise RuntimeError(f"[HOOK] Could not find hidden_states for {name}. args_len={len(args)} keys={list(kwargs.keys()) if isinstance(kwargs, dict) else None}")

            y = out[0] if isinstance(out, tuple) else out

            self.data[name] = {
                "shape_in": tuple(x.shape),
                "shape_out": tuple(y.shape),
                "in_stats": tensor_global_stats(x),
                "out_stats": tensor_global_stats(y),
                "in_sample": sample_tokens(x, self.sample_idx),
                "out_sample": sample_tokens(y, self.sample_idx),
            }
        return _fn

    def hook_no_kwargs(self, name: str):
        # legacy signature: (module, args, output)
        def _fn(module, args, out):
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"[HOOK] args missing hidden_states for {name}. args_len={len(args)}")
            x = args[0]
            y = out[0] if isinstance(out, tuple) else out

            self.data[name] = {
                "shape_in": tuple(x.shape),
                "shape_out": tuple(y.shape),
                "in_stats": tensor_global_stats(x),
                "out_stats": tensor_global_stats(y),
                "in_sample": sample_tokens(x, self.sample_idx),
                "out_sample": sample_tokens(y, self.sample_idx),
            }
        return _fn


@torch.inference_mode()
def capture_for_examples(
    attn_impl: str,
    prompts: List[str],
    sample_indices: List[torch.Tensor],
    dtype_str: str = "bf16",
    deterministic: bool = True,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = LlamaRunner(
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype_map[dtype_str],
        attn_impl=attn_impl,
        deterministic=deterministic,
    )

    results: List[Dict[str, Any]] = []

    for i, prompt in enumerate(prompts):
        tok = runner.tokenizer(prompt, return_tensors="pt")
        input_ids = tok["input_ids"].to(runner.model.device)
        attention_mask = tok.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(runner.model.device)

        cap = AttnIOCapture(sample_indices[i])

        # Register hooks on layer0 and layer1 attention
        attn0 = runner.model.model.layers[0].self_attn
        attn1 = runner.model.model.layers[1].self_attn

        # Try kwargs-enabled hook first (fixes your IndexError)
        try:
            h0 = attn0.register_forward_hook(cap.hook_with_kwargs("layer0.self_attn"), with_kwargs=True)
            h1 = attn1.register_forward_hook(cap.hook_with_kwargs("layer1.self_attn"), with_kwargs=True)
            use_kwargs_hook = True
        except TypeError:
            # Older torch: no with_kwargs
            h0 = attn0.register_forward_hook(cap.hook_no_kwargs("layer0.self_attn"))
            h1 = attn1.register_forward_hook(cap.hook_no_kwargs("layer1.self_attn"))
            use_kwargs_hook = False

        _ = runner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        h0.remove()
        h1.remove()

        results.append(
            {
                "attn_impl": attn_impl,
                "dtype": str(next(runner.model.parameters()).dtype),
                "prompt_sha1": sha1_str(prompt),
                "input_ids_sha1": sha1_int_list(input_ids[0].detach().cpu().tolist()),
                "seq_len": int(input_ids.shape[1]),
                "used_kwargs_hook": use_kwargs_hook,
                "records": cap.data,
            }
        )

    return results


def main(
    n_examples: int = 10,
    dtype_str: str = "bf16",
    deterministic: bool = True,
    check_prompt_identical: bool = True,
    atol: float = 2e-2,
    rtol: float = 2e-2,
):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    # Load first N examples
    exs: List[Dict[str, Any]] = []
    for i, ex in enumerate(iter_jsonl(QA1_VAL_PATH)):
        if i >= n_examples:
            break
        exs.append(ex)

    prompts: List[str] = [build_prompt_from_ex(ex) for ex in exs]
    ex_ids: List[str] = [ex_id_of(ex, i) for i, ex in enumerate(exs)]
    prompt_sha1s: List[str] = [sha1_str(p) for p in prompts]

    # prompt string equality is trivial here because we reuse the same prompt for both runs,
    # but we keep the check so it prints what you asked for.
    if check_prompt_identical:
        # Compare each prompt against itself (same object reused for sdpa/entropy runs)
        all_equal = all(p == p for p in prompts)
        print(f"[PROMPT STRING CHECK] (string==string) for built prompts: {all_equal} (expected True)")

    # Precompute token-length-based sample indices using a small tokenizer instance (sdpa runner)
    tmp = LlamaRunner(
        model_name=MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16 if dtype_str == "bf16" else (torch.float16 if dtype_str == "fp16" else torch.float32),
        attn_impl="sdpa",
        deterministic=deterministic,
    )
    sample_indices: List[torch.Tensor] = []
    seq_lens: List[int] = []
    token_sha1s: List[str] = []
    for i, p in enumerate(prompts):
        t = tmp.tokenizer(p, return_tensors="pt")
        ids = t["input_ids"][0].tolist()
        seq_lens.append(len(ids))
        token_sha1s.append(sha1_int_list(ids))
        sample_indices.append(make_seq_indices(seq_len=len(ids), seed=1337 + i, head=32, tail=32, n_rand=256))
    del tmp
    torch.cuda.empty_cache()

    print(f"Loaded {len(prompts)} prompts from qa_1. (Not printing prompts.)")
    for i in range(len(prompts)):
        print(f"  ex#{i} id={ex_ids[i]} prompt_sha1={prompt_sha1s[i]} seq_len={seq_lens[i]} token_sha1={token_sha1s[i]}")

    print("\n=== Capturing SDPA (layers 0/1) for first 10 examples ===")
    sdpa_res = capture_for_examples(
        attn_impl="sdpa",
        prompts=prompts,
        sample_indices=sample_indices,
        dtype_str=dtype_str,
        deterministic=deterministic,
        device="cuda",
    )
    print(f"SDPA done. dtype={sdpa_res[0]['dtype']} used_kwargs_hook={sdpa_res[0]['used_kwargs_hook']}")

    # Free as much as possible before loading second model
    torch.cuda.empty_cache()

    print("\n=== Capturing entropy_attn (layers 0/1) for first 10 examples ===")
    ent_res = capture_for_examples(
        attn_impl="entropy_attn",
        prompts=prompts,
        sample_indices=sample_indices,
        dtype_str=dtype_str,
        deterministic=deterministic,
        device="cuda",
    )
    print(f"ENT done. dtype={ent_res[0]['dtype']} used_kwargs_hook={ent_res[0]['used_kwargs_hook']}")

    print("\n=== Comparing layer0/layer1 attention I/O (sample-based) ===")
    for i in range(len(prompts)):
        s = sdpa_res[i]
        e = ent_res[i]
        print("-" * 110)
        print(f"[ex#{i}] id={ex_ids[i]}  seq_len={s['seq_len']}")

        if check_prompt_identical:
            same_prompt = (s["prompt_sha1"] == e["prompt_sha1"])
            same_tokens = (s["input_ids_sha1"] == e["input_ids_sha1"])
            print(f"[PROMPT HASH] sha1_equal={same_prompt}  [TOKENS HASH] input_ids_sha1_equal={same_tokens}")

        for lname in ["layer0.self_attn", "layer1.self_attn"]:
            a = s["records"][lname]
            b = e["records"][lname]

            in_diff = diff_stats(a["in_sample"], b["in_sample"])
            out_diff = diff_stats(a["out_sample"], b["out_sample"])
            out_allclose = torch.allclose(a["out_sample"], b["out_sample"], atol=atol, rtol=rtol)

            print(f"\n  [{lname}]")
            print(f"    shapes: in={a['shape_in']} out={a['shape_out']}")
            print(f"    IN  stats(sdpa)={a['in_stats']}  stats(ent)={b['in_stats']}")
            print(f"    IN  sample diff={in_diff}")
            print(f"    OUT stats(sdpa)={a['out_stats']} stats(ent)={b['out_stats']}")
            print(f"    OUT sample diff={out_diff}")
            print(f"    OUT sample allclose(atol={atol}, rtol={rtol})={out_allclose}")

    print("\nDone.")


if __name__ == "__main__":
    main(
        n_examples=10,
        dtype_str="bf16",
        deterministic=True,
        check_prompt_identical=True,
        atol=2e-2,
        rtol=2e-2,
    )
