# entropy_attn_llama.py
import torch
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from models.attn_patch import entropy_attention_forward  # jeff's entropy-attn wrapper


def _register_entropy_attn():
    """
    Register 'entropy_attn' into HF attention dispatcher so config.attn_implementation='entropy_attn'
    will route to our function.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    state = {"printed": False}

    def wrapped(*args, **kwargs):
        if not state["printed"]:
            state["printed"] = True
            print("[VERIFY] entropy_attention_forward CALLED")
        return entropy_attention_forward(*args, **kwargs)

    # HF has had different shapes for this across versions:
    # - sometimes dict-like
    # - sometimes has .register()
    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register("entropy_attn", wrapped)
    else:
        ALL_ATTENTION_FUNCTIONS["entropy_attn"] = wrapped

    # Print what got registered
    try:
        fn = ALL_ATTENTION_FUNCTIONS.get("entropy_attn", None)
    except Exception:
        fn = ALL_ATTENTION_FUNCTIONS["entropy_attn"]
    print("[VERIFY] Registered ALL_ATTENTION_FUNCTIONS['entropy_attn'] ->", fn)

    return wrapped


@dataclass
class EntropyAttnLlama:
    model_name: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    attn_impl: str = "entropy_attn"  # <-- IMPORTANT

    def __post_init__(self):
        # 0) Register BEFORE model is instantiated
        _register_entropy_attn()

        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) Config: force our implementation
        config = AutoConfig.from_pretrained(self.model_name)
        config.attn_implementation = self.attn_impl
        # keep this for older/newer compatibility
        setattr(config, "_attn_implementation", self.attn_impl)

        # 3) Load model on a single GPU (predictable)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            dtype=self.dtype,       
            device_map=None,
        ).to(self.device)

        self.model.eval()

        print(
            f"[EntropyAttnLlama] attn_impl="
            f"{getattr(self.model.config, '_attn_implementation', None)} / "
            f"{getattr(self.model.config, 'attn_implementation', None)}"
        )

    @torch.inference_mode()
    def generate_one(self, prompt: str, max_new_tokens: int = 64, stop_on_newline: bool = True) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # if stop_on_newline:
        #     text = text.splitlines()[0]
        # return text.strip()

        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        if gen_ids.numel() == 0:
            return ""
        
        text = text.strip()
        if stop_on_newline:
            lines = text.splitlines()
            text = (lines[0].strip() if lines else "")
        return text




