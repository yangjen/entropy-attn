#attention_llama.py
import torch
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from models.attn_patch import entropy_attention_forward  # only needed for entropy_attn

def _register_entropy_attn():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    state = {"printed": False}

    def wrapped(*args, **kwargs):
        if not state["printed"]:
            state["printed"] = True
            print("[VERIFY] entropy_attention_forward CALLED")
        return entropy_attention_forward(*args, **kwargs)

    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register("entropy_attn", wrapped)
    else:
        ALL_ATTENTION_FUNCTIONS["entropy_attn"] = wrapped

    print("[VERIFY] Registered entropy_attn attention impl")
    return wrapped

_STR2DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

@dataclass
class LlamaRunner:
    model_name: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    attn_impl: str = "sdpa"  # "sdpa", "flash_attention_2", "eager", "entropy_attn"
    deterministic: bool = True

    def __post_init__(self):
        if self.deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if self.attn_impl == "entropy_attn":
            _register_entropy_attn()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.model_name)
        config.attn_implementation = self.attn_impl
        setattr(config, "_attn_implementation", self.attn_impl)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=self.dtype,   # IMPORTANT: use torch_dtype
            device_map=None,
        ).to(self.device)

        self.model.eval()
        print(
            f"[LlamaRunner] attn_impl="
            f"{getattr(self.model.config, '_attn_implementation', None)} / "
            f"{getattr(self.model.config, 'attn_implementation', None)}  "
            f"dtype={next(self.model.parameters()).dtype}"
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
        if gen_ids.numel() == 0:
            return ""
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if stop_on_newline:
            text = (text.splitlines()[0].strip() if text else "")
        return text
