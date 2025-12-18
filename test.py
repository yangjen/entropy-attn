import torch
import numpy as np
from entropy_attn_triton import attention

def test(Z=1, H=8, N_CTX=512, HEAD_DIM=128):
    device = "cuda:0"
    dtype = torch.float16

    q = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    k = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    v = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    scale = 1 / np.sqrt(HEAD_DIM)

    qk = torch.einsum("bhqd,bhkd->bhqk", q * scale, k)
    mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), 1)
    qk = qk - (mask * 1e8).half()
    A = qk.softmax(dim=-1)

    logA = torch.log_softmax(qk, dim=-1).float()
    entropy = -(A * logA).masked_fill(mask, 0.0).sum(dim=-1)

    out = torch.einsum("bhqk,bhkd->bhqd", A, v)

    triton_out, triton_entropy = attention(q, k, v, True, scale)

    diff = (out - triton_out).abs()
    print(f"{diff.amax()=}")

    print(f"{entropy=}")
    print(f"{triton_entropy=}")

    entropy_diff = (entropy - triton_entropy).abs()
    print(f"{entropy_diff.amax()=} {entropy_diff.mean()=}")

if __name__ == "__main__":
    test()
