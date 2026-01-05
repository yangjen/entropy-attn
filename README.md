## To run tests on the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_kernel.py`

## To run tests on the end to end model with the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl flash_attention_2`
`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl entropy_attn`

## Baseline (SDPA)
Baseline predicions were generated with the RULER's run.sh script (will need to clone the RULER repo - https://github.com/NVIDIA/RULER.git)
Results were then evaluated with score_ruler_preds.py

## To run on RULER with entropy_attn kernel
`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python run_ruler_eval.py   --model meta-llama/Llama-3.1-8B-Instruct   --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data   --tasks qa_2   --max_new_tokens 64   --compact   --use_judge   --judge_on_all_non_em   --log_every 10`
