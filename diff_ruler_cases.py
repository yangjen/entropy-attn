import json

SDPA_PATH = "RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/qa_1_sdpa_predictions_timed.jsonl"
ENT_PATH  = "RULER_outputs/llama3.1-8b-chat/synthetic/32768/data/qa_1/qa_1_entropy_attn_predictions_timed.jsonl"


def ruler_hit(pred, refs):
    p = (pred or "").lower()
    return any((r or "").lower() in p for r in refs)

def load_list(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

sdpa = load_list(SDPA_PATH)
ent  = load_list(ENT_PATH)

if len(sdpa) != len(ent):
    raise ValueError(f"Length mismatch: sdpa={len(sdpa)} ent={len(ent)}")

diff = []
for i, (s, e) in enumerate(zip(sdpa, ent)):
    refs = s.get("outputs", [])
    sp = s.get("prediction", "")
    ep = e.get("prediction", "")

    hs = ruler_hit(sp, refs)
    he = ruler_hit(ep, refs)

    if hs != he:
        diff.append((i, refs, hs, sp, he, ep))

print(f"\nTotal differing cases: {len(diff)} out of {len(sdpa)}\n")

N = 40
for j, (i, refs, hs, sp, he, ep) in enumerate(diff[:N]):
    print("=" * 80)
    print(f"[{j}] line={i+1}")
    print("GOLD:", refs)
    print("SDPA HIT:", hs)
    print("SDPA:", sp)
    print("ENT  HIT:", he)
    print("ENT :", ep)

