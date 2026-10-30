"""Check that the HF-format model.safetensors holds the same tensors as the official sam3.pt.

    python -I scripts/check_hf_weights.py models/sam3/sam3.pt models/sam3/model.safetensors

sam3rs loads model.safetensors directly; unmatched entries should only be computed RoPE
buffers and the detector point projections and text projection, which the image path
never uses
"""
import hashlib, sys, torch
from safetensors.torch import load_file
hf = load_file(sys.argv[2])
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
ck = ck.get("model", ck)
def h(t): return hashlib.sha1(t.float().contiguous().numpy().tobytes()).hexdigest()
hf_hash = {}
for k, v in hf.items(): hf_hash.setdefault(h(v), []).append(k)
missing, matched = [], 0
for k, v in ck.items():
    if not (k.startswith("detector.") or k.startswith("tracker.")): continue
    if any(s in k for s in ["maskmem", "memory", "obj_ptr", "no_obj", "mask_downsample", "tpos", "occlusion", "no_mem_pos"]): continue
    cands = [v]
    if v.ndim >= 1 and v.shape[0] % 3 == 0 and ("qkv" in k or "in_proj" in k):
        cands = list(v.chunk(3, 0))
    if k.endswith("pos_embed") and v.ndim == 3 and v.shape[1] == 577:
        cands = [v[:, 1:]]
    ok = all(h(c) in hf_hash for c in cands)
    if ok: matched += 1
    else: missing.append((k, tuple(v.shape)))
print("matched", matched, "unmatched", len(missing))
for m in missing[:60]: print("  ", m)
