"""Run the official SAM 3 PyTorch model as the parity reference and the PyTorch baseline.

    python -I scripts/reference.py dump --ckpt models/sam3/sam3.pt --out models/parity
    python -I scripts/reference.py bench --ckpt models/sam3/sam3.pt --precision bf16

`dump` writes one safetensors file per image with the preprocessed input, the image
features, and every prompt's outputs, computed in true FP32 (TF32 off). `bench` times the
osam benchmark protocol (dogs.jpg, median of 5 after one warm-up, model load excluded).

Upstream runs the ViT MLP through `addmm_act`, which casts fc1 to BF16 and only works
under BF16 autocast. FP32 mode swaps it for an FP32 linear with the same tanh GELU that
`_addmm_activation` applies on CUDA, so the reference differs from upstream only in dtype
"""

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from safetensors.torch import save_file

import sam3.model.vitdet as vitdet
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

ROOT = pathlib.Path(__file__).resolve().parent.parent

# osam's benchmark prompts for dogs.jpg, in original pixel coordinates
BENCH_PROMPTS = [
    {"name": "point", "points": [[1439, 504, 1]]},
    {"name": "box", "boxes": [[1233, 376, 1649, 691, 1]]},
    {
        "name": "negative box",
        "boxes": [[1233, 376, 1649, 691, 1], [1400, 500, 1500, 600, 0]],
    },
    {"name": "text", "text": "dog"},
]


def _fp32_addmm_act(activation, linear, mat1):
    y = F.linear(mat1, linear.weight, linear.bias)
    if activation in [F.relu, torch.nn.ReLU]:
        return F.relu(y)
    return F.gelu(y, approximate="tanh")


def build(ckpt: str, precision: str) -> Sam3Processor:
    model = build_sam3_image_model(
        checkpoint_path=ckpt, load_from_HF=False, enable_inst_interactivity=True
    )
    # the tracker enters BF16 autocast for the whole process when it is built
    model.inst_interactive_predictor.model.bf16_context.__exit__(None, None, None)
    if precision == "fp32":
        vitdet.addmm_act = _fp32_addmm_act
        # model_builder turns TF32 on at import
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    return Sam3Processor(model)


def _box_cxcywh(box, width, height):
    x0, y0, x1, y1 = box[:4]
    return [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]


def run_prompt(processor: Sam3Processor, state: dict, prompt: dict) -> dict:
    """Run one prompt the way osam does: the text encoder runs on every detection prompt"""
    model = processor.model
    width, height = state["original_width"], state["original_height"]
    if "points" in prompt:
        pts = np.array(prompt["points"], dtype=np.float32)
        kwargs = dict(point_coords=pts[:, :2], point_labels=pts[:, 2], multimask_output=True)
        with nvtx.range("torch:point_decoder"):
            _, iou0, low0 = model.predict_inst(state, **kwargs)
            masks, iou, low = model.predict_inst(state, mask_input=low0[[iou0.argmax()]], **kwargs)
        return {"iou_first": iou0, "low_res_first": low0, "iou": iou, "low_res": low, "masks": masks}

    processor.reset_all_prompts(state)
    text = prompt.get("text", "visual")
    with nvtx.range("torch:language_encoder"):
        state["backbone_out"].update(model.backbone.forward_text([text], device=processor.device))
    geo = model._get_dummy_prompt()
    for box in prompt.get("boxes", []):
        cxcywh = torch.tensor(_box_cxcywh(box, width, height), device="cuda").view(1, 1, 4)
        label = torch.tensor([bool(box[4])], device="cuda").view(1, 1)
        geo.append_boxes(cxcywh, label)
    state["geometric_prompt"] = geo
    with nvtx.range("torch:decoder"):
        out = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=geo,
            find_target=None,
        )
    # Sam3Processor._forward_grounding's postprocessing, inlined so the decoder runs once
    probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    keep = probs > processor.confidence_threshold
    boxes = box_cxcywh_to_xyxy(out["pred_boxes"][keep])
    boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)
    masks = F.interpolate(
        out["pred_masks"][keep].unsqueeze(1), (height, width), mode="bilinear", align_corners=False
    ).sigmoid()
    return {
        "scores_all": probs[0],
        "boxes_all": out["pred_boxes"][0],
        "mask_logits_all": out["pred_masks"][0],
        "presence_logit": out["presence_logit_dec"][0],
        "scores": probs[keep],
        "boxes_xyxy": boxes,
        "masks": masks[:, 0] > 0.5,
        "language_features": state["backbone_out"]["language_features"][:, 0],
        "language_mask": state["backbone_out"]["language_mask"][0],
    }


def _to_tensor(v) -> torch.Tensor:
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    v = v.detach().cpu()
    if v.dtype == torch.bool:
        v = v.to(torch.uint8)
    if v.is_floating_point():
        v = v.float()
    return v.contiguous()


@torch.inference_mode()
def dump(args) -> None:
    processor = build(args.ckpt, "fp32")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(pathlib.Path(args.prompts).read_text())
    for spec in specs:
        image = PIL.Image.open(ROOT / spec["image"]).convert("RGB")
        rgb = np.asarray(image)
        tensors = {"image_rgb": torch.from_numpy(rgb.copy())}

        # the processor's own transform, kept so sam3rs can compare its resize
        pixels = processor.transform(
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(processor.device)
        )
        tensors["pixels"] = pixels

        trunk = {}
        hook = processor.model.backbone.vision_backbone.trunk.register_forward_hook(
            lambda _m, _i, o: trunk.update(out=o[-1])
        )
        state = processor.set_image(image)
        hook.remove()
        bo = state["backbone_out"]
        tensors["trunk"] = trunk["out"][0]
        for i in range(3):
            tensors[f"fpn{i}"] = bo["backbone_fpn"][i][0]
            tensors[f"pos{i}"] = bo["vision_pos_enc"][i][0]
            tensors[f"sam2_fpn{i}"] = bo["sam2_backbone_out"]["backbone_fpn"][i][0]

        meta = {"image": spec["image"], "prompts": []}
        for prompt in spec["prompts"]:
            result = run_prompt(processor, state, prompt)
            key = prompt["name"].replace(" ", "_")
            for k, v in result.items():
                tensors[f"{key}.{k}"] = _to_tensor(v)
            meta["prompts"].append(prompt)
        tensors = {k: _to_tensor(v) for k, v in tensors.items()}
        name = pathlib.Path(spec["image"]).stem
        save_file(tensors, out_dir / f"{name}.safetensors", metadata={"meta": json.dumps(meta)})
        print(f"wrote {name}: {len(tensors)} tensors")


def _median_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


@torch.inference_mode()
def bench(args) -> None:
    processor = build(args.ckpt, args.precision)
    image = PIL.Image.open(ROOT / args.image).convert("RGB")
    with nvtx.range("encode warmup"):
        state = processor.set_image(image)

    def encode():
        with nvtx.range("encode"):
            processor.set_image(image)

    row = [f"{_median_ms(encode, args.repeat):.1f}"]
    for prompt in BENCH_PROMPTS:
        with nvtx.range(f"{prompt['name']} warmup"):
            run_prompt(processor, state, prompt)

        def generate(prompt=prompt):
            with nvtx.range(prompt["name"]):
                run_prompt(processor, state, prompt)

        row.append(f"{_median_ms(generate, args.repeat):.1f}")
    print("| Runtime | Encode | " + " | ".join(p["name"] for p in BENCH_PROMPTS) + " |")
    print(f"| PyTorch {args.precision} | " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--prompts", default=str(ROOT / "tests/parity_prompts.json"))
    b = sub.add_parser("bench")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    b.add_argument("--image", default="models/images/dogs.jpg")
    b.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    {"dump": dump, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
