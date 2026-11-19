"""LoRAs on an INT8 model, baked as if from bf16, with the model's remainder file (ksd's fork of int8-fast:
github.com/akaCube/ComfyUI-INT8-Fast, branch ksd; the remainder file is made by make_remainder.py in akaCube/ksd,
ksd/runpod/int8_delta).

When the model was quantized, each weight was rounded to its INT8 step and the rest thrown away. int8-fast bakes
a LoRA into INT8 weights by turning them back into floats, adding the LoRA (ComfyUI's own code: every LoRA type)
and rounding again; without the thrown-away rest, a weight moves to the wrong step about half the time it matters,
and the picture drifts (measured on 2026-11-18 and 19). The remainder file (ksd/runpod/int8_delta/
make_remainder.py, 4 bits a weight) gives that rest back: the weight becomes (q + remainder) * scale before the
LoRA is added. The images at the same seed then look like those of a file baked from bf16.

The remainder file is found next to the model file: <model stem>_remainder<bits>.safetensors, made from exactly
that file (its metadata names it). Without one, everything works as before.

Used in three places of int8-fast (marked "# ksd: remainder"):
  use_for(path)        the INT8 loader, when it loads a model file
  with_remainder(...)  int8-fast's two ways of baking a LoRA from ComfyUI's LoRA loaders (whole and per layer)
  bake_at_load(...)    the pre-LoRA loader with an INT8 file, which used to leave the INT8 layers out entirely
"""
import json
import logging
import os
import re

import torch

_state = {"path": None, "handle": None, "keys": {}, "bits": None, "model": None, "looked": 0.0}


def norm(key):
    """A layer's name without prefixes and without .weight, so model, file and LoRA names meet."""
    key = re.sub(r"\.(weight|remainder)$", "", key)
    return re.sub(r"^(model\.)?(diffusion_model\.|transformer\.)?", "", key)


def use_for(model_path):
    """Opens the remainder file that belongs to this model file, if there is one."""
    import time
    from safetensors import safe_open
    _state.update(path=None, handle=None, keys={}, bits=None, model=model_path, looked=time.time())
    folder, name = os.path.split(model_path)
    stem = os.path.splitext(name)[0]
    for f in sorted(os.listdir(folder)):
        if not re.fullmatch(re.escape(stem) + r"_remainder\d+\.safetensors", f):
            continue
        path = os.path.join(folder, f)
        handle = safe_open(path, framework="pt", device="cpu")
        meta = handle.metadata() or {}
        if meta.get("int8") and meta["int8"] != name:
            logging.warning(f"ksd remainder: {f} was made from {meta['int8']}, not {name}: not used")
            continue
        keys = {norm(k): k for k in handle.keys() if k.endswith(".remainder")}
        _state.update(path=path, handle=handle, keys=keys, bits=int(meta.get("bits", 4)))
        logging.info(f"ksd remainder: {f} ({_state['bits']} bits, {len(keys)} layers): LoRAs bake as from bf16")
        return True
    return False


def unpack(packed, bits, n):
    """Codes back from bytes (make_remainder.pack), as remainders in steps: the middle of each code's range."""
    per, levels = 8 // bits, 2 ** bits
    parts = [(packed >> (i * bits)) & (levels - 1) for i in range(per)]
    c = torch.stack(parts, dim=2).reshape(packed.shape[0], -1)[:, :n]
    return (c.float() + 0.5) / levels - 0.5


def _look_again():
    """A remainder file that arrives after the model was loaded (a worker fetching it in the background) is used
    from then on: ComfyUI keeps a loaded model, so use_for would not run again. Looked for at most every 10 s."""
    import time
    if _state["handle"] is None and _state["model"] and time.time() - _state["looked"] > 10:
        use_for(_state["model"])


def with_remainder(key, q):
    """The INT8 weight q of layer `key` with its remainder added (float, in steps), or q as it is."""
    _look_again()
    k = _state["keys"].get(norm(key)) if _state["handle"] is not None else None
    if k is None or q.dtype != torch.int8:
        return q
    packed = _state["handle"].get_tensor(k).to(q.device)
    return q.float() + unpack(packed, _state["bits"], q.shape[1])


def bake_at_load(module, key, weight, scale, patches, quant_conf, device):
    """The pre-LoRA loader's patches baked into an INT8 weight as it loads, the way int8-fast's own LoRA path does
    it (back to floats, out of the rotation, ComfyUI's calculate_weight, into the rotation, rounded with the same
    scale), with the remainder added first when there is one."""
    import comfy.lora
    from .int8_quant import dequantize, quantize_int8
    q = weight.to(device)
    s = scale.to(device) if isinstance(scale, torch.Tensor) else scale
    x = dequantize(with_remainder(key, q), s)
    group = int((quant_conf or {}).get("convrot_groupsize", 256)) if (quant_conf or {}).get("convrot") else 0
    if group and x.shape[1] % group == 0:
        from .convrot import build_hadamard, rotate_weight
        h = build_hadamard(group, device=x.device, dtype=x.dtype)
        x = rotate_weight(x, h, group_size=group)
    x = comfy.lora.calculate_weight(module._format_lora_patches(patches), x, key)
    if group and x.shape[1] % group == 0:
        x = rotate_weight(x, h, group_size=group)
    return quantize_int8(x, s).to(weight.device)
