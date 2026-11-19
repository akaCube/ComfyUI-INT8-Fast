# ksd's fork of int8-fast

The branch `ksd` is int8-fast 1.3.4 (upstream commit db4cd8e, as the Comfy Registry ships it) with two
changes, used by ksd's workers (github.com/akaCube/ksd):

1. **Kernel tuning per size range** (`int8_fused_kernel.py`). Triton tuned the INT8 matmul kernels again for
   every new number of tokens, which costs 6 to 17 s each time on an RTX 4090. The tuning key now uses a size
   range; `KSD_INT8_ONE_CONFIG=1` skips tuning altogether.
2. **LoRAs with the remainder file** (`int8_remainder.py`, and lines marked `# ksd: remainder`). Baking a LoRA
   into INT8 weights loses what quantizing threw away, and the picture drifts. With the model's remainder file
   next to it (`<model stem>_remainder4.safetensors`) the weight becomes (q + remainder) * scale before the
   LoRA is added, and the result looks like a file baked from bf16. Works with ComfyUI's LoRA loaders and with
   the pre-LoRA loader, which with an INT8 file used to bake only the layers kept in bf16.

Upstream changes are merged on purpose, with a measured run. The licence stays AGPL-3.0 (LICENSE).
