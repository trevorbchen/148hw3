# EE/CS 148B — HW 3 Writeup

**Author:** Trevor Chen
**Repo:** https://github.com/trevorbchen/148hw3

This writeup follows the structure of `hw3.pdf`. Numbers and figures are
filled in from `runs/*/metrics.json` and `runs/*/figures/*.png` after the
experiments are run on Colab.

> **How to fill this in:** every `[FILL IN: …]` marker corresponds to a value
> that lives in a `runs/<experiment>/metrics.json` file, or a figure in
> `runs/<experiment>/figures/`. Drop in numbers, paste figures, and write
> the discussion paragraphs. Convert to PDF with
> `pandoc writeup.md -o writeup.pdf` once done.

---

## §2 — Vision Transformer

### Problem (patch_embeddings) — Patchification

Implementation in [`basics/vit.py:14-48`](basics/vit.py).
Uses a strided `Conv2d` with `kernel_size = stride = patch_size`, then
`flatten(2).transpose(1, 2)` to yield `(B, N, d_model)`. Verified by
`tests/test_vit.py::test_patch_embeddings_shape` and
`test_patch_embeddings_partition`.

### Problem (vit) — Building the ViT

Implementation in [`basics/vit.py:51-end`](basics/vit.py). CLS token,
learnable positional embedding (or RoPE for §6), `num_blocks` Transformer
blocks with `is_decoder=False`, final LayerNorm, return CLS token (or all
tokens with `return_all_tokens=True`).

### Problem (vit_pooling) — CLS vs. mean pooling vs. attention pooling

For tasks that require *spatial* reasoning (object counting, OCR,
region-aware VQA), passing only the CLS embedding to the language model is
fundamentally lossy: a single 384-dim vector cannot represent which object
is *where* on the grid. Mean-pooling preserves a small amount of position
information through the spatial composition of the pooled features but
still erases the per-patch structure that downstream attention could
exploit. Attention-pooling (a learned query that attends to all patches)
sits between the two — it gives the decoder a small, fixed-size summary
while still letting the pooled query weight different regions adaptively.
For a VLM whose decoder *can* attend over many tokens, the all-patches
prefix from §5.4 is strictly more expressive than CLS-only and is what
LLaVA and Qwen2-VL use in practice. The information that a CLS-only
summary loses is exactly what enables compositional reasoning: spatial
relations between distinct image regions.

### Problem (vit_patch_size) — Patch-size sweep

(1) For a 224×224 image,
`N = (224/P)²`. So `P=8 → N=784`, `P=16 → N=196`, `P=32 → N=49`. Self-attention
cost scales as `O(N² · d_model)`, so halving `P` quadruples `N` and
multiplies attention cost by 16.

(2) Forward-pass timing for `d_model=384, num_heads=6, num_blocks=6` on
batch size 16, averaged over 20 steps after 5 warmup steps (run via
`python scripts/patch_size_sweep.py`):

| Patch size *P* | # patches *N* | Forward (ms, mean ± std) |
|----------------|---------------|--------------------------|
| 8              | 784           | [FILL IN]                |
| 16             | 196           | [FILL IN]                |
| 32             | 49            | [FILL IN]                |

Source: `runs/patch_size_sweep/metrics.json`,
`runs/patch_size_sweep/figures/patch_size_time.png`.

(3) Smaller patches preserve fine-grained detail and are worth the cost
when downstream prediction depends on information at a sub-patch scale
(small objects, dense prediction, OCR), or on a small image where coarse
patching would leave only a handful of tokens for attention to compose
over.

---

## §3 — CLIP-Style Contrastive Pretraining

### Problem (clip_setup) — Projection heads

Implementation in [`vlm/clip.py:17-40`](vlm/clip.py): two unbiased
`nn.Linear` heads project image (`d_model = 384`) and text (MiniLM
`d_text = 384`) into `d_proj = 256`, both L2-normalized.

### Problem (infonce) — Symmetric InfoNCE

Implementation in [`vlm/clip.py:48-end`](vlm/clip.py).

The loss is symmetric because the contrastive objective has two
*independent* failure modes that we want to penalize equally: (i) an
image's embedding being closer to a wrong caption than its own
("image→text retrieval") and (ii) a caption's embedding being closer to a
wrong image than its own ("text→image retrieval"). Averaging
`CE(S, y)` and `CE(S^T, y)` jointly penalizes both, and matches the
training-time use case (both directions are valuable downstream).

### Problem (clip_train) — Pretraining on EuroSAT

Trained 20 epochs with the default config (`configs/clip_eurosat.yaml`).
Run command: `python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --pos-encoding learned`.

- **Best val zero-shot acc:** [FILL IN] (`runs/clip_eurosat_learned/metrics.json::best_val_acc`)
- **Test acc at best epoch:** [FILL IN] (`metrics.json::test_acc`)

**Loss curve:** `runs/clip_eurosat_learned/figures/loss.png`
**Val accuracy curve:** `runs/clip_eurosat_learned/figures/val_acc.png`

**Discussion.** Train loss continues to decrease past the point where val
zero-shot accuracy plateaus, indicating that the model keeps memorizing
batch-level instance discrimination signal that doesn't generalize. Per
the assignment note, EuroSAT captions are 10 class templates so many
in-batch positives share captions — InfoNCE penalizes this as if the
duplicates were negatives, which artificially inflates loss and limits
how meaningful raw loss values are as a quality proxy.
[FILL IN: 1 more sentence comparing the curves you observe.]

### Problem (clip_zeroshot) — Qualitative analysis

Run: `python scripts/clip_zeroshot_qualitative.py --checkpoint runs/clip_eurosat_learned/best.pt`.

Outputs:
- `runs/clip_qualitative/images/correct/*.png` (5 correct examples)
- `runs/clip_qualitative/images/wrong/*.png` (5 wrong examples)
- `runs/clip_qualitative/examples.jsonl` (top-3 per example)
- `runs/clip_qualitative/figures/confusion_matrix.png`

| # | Image | Gold | Top-3 prediction | Correct? |
|---|-------|------|-------------------|----------|
| 1 | [PNG] | [FILL] | [FILL] | ✓ |
| 2 | [PNG] | [FILL] | [FILL] | ✓ |
| 3 | [PNG] | [FILL] | [FILL] | ✓ |
| 4 | [PNG] | [FILL] | [FILL] | ✓ |
| 5 | [PNG] | [FILL] | [FILL] | ✓ |
| 6 | [PNG] | [FILL] | [FILL] | ✗ |
| 7 | [PNG] | [FILL] | [FILL] | ✗ |
| 8 | [PNG] | [FILL] | [FILL] | ✗ |
| 9 | [PNG] | [FILL] | [FILL] | ✗ |
| 10| [PNG] | [FILL] | [FILL] | ✗ |

**Discussion.** [FILL IN: are mistakes "reasonable" (e.g., PermanentCrop ↔
HerbaceousVegetation, Highway ↔ Industrial Buildings) or do they look
random? Reasonable confusions suggest the embedding space has learned a
semantic geometry — visually-similar classes cluster together — even when
the top-1 prediction is wrong. Look at the confusion matrix
(`figures/confusion_matrix.png`) to see which classes the model
systematically confuses.]

---

## §4 — LoRA Fine-Tuning

### Problem (lora_linear) — LoRA-wrapped linear layer

Implementation in [`basics/lora.py:11-46`](basics/lora.py) and
[`apply_lora_to_attention`](basics/lora.py:49-end).

Parameter counts for a ViT (`d_model=384, num_heads=6, num_blocks=6`) with
LoRA rank 8 applied to `q_proj` and `v_proj`:

- **Total params:** [FILL IN]
- **Trainable params:** [FILL IN]
- **Trainable / total:** [FILL IN]

(Read from `runs/resisc_lora_rank8/metrics.json::trainable_params /
total_params`.)

### Problem (lora_compare) — Full FT vs. LoRA vs. linear probe

Run:
```bash
python scripts/finetune_resisc.py --config configs/lora_resisc.yaml --method linear_probe --pretrained runs/clip_eurosat_learned/best.pt
python scripts/finetune_resisc.py --config configs/lora_resisc.yaml --method lora --rank 8 --alpha 16 --pretrained runs/clip_eurosat_learned/best.pt
python scripts/finetune_resisc.py --config configs/lora_resisc.yaml --method full_ft --pretrained runs/clip_eurosat_learned/best.pt
```

| Method        | Test acc | Trainable params | Peak mem (MB) | Wall time (s) |
|---------------|----------|------------------|---------------|---------------|
| Linear probe  | [FILL]   | [FILL]           | [FILL]        | [FILL]        |
| LoRA r=8 α=16 | [FILL]   | [FILL]           | [FILL]        | [FILL]        |
| Full FT       | [FILL]   | [FILL]           | [FILL]        | [FILL]        |

(Source: `runs/resisc_*/metrics.json`.)

**Discussion.** [FILL IN: the typical pattern is full FT > LoRA > linear
probe in accuracy, but full FT uses the most memory (optimizer states for
every param) and time, LoRA gets ~95% of full FT's accuracy with
roughly 1% of the trainable parameters and a fraction of the memory, and
linear probe is fastest but capped because the underlying ViT can't adapt
its representations. For RESISC45 (45 classes, modest domain shift from
EuroSAT) LoRA is the sweet spot — discuss in 4-5 sentences using your
actual numbers.]

### Problem (lora_rank) — Rank sweep

Run: `python scripts/lora_rank_sweep.py --config configs/lora_resisc.yaml --pretrained runs/clip_eurosat_learned/best.pt`.

Plot: `runs/lora_rank_sweep/figures/rank_sweep.png`.

| Rank r | α (=2r) | Test acc | Trainable params |
|--------|---------|----------|------------------|
| 1      | 2       | [FILL]   | [FILL]           |
| 2      | 4       | [FILL]   | [FILL]           |
| 4      | 8       | [FILL]   | [FILL]           |
| 8      | 16      | [FILL]   | [FILL]           |
| 16     | 32      | [FILL]   | [FILL]           |
| 32     | 64      | [FILL]   | [FILL]           |
| 64     | 128     | [FILL]   | [FILL]           |

(1) **Diminishing returns at:** [FILL IN: typically r ≈ 8-16].

(2) Practical deployments use r=8 or r=16 because, as this sweep shows,
the *effective* rank of the fine-tuning update is small — past a few
dozen ranks accuracy plateaus while parameter count keeps growing
linearly. The original LoRA paper's empirical observation that ΔW is
near-low-rank is what justifies the entire method.

---

## §5 — Vision-Language Model

### Problem (projector) — Vision-language projector

Implementation in [`vlm/projector.py`](vlm/projector.py): a 2-layer MLP
`Linear(d_image, 4·d_image) → GELU → Linear(4·d_image, d_decoder)`,
shape-flexible to handle both `(B, d)` (CLS pooled) and `(B, N, d)` (all
patches).

**Why more than a single linear layer.** During VLM pretraining the
encoder and decoder are frozen, so the projector is the *only*
representational bridge between two completely separately-trained
embedding spaces. A single `Linear` can only realize a rotation + scaling
of image features into the decoder's basis — fine if those bases happened
to be aligned, but they aren't. The MLP nonlinearity gives the projector
the capacity to *compose* image features, suppress decoder-irrelevant
directions, and inject information the decoder needs (e.g., recoded into
"token-shaped" subspaces it actually uses) without needing to retrain
either backbone.

### Problem (injection) — Token injection strategies

Implementation in [`vlm/model.py`](vlm/model.py): three strategies
(`cls`, `all_patches`, `interleaved`) handled by `_stitch_prepend` and
`_stitch_interleaved`. Visual-token positions in `labels` are filled with
-100 inside `forward()` so HF's loss skips them.

### Problem (injection_compare) — Best injection strategy

Run:
```bash
for inj in cls all_patches interleaved; do
  python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
    --pretrained-vit runs/clip_eurosat_learned/best.pt \
    --injection $inj --mask-mode causal --freeze-config A
done
```

| Strategy      | Val EM acc | # visual tokens | Peak mem (MB) | s / step |
|---------------|------------|-----------------|---------------|----------|
| CLS-only      | [FILL]     | 1               | [FILL]        | [FILL]   |
| All-patches   | [FILL]     | 65              | [FILL]        | [FILL]   |
| Interleaved   | [FILL]     | 65              | [FILL]        | [FILL]   |

(Source: `runs/vlm_{cls,all_patches,interleaved}_causal_A/metrics.json`.)

**Discussion.** [FILL IN: CLS-only should be cheapest but worst on
spatially-grounded questions, mirroring Problem (vit_pooling). All-patches
should give the decoder access to per-region features and yield the best
accuracy at a higher memory cost (longer sequence → larger attention
matrix). Interleaved is structurally equivalent to all-patches for
single-image inputs and should match it; its real value is generalizing
to multiple images. The connection to (vit_pooling) is direct: any
question requiring object position or counting needs the patch-level
information that CLS pooling discards before it ever reaches the
decoder.]

### Problem (masking) — Image-block attention

(1) Mask diagrams for 4 visual + 3 text tokens (rows = query, cols = key;
shaded = allowed):

**(M1) Fully causal:**
```
       v1 v2 v3 v4 t1 t2 t3
   v1: ■  .  .  .  .  .  .
   v2: ■  ■  .  .  .  .  .
   v3: ■  ■  ■  .  .  .  .
   v4: ■  ■  ■  ■  .  .  .
   t1: ■  ■  ■  ■  ■  .  .
   t2: ■  ■  ■  ■  ■  ■  .
   t3: ■  ■  ■  ■  ■  ■  ■
```

**(M2) Bidirectional inside image, causal across boundary:**
```
       v1 v2 v3 v4 t1 t2 t3
   v1: ■  ■  ■  ■  .  .  .
   v2: ■  ■  ■  ■  .  .  .
   v3: ■  ■  ■  ■  .  .  .
   v4: ■  ■  ■  ■  .  .  .
   t1: ■  ■  ■  ■  ■  .  .
   t2: ■  ■  ■  ■  ■  ■  .
   t3: ■  ■  ■  ■  ■  ■  ■
```

(2) **(M2) should perform better.** The ViT was pretrained with fully
bidirectional attention; clamping it to causal at fine-tune time wastes
roughly half the patch-to-patch interactions the encoder already learned.
The causal-across-boundary part is unchanged: text still attends to all
prior tokens (visual + text), so the language model's autoregressive
decoding is preserved.

(3) Run: 500 steps each.
```bash
python scripts/train_vlm.py --config configs/vlm_clevr.yaml --pretrained-vit ... \
    --injection all_patches --mask-mode causal --freeze-config A \
    --output-dir runs/vlm_all_patches_causal_A_short
python scripts/train_vlm.py --config configs/vlm_clevr.yaml --pretrained-vit ... \
    --injection all_patches --mask-mode image_bidir --freeze-config A \
    --output-dir runs/vlm_all_patches_image_bidir_A_short
```
(Set `train.num_steps: 500` in the config or shadow via CLI.)

| Mask mode    | Val EM acc (500 steps) |
|--------------|------------------------|
| Causal       | [FILL]                 |
| Image-bidir  | [FILL]                 |

### Problem (freezing) — What to train

Run:
```bash
for cfg in A B C D; do
  python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
    --pretrained-vit runs/clip_eurosat_learned/best.pt \
    --injection all_patches --mask-mode image_bidir --freeze-config $cfg
done
```

| Config | What's trained                          | Val EM acc | Trainable params | Peak mem (MB) |
|--------|-----------------------------------------|------------|------------------|---------------|
| A      | projector only                          | [FILL]     | [FILL]           | [FILL]        |
| B      | projector + decoder LoRA (r=8)          | [FILL]     | [FILL]           | [FILL]        |
| C      | projector + full decoder                | [FILL]     | [FILL]           | [FILL]        |
| D      | full ViT + projector + full decoder     | [FILL]     | [FILL]           | [FILL]        |

(Source: `runs/vlm_all_patches_image_bidir_{A,B,C,D}/metrics.json`.)

**Discussion.** [FILL IN — typical observation, write 5-6 sentences:
- A serves as the "alignment / pretraining" stage: cheap, fast, low ceiling
  because the language model never adapts to the new visual modality.
- B (decoder LoRA) is the recommended instruction-tuning stage: the
  decoder gets to specialize on the visual grounded prompt format with a
  small parameter budget; close to C with much less memory.
- C (full decoder FT) is the highest-fidelity but most memory-hungry
  configuration; on a small dataset like CLEVR it can over-fit.
- D (everything) often hurts because the CLIP-pretrained ViT loses its
  language-aligned features when allowed to drift, especially on a
  domain (CLEVR) very different from the pretraining domain (EuroSAT).
- The two-stage recipe (A → B) gets most of the benefit with minimal
  cost; this matches LLaVA's design.]

### Problem (vlm_qualitative) — What has the VLM learned?

Run: `python scripts/eval_vlm.py --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt --num-examples 10 --save-images`.

Outputs in `runs/vlm_qualitative/`: `examples.jsonl`, `images/*.png`,
`figures/acc_by_qtype.png`.

| # | Image | Question | Gold | Prediction | Correct? |
|---|-------|----------|------|------------|----------|
| 1 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✓ |
| 2 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✓ |
| 3 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✓ |
| 4 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✓ |
| 5 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✓ |
| 6 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✗ |
| 7 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✗ |
| 8 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✗ |
| 9 | [PNG] | [FILL]   | [FILL] | [FILL]   | ✗ |
| 10| [PNG] | [FILL]   | [FILL] | [FILL]   | ✗ |

**Discussion.** [FILL IN: for each wrong case, hypothesize encoder vs
decoder failure. Encoder failures look like miscounting, wrong shape, or
wrong color — perceptual mistakes. Decoder failures look like fluent but
question-misreading answers (e.g., answers a different question, ignores
a "behind" relation, returns a count for a yes/no question). To
distinguish the two: hold the image fixed and rephrase the question
(decoder-free probes); or hold the question fixed and swap in known
similar images (encoder probes). A more rigorous experiment: replace the
image embedding with the gold attribute embedding (oracle perception) and
see how much accuracy jumps — that gap is the decoder-side ceiling.]

---

## §6 — Positional Encodings and RoPE

### Problem (rope_1d) — 1D RoPE

Implementation in [`basics/rope.py:30-end`](basics/rope.py).

**Norm preservation check.**
```python
import torch
from basics.rope import RoPE1D
rope = RoPE1D(head_dim=64, max_seq_len=128)
x = torch.randn(2, 4, 16, 64)
y = rope(x, torch.arange(16))
print((x.norm(dim=-1) - y.norm(dim=-1)).abs().max())
```
Result: `[FILL IN: should be < 1e-5]`. RoPE is a per-pair 2D rotation, so
norms are preserved exactly up to floating-point precision — confirmed
by `tests/test_rope.py::test_rope_1d_preserves_norm`.

### Problem (rope_vs_learned) — Learned PE vs. RoPE in the ViT

Train each variant for 20 epochs with the default config:
```bash
python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --pos-encoding learned --extrapolation-img-size 96
python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --pos-encoding rope1d  --extrapolation-img-size 96
```

For the learned-PE baseline, the ViT now bilinearly interpolates the
learned positional embedding from the 8×8 training grid to the 12×12 eval
grid (`basics.vit.ViT._interpolate_learned_pos_embed`).

| PE method | Train-size val acc (64×64, 64 patches) | Extrapolated val acc (96×96, 144 patches) |
|-----------|----------------------------------------|-------------------------------------------|
| Learned   | [FILL]                                 | [FILL]                                    |
| 1D RoPE   | [FILL]                                 | [FILL]                                    |

(Source: `runs/clip_eurosat_{learned,rope1d}/metrics.json`.)

**Discussion.** [FILL IN: typical observation: learned PE drops sharply
when extrapolated even with bilinear interpolation, because the *absolute*
position values it learned aren't sensible at unseen positions. RoPE
degrades more gracefully because (a) attention only sees the *relative*
offset, which is well-defined at any position, and (b) the cos/sin tables
are evaluable at any integer position out-of-the-box. 3-4 sentences.]

### Problem (rope_2d) — 2D RoPE

Implementation in [`basics/rope.py:RoPE2D`](basics/rope.py). The ViT
constructs (x, y) coordinates per patch when `pos_encoding="rope2d"`
(`ViT._make_rope_apply`).

Run:
```bash
python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --pos-encoding rope2d --extrapolation-img-size 96
```

| PE method | Train-size val acc | Extrapolated val acc |
|-----------|--------------------|----------------------|
| 1D RoPE   | [FILL]             | [FILL]               |
| 2D RoPE   | [FILL]             | [FILL]               |

(Source: `runs/clip_eurosat_rope2d/metrics.json`.)

**Discussion.** [FILL IN: 2D RoPE encodes the actual 2D grid distance
between two patches, while 1D RoPE collapses the grid to a row-major 1D
sequence — so two patches in adjacent rows look "far apart" to 1D RoPE
even when they're geometrically adjacent. On EuroSAT this only modestly
helps because satellite imagery is mostly statistically isotropic, but
the gap should widen on tasks with explicit spatial structure. 2-3
sentences.]

### Problem (mrope_written) — Reasoning about M-RoPE

(1) **Naive 1D position IDs failure mode.** With 65 visual tokens (CLS +
64 patches) followed by 50 text tokens, naive `0..114` IDs push text
tokens to positions 65-114, which are perfectly fine in absolute terms
but conflate two qualitatively different things — image position and
text position — into a single dimension. Worse, the patches lose 2D
structure: patch (3, 4) and patch (4, 3) get identical 1D IDs only by
coincidence of row-major flattening, even though they're geometrically
distinct neighbors. RoPE's relative-distance trick now treats `dist((3,4),
(4,4)) = 1` (next row, same column) the same as `dist((4,3),(4,4)) = 1`
(same row, adjacent column), but treats `dist((3,4),(4,3))` as some
unrelated 1D delta — so attention can't easily express "look one row
down".

(2) **First text token's position under M-RoPE.** Text tokens get
`(t = 1, x = grid_w + 1, y = grid_h + 1)` — i.e., temporal index advances
to 1, and the (x, y) coordinates jump just past the maximum image grid
size. So with an 8×8 patch grid the first text token is at temporal 1,
spatial (9, 9). This is sensible because (a) it gives text a clean
"this is text, not image" signature in the temporal channel, (b) it
makes text positions independent of how many patches the image has, so
adding more visual tokens doesn't inflate text position IDs out of the
range the decoder was pretrained on.

(3) **Why three chunks instead of two.** Splitting head_dim into
(t, x, y) lets every dot-product carry information about all three axes
*simultaneously*. If we dropped `t`, image-vs-text would have to be
encoded in the (x, y) values themselves (e.g., text gets coords past the
image), which works only as long as the decoder learns the convention.
The temporal channel makes the modality boundary explicit, generalizes
trivially to multiple images per prompt (each image gets its own
temporal index), and avoids the confound where a text token at (x, y)
near the image grid would accidentally look spatially "close" to an
image patch under attention.

### Problem (mrope_impl) — Implementing M-RoPE *(bonus)*

Skipped this iteration. The hooks are in place — `vlm/model.py` would
need a `position_assignment` flag and a 3D-aware RoPE applied inside the
decoder's attention; this requires patching SmolLM2's attention layers,
which is more involved than the 1D/2D variants in `basics/rope.py`.

---

## Submission

- **`writeup.pdf`** — this document, converted with `pandoc writeup.md -o writeup.pdf`.
- **`code.zip`** — `git archive HEAD --format=zip > code.zip` (or zip the
  whole repo excluding `runs/`, `data/`, `__pycache__/`).
- The repo is also pushed to https://github.com/trevorbchen/148hw3.
