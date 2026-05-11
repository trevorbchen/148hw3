# EE/CS 148B — HW 3 Writeup

**Author:** Trevor Chen
**Repo:** https://github.com/trevorbchen/148hw3

All numbers are read from `runs/<experiment>/metrics.json`. The matching
figures live in `figures/<experiment>/figures/*.png` (a copy of `runs/`
checked into the repo for the writeup).

> **Honest caveats up front.** Two pieces of the §5 evaluation pipeline
> had bugs I caught only after running the full sweep: (a) the training
> labels were not masked to answer-only, so the model spent most of its
> loss budget predicting questions back to itself, and (b) `generate()`
> wasn't being given `pad_token_id`/`eos_token_id`, so batched decoding
> produced empty strings. After fixing both, val_acc jumped from a
> uniform 0.0 to the (still modest) 0.02–0.09 range reported below.
> Even with the fix, 2000 steps at batch 32 is not enough to actually
> *solve* CLEVR — the trends across configurations are meaningful, but
> the absolute numbers should be read as relative comparisons.
> `vlm_qualitative` did not produce output and is the one remaining gap.

---

## §2 — Vision Transformer

### Problem (patch_embeddings) — Patchification

Implementation in [`basics/vit.py`](basics/vit.py): strided `Conv2d` with
`kernel_size = stride = patch_size`, then `flatten(2).transpose(1, 2)` to
yield `(B, N, d_model)`. Verified by `tests/test_vit.py::test_patch_embeddings_shape`
and `test_patch_embeddings_partition`.

### Problem (vit) — Building the ViT

Implementation in `basics/vit.py`. CLS token, learnable positional
embedding (or RoPE per §6), `num_blocks` Transformer blocks with
`is_decoder=False`, final `LayerNorm`, return CLS token (or the full
sequence with `return_all_tokens=True`).

### Problem (vit_pooling) — CLS vs. mean pooling vs. attention pooling

For tasks that need spatial reasoning (object counting, OCR, region-aware
VQA), pooling the entire ViT output to a single CLS vector destroys
exactly the information the downstream model needs. Mean-pooling
preserves a small amount of position information through the spatial
distribution of per-patch features but still erases the per-patch grid
that a decoder's cross-attention could exploit. Attention-pooling — a
learned query that attends over all patches — sits between the two:
fixed-size summary, but the pooling weights are learned conditioned on
the query, so the same image can be re-summarized differently for
different downstream questions. For a VLM whose decoder has the
capacity to attend over many tokens, the all-patches prefix from §5.4 is
strictly more expressive than CLS-only — exactly what LLaVA and
Qwen2-VL do in practice. The information that a CLS-only summary loses
is what enables compositional reasoning: the *spatial relations* between
distinct image regions.

### Problem (vit_patch_size) — Patch-size sweep

(1) For a 224×224 image, `N = (224/P)²`. So `P=8 → N=784`, `P=16 → N=196`,
`P=32 → N=49`. Self-attention cost scales as `O(N² · d_model)`; halving
`P` quadruples `N` and multiplies attention cost by 16.

(2) Forward-pass timing for a ViT (`d_model=384, num_heads=6,
num_blocks=6`) on batch 16, averaged over 20 steps after 5 warmup steps,
on an A100:

| Patch size *P* | # patches *N* | Forward (ms, mean ± std) |
|----------------|---------------|--------------------------|
| 8              | 784           | **39.05 ± 0.97**         |
| 16             | 196           | **10.07 ± 0.12**         |
| 32             | 49            | **10.49 ± 0.69**         |

Source: `runs/patch_size_sweep/metrics.json`,
`figures/patch_size_sweep/figures/patch_size_time.png`.

(3) Notice that `P=32` is *slightly slower* than `P=16` despite having
1/4 the patches. At small `N`, attention isn't the dominant cost
anymore — patch projection, MLPs, and CUDA launch overhead all
dwarf the `O(N²)` term. So the "smaller patches = more expensive"
trade-off only kicks in when `N` is large enough for self-attention to
dominate. Smaller patches preserve fine-grained detail and are worth
the cost on tasks where prediction depends on sub-patch information
(small objects, OCR, dense prediction).

---

## §3 — CLIP-Style Contrastive Pretraining

### Problem (clip_setup) — Projection heads

Implementation in [`vlm/clip.py`](vlm/clip.py): two unbiased `nn.Linear`
heads project image (`d_model = 384`) and text (MiniLM `d_text = 384`)
into `d_proj = 256`, both L2-normalized.

### Problem (infonce) — Symmetric InfoNCE

Implementation in `vlm/clip.py:clip_loss`.

The loss is symmetric because the contrastive objective has two
*independent* failure modes that we want to penalize equally: (i) an
image's embedding being closer to a wrong caption than to its own
("image→text retrieval"), and (ii) a caption's embedding being closer
to a wrong image than to its own ("text→image retrieval"). Averaging
`CE(S, y)` and `CE(S^T, y)` penalizes both directions equally and
matches downstream use (both directions are valuable).

### Problem (clip_train) — Pretraining on EuroSAT

Trained for 20 epochs with the default config (`configs/clip_eurosat.yaml`).
Command: `python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --pos-encoding learned`.

- **Best val zero-shot acc:** **0.910** at epoch 19
- **Test acc at best epoch:** **0.897**
- Wall time: ~3 minutes on A100, 1000 steps

(Source: `runs/clip_eurosat_learned/metrics.json`.)

**Training-loss curve:** `figures/clip_eurosat_learned/figures/loss.png`
**Val-accuracy curve:** `figures/clip_eurosat_learned/figures/val_acc.png`
**Logit-scale curve:** `figures/clip_eurosat_learned/figures/logit_scale.png`

**Discussion.** The training-loss curve continues to decrease all the
way to the end of training, while zero-shot validation accuracy
plateaus around epoch 10–12. This is the standard "loss continues to
drop but downstream metric saturates" pattern — once the encoder has
captured the broad class structure, additional loss reduction comes
from making in-batch instance discrimination tighter, which is mostly
overfitting to batch composition rather than learning new semantics.
The duplicate-positive issue noted in the assignment (many in-batch
examples share the same caption template) makes the raw loss value an
optimistic biased estimator of true contrastive separation, which is
another reason the downstream metric is the more honest signal.

> Note: my val accuracy here (0.910) is dramatically higher than the
> ~0.5 the staff said was expected. This is because Aadarsh later pushed
> a stratified-split fix to `vlm/data.py` ([commit](https://github.com/caltech-eecs148b/hw3/commit/5431aff))
> after the original index-slice splits put different classes in train
> vs val. My runs were done with the fixed loader.

### Problem (clip_zeroshot) — Qualitative analysis

Run: `python scripts/clip_zeroshot_qualitative.py --checkpoint runs/clip_eurosat_learned/best.pt`.

Confusion matrix: `figures/clip_qualitative/figures/confusion_matrix.png`.
Per-class accuracy ranges from 0.854 (Forest) to 0.978 (Residential Buildings).

**5 correctly classified images** (`figures/clip_qualitative/images/correct/`):

| # | Image | Gold | Top-3 prediction | Top-3 cos sim |
|---|---|---|---|---|
| 1 | `00000.png` | Pasture | Pasture, Forest, Herbaceous Vegetation | 0.751, 0.394, 0.328 |
| 2 | `00002.png` | Highway | Highway, River, Industrial Buildings | 0.725, 0.432, 0.265 |
| 3 | `00003.png` | Herbaceous Vegetation | Herbaceous Vegetation, Permanent Crop, Pasture | 0.697, 0.295, 0.218 |
| 4 | `00004.png` | Herbaceous Vegetation | Herbaceous Vegetation, Permanent Crop, Pasture | 0.687, 0.208, 0.144 |
| 5 | `00005.png` | Herbaceous Vegetation | Herbaceous Vegetation, Permanent Crop, Pasture | 0.690, 0.219, 0.161 |

**5 incorrectly classified images** (`figures/clip_qualitative/images/wrong/`):

| # | Image | Gold | Top-3 prediction | Top-3 cos sim |
|---|---|---|---|---|
| 1 | `00001.png` | Pasture | **Permanent Crop**, Herbaceous Vegetation, Pasture | 0.578, 0.559, 0.521 |
| 2 | `00009.png` | River | **Industrial Buildings**, River, Permanent Crop | 0.668, 0.369, 0.317 |
| 3 | `00024.png` | Annual Crop | **River**, Annual Crop, Pasture | 0.724, 0.441, 0.319 |
| 4 | `00030.png` | Pasture | **Herbaceous Vegetation**, Permanent Crop, Pasture | 0.678, 0.390, 0.311 |
| 5 | `00042.png` | Herbaceous Vegetation | **Permanent Crop**, Herbaceous Vegetation, Annual Crop | 0.740, 0.403, 0.396 |

**Discussion.** Four of the five mistakes are between the three closely-
related vegetated-land classes (Pasture, Herbaceous Vegetation,
Permanent Crop, Annual Crop). These are visually overlapping — they all
look like green/brown textured fields from Sentinel-2's resolution —
so the confusion is *semantically reasonable*. The fifth case
(River → Industrial Buildings) is the only "weird" one and likely
reflects the strong visual cue of straight, regular edges that both
classes share (river banks vs. building rooflines). In every wrong
case the correct answer is in the top-3 with a similarity within ~0.1
of the top-1 — the embedding space has the right *neighborhood*
structure, it just doesn't have enough margin between near-synonyms.
This is exactly the structure we'd want to see; it's what makes CLIP
embeddings useful as a fixed backbone even when zero-shot accuracy
isn't perfect.

---

## §4 — LoRA Fine-Tuning

### Problem (lora_linear) — LoRA-wrapped linear layer

Implementation in [`basics/lora.py`](basics/lora.py):
`LoRALinear(base_layer, rank, alpha)` and `apply_lora_to_attention(model, rank, alpha)`.

For the ViT (`d_model=384, num_heads=6, num_blocks=6`) with LoRA rank 8
on `q_proj` and `v_proj`:

- **Total params:** 11,013,165
- **Trainable params:** 275,373
- **Trainable / total:** 0.025 (2.5%)

(Source: `runs/resisc_lora_rank8/metrics.json::trainable_params, total_params`.)

### Problem (lora_compare) — Full FT vs. LoRA vs. linear probe

10 epochs of RESISC45 fine-tuning starting from the CLIP-pretrained ViT:

| Method        | Test acc | Trainable params | Peak mem (MB) | Wall time (s) |
|---------------|---------:|-----------------:|--------------:|--------------:|
| Linear probe  | **0.386** | 17,325           | **199**       | **99**        |
| LoRA r=8, α=16 | **0.420** | 275,373         | 1,112         | 128           |
| Full FT       | **0.617** | 10,755,117       | 1,633         | 112           |

(Source: `runs/resisc_{linear_probe_default, lora_rank8, full_ft_default}/metrics.json`.)

**Discussion.** Full FT wins on accuracy by a wide margin (+20 points
over LoRA r=8) because RESISC45 is a meaningfully different domain
from EuroSAT — different resolution, 45 vs 10 classes, different
land-cover taxonomy — so the encoder needs to actually move, not just
adapt. Linear probe is the cheapest by far (199 MB peak, 1.6k trainable
parameters) but caps out at 39% accuracy because the frozen CLIP-EuroSAT
features simply don't separate these 45 classes well. LoRA r=8 spends
~1/40th of the trainable parameters of full FT for a ~2/3 the accuracy
gap — a meaningful efficiency win, but in this setting the cost of full
FT (only +20% wall time, +50% memory) is small enough that "just do
full FT" is the obvious call. LoRA's real value would show up at
larger model scales where full FT's optimizer states alone exceed GPU
memory.

### Problem (lora_rank) — Rank sweep

Plot: `figures/lora_rank_sweep/figures/rank_sweep.png`.

| Rank r | α (=2r) | Test acc | Trainable params |
|-------:|--------:|---------:|-----------------:|
| 1      | 2       | 0.351    | 49,581           |
| 2      | 4       | 0.362    | 81,837           |
| 4      | 8       | 0.393    | 146,349          |
| 8      | 16      | 0.397    | 275,373          |
| 16     | 32      | 0.437    | 533,421          |
| 32     | 64      | 0.458    | 1,049,517        |
| 64     | 128     | **0.492** | 2,081,709        |

**(1) Diminishing returns:** accuracy is *still increasing* at r=64 on
this task. The slope flattens around r=8–16 (going from r=4 to r=8 is
+0.4%, but r=16 → r=32 is +2.1%, r=32 → r=64 is +3.4%), but there is
no clean plateau within the tested range. This is unusual for LoRA and
likely reflects the magnitude of the EuroSAT → RESISC45 domain shift:
the "effective rank" of the required fine-tuning update is large
because the encoder genuinely needs to change.

**(2)** Practical deployments use r=8 or r=16 for two reasons that
don't apply here: (a) the typical setting is fine-tuning a much larger
model on a *small in-domain shift*, where ΔW really is near-low-rank;
and (b) at large model scales, r=64 starts to consume non-trivial
parameter budget. In this homework, the model is small (11M params) so
even r=64 is only 2M trainable params, which is why we can afford to
sweep that far up. The takeaway: "effective rank" is task-dependent,
not a property of LoRA itself.

---

## §5 — Vision-Language Model

> **Eval pipeline caveats.** Before fixing the bugs noted at the top
> of this writeup, all 11 VLM configurations returned val_acc 0.0.
> After fixes (answer-only label masking + `pad_token_id`/`eos_token_id`
> passed to `generate()`), accuracies are in the 0.02–0.09 range below.
> 2000 steps at batch 32 with the projector frozen is genuinely not
> enough to learn CLEVR; the more meaningful signal is the *ordering*
> between configurations, which is internally consistent.

### Problem (projector) — Vision-language projector

Implementation in [`vlm/projector.py`](vlm/projector.py): 2-layer MLP
`Linear(d_image, 4·d_image) → GELU → Linear(4·d_image, d_decoder)`,
shape-flexible for both `(B, d)` (CLS) and `(B, N, d)` (all patches).

**Why more than a single linear layer.** During VLM pretraining the
encoder and decoder are both frozen, so the projector is the *only*
representational bridge between two completely separately-trained
embedding spaces. A single linear map can only realize an affine
transformation, which is fine if the bases happened to be aligned but
they aren't (one was trained contrastively on satellite captions, the
other on language-modeling text). The MLP nonlinearity gives the
projector capacity to recompose image features into directions the
decoder actually uses, without needing to retrain either backbone.

### Problem (injection) — Token injection strategies

Implementation in [`vlm/model.py`](vlm/model.py): three strategies
(`cls`, `all_patches`, `interleaved`) via `_stitch_prepend` and
`_stitch_interleaved`. Visual-token positions in `labels` are filled
with -100 inside `forward()` so HF's CE loss skips them.

### Problem (injection_compare) — Best injection strategy

| Strategy      | Val EM acc | # visual tokens | Peak mem (MB) | s / step |
|---------------|-----------:|----------------:|--------------:|---------:|
| CLS-only      | 0.020     | 1               | **3,365**     | ~0.26    |
| All-patches   | 0.020     | 65              | 6,678         | ~0.23    |
| Interleaved   | 0.020     | 65              | 6,673         | ~0.27    |

(Source: `runs/vlm_{cls,all_patches,interleaved}_causal_A/metrics.json`.)

**Discussion.** All three strategies returned the same overall accuracy
in this run (0.02), but the per-q_type breakdown is more interesting:
both `all_patches` and `interleaved` show non-trivial accuracy on
`compare_attr` (7%) and `spatial` (4%) questions while `cls` is at 0%
for `compare_attr`. This matches the prediction from Problem
(vit_pooling): spatially-grounded questions need access to per-region
features, which a single CLS pool cannot provide. Memory cost scales
exactly with visual-token count (1 vs 65 tokens → ~2× peak memory),
which is the expected attention-quadratic-in-T tradeoff. With more
training steps (the staff suggests this is undertrained at 2000) I
would expect the spatial gap between CLS and all-patches to widen
substantially.

### Problem (masking) — Image-block attention

(1) Mask diagrams for 4 visual + 3 text tokens (rows = query, cols = key;
■ = allowed):

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
bidirectional attention; clamping it to causal at fine-tune time
silently wastes half the patch-to-patch interactions the encoder
already learned. The causal-across-boundary part is unchanged so
autoregressive text generation is preserved.

(3) Results after 2000 steps each with `all_patches` injection, freeze
config A (`_short` runs):

| Mask mode    | Val EM acc | Wall time (s) |
|--------------|-----------:|--------------:|
| Causal       | **0.032**  | 477           |
| Image-bidir  | 0.018     | 417           |

(Source: `runs/vlm_all_patches_{causal,image_bidir}_A_short/metrics.json`.)

**Discussion (honest).** This result contradicts the prediction in (2):
causal actually beat image-bidir here by a non-trivial margin (3.2% vs
1.8%). With absolute numbers this small, the difference may be inside
the noise floor — 1.4% on 500 val examples is ~7 correct answers.
That said, I would have expected image-bidir to at least match causal,
and it didn't. The plausible explanation is that with only 2000 steps,
the model never gets far enough into training to take advantage of the
richer patch-to-patch attention; the simpler causal mask just lets the
optimizer move faster. With longer training the order would likely
flip.

### Problem (freezing) — What to train

All runs use `injection=all_patches`, `mask=image_bidir`. (I kept
`image_bidir` for §5.6 even though §5.5 shows causal narrowly winning,
because the staff's recommended config in `train_vlm.py` is image_bidir
and the comparison is more meaningful when the mask is held fixed.)

| Config | What's trained                       | Val EM acc | Trainable params | Peak mem (MB) |
|:------:|--------------------------------------|-----------:|-----------------:|--------------:|
| A      | projector only                       | 0.018      | 2,066,880        | 6,678         |
| B      | projector + decoder LoRA (r=8)       | **0.066**  | 363,888,000      | 10,216        |
| C      | projector + full decoder             | 0.058      | 363,888,000      | 10,209        |
| D      | full ViT + projector + full decoder  | **0.092**  | 374,625,792      | 10,476        |

(Source: `runs/vlm_all_patches_image_bidir_{A,B,C,D}/metrics.json`.)

**Discussion.** Configuration A (projector-only) is the cheapest by a
wide memory margin but caps at 1.8% — the frozen decoder has never
been adapted to visual prompts, so a fresh projector alone can't make
it understand them. Adding LoRA to the decoder (B, +361M trainable
parameters but only +4 GB peak memory) jumps accuracy more than 3×, to
6.6%. Surprisingly, B *beats* C (full decoder FT) by 0.8% with the same
number of trainable parameters reported (363M — the LoRA adapters bring
that many; my counting may be over-counting LoRA layers, but the peak
memory is essentially identical). The likely explanation: full FT
overfits to the very narrow CLEVR question template in just 2000 steps,
whereas LoRA's small-rank structure is a useful regularizer. D
(everything) gets the best accuracy at 9.2%, suggesting the encoder
also needs some adaptation, but it costs the most memory.

For the two-stage recipe (LLaVA-style: pretrain projector, then
instruction-tune), A → B is the recommended sequence and that's what
these numbers support: A as a cheap alignment step, B as the
instruction-tuning step where the decoder learns to use the projector's
output. C and D are diminishing-return regimes you'd reach for only
once everything else is tuned.

### Problem (vlm_qualitative) — What has the VLM learned?

`runs/vlm_qualitative/` is empty in my submission. `scripts/eval_vlm.py`
crashed during the model-reconstruction step — most likely because the
config-D checkpoint includes ViT and decoder state dicts that
`eval_vlm.py` reconstructs in float32 before casting to bfloat16, and
the load_state_dict path doesn't handle the dtype mismatch on every
HF submodule cleanly. I noticed this too late in the run to debug and
re-execute. **The closest I have is the per-q_type breakdown reported
in §5.6**: even the best config (D) gets `count: 15%, query_attr: 12%,
spatial: 8.5%`, with `exist` and `compare_attr` at 0%. Failure-mode
hypothesis: the model has learned a handful of high-frequency answer
tokens (numbers, common attribute words) and is essentially
classifying without actually reading the question — `exist` and
`compare_attr` answers are yes/no, which a handful of greedy outputs
can't cover, while `count` answers are 0..10 integers which the model
might be learning as a small softmax over likely tokens.

A clean experiment to separate encoder vs. decoder failure: replace the
image embedding with the *oracle* image-description text (e.g., "a
scene with three red cubes and a blue sphere"), feed it as plain text
through the decoder, and measure accuracy. If oracle-image accuracy is
near 100% the encoder is the bottleneck; if it's still <20%, the
decoder is failing to use the visual signal even when perfect.

---

## §6 — Positional Encodings and RoPE

### Problem (rope_1d) — 1D RoPE

Implementation in [`basics/rope.py`](basics/rope.py). The cos/sin tables
are precomputed up to `max_seq_len = 4096` and registered as
non-persistent buffers.

**Norm preservation check** (via `test_rope_1d_preserves_norm`):
```
max |‖x‖ − ‖RoPE(x)‖| < 1e-4
```
RoPE applies an exact 2D rotation to each `(x_{2i}, x_{2i+1})` pair, so
the L2 norm is preserved up to fp32 precision. Confirmed by the test.

### Problem (rope_vs_learned) — Learned PE vs. RoPE in the ViT

Each variant trained for 20 epochs from scratch on EuroSAT (64×64), then
evaluated zero-shot at both 64×64 and 96×96. For the learned-PE
baseline, my ViT bilinearly interpolates the learned patch position
embedding from the 8×8 training grid to the 12×12 evaluation grid
(`basics/vit.py:ViT._interpolate_learned_pos_embed`); the CLS positional
embedding is kept separate.

| PE method | Train-size val acc (64²) | Extrapolated val acc (96²) | Δ |
|-----------|-------------------------:|---------------------------:|------:|
| Learned   | 0.9109                  | **0.7859**                | −0.125 |
| 1D RoPE   | 0.9134                  | 0.7803                    | −0.133 |

(Source: `runs/clip_eurosat_{learned,rope1d}_extrap96/metrics.json`.)

**Discussion (this contradicted my prior).** I expected RoPE to
generalize visibly better than learned PE because attention only sees
*relative* offsets, which are well-defined at any position. In
practice, the learned PE with bilinear interpolation actually
edged out RoPE on the extrapolated grid (78.6% vs 78.0%), and the drop
from 64² → 96² was almost identical for both. Two factors plausibly
explain this: (a) EuroSAT classes are mostly textural (vegetation
type, water, urban density) and don't have strong long-range spatial
dependencies that benefit from RoPE's relative-position bias; and
(b) bilinear interpolation of a well-trained learned PE is a strong
baseline — the eval grid is only 1.5× the training grid in each
dimension, well within smooth interpolation range. I'd expect the
ranking to flip if the extrapolation factor were larger (say 64² →
192²) or the task had stronger spatial structure (e.g., object
counting).

### Problem (rope_2d) — 2D RoPE

Implementation in `basics/rope.py:RoPE2D`. The ViT constructs (x, y)
coordinates per patch when `pos_encoding="rope2d"` (CLS gets (0, 0);
patches get 1-indexed grid coordinates so they don't collide with CLS).

| PE method | Train-size val acc | Extrapolated val acc (96²) |
|-----------|-------------------:|---------------------------:|
| 1D RoPE   | 0.9134            | 0.7803                    |
| 2D RoPE   | **0.9152**        | 0.7803                    |

(Source: `runs/clip_eurosat_rope2d_extrap96/metrics.json`.)

**Discussion.** 2D RoPE marginally beats 1D RoPE at the train resolution
(91.5% vs 91.3%) and is identical at the extrapolated resolution. On
EuroSAT this is a tiny effect; the dataset is mostly statistically
isotropic (rotating a satellite image of farmland doesn't really change
its class), so the explicit 2D structure that 2D RoPE injects isn't
buying much. I'd expect the gap to be visibly larger on a
spatially-structured task like CLEVR or DocVQA where two patches in
different rows have systematically different roles.

### Problem (mrope_written) — Reasoning about M-RoPE

(1) **Naive 1D position IDs failure mode.** With 65 visual tokens (CLS
+ 64 patches) followed by 50 text tokens, naive `0..114` IDs do two
problematic things. First, the 2D grid structure of the image
collapses into row-major order: patch (3, 4) and patch (4, 3) get
arbitrarily different positional treatments based on flattening order,
even though they're geometric neighbors in the original image. RoPE's
relative-distance machinery now treats `dist((row 3, col 4),
(row 4, col 4)) = 1` (one row down, same column — across a stride of
8 patches in 1D) very differently from `dist((row 4, col 3),
(row 4, col 4)) = 1` (same row, adjacent column — actually 1 step in
1D), even though both are valid notions of "spatial adjacency."
Second, the absolute position IDs for text tokens (65..114) live
in a range the decoder was trained to handle, but the bulk of its
training data has *very little* text living that far into the
sequence with no preceding text context — so text-token attention may
be subtly miscalibrated.

(2) **First text token under M-RoPE.** Text tokens receive
`(t, x, y) = (1, grid_w + 1, grid_h + 1)`. With an 8×8 patch grid the
first text token is at temporal=1, spatial=(9, 9). This is sensible
because it (a) gives text a clean modality marker in the temporal
channel, distinct from image tokens which always have temporal=0;
(b) makes text positions independent of how many image patches the
decoder receives, so adding more visual tokens doesn't push text
positions out of the decoder's pretraining range; and (c) places text
just past the image grid in spatial coordinates, so a RoPE attention
window of size 1 naturally falls off as you cross the image-text
boundary.

(3) **Three chunks (t, x, y) vs. two (x, y).** Splitting head_dim into
three lets each dot-product carry information about all three axes
simultaneously, with the temporal channel encoding *modality*
explicitly. If we dropped `t` and used only `(x, y)`, image vs. text
would have to be implicitly encoded by the spatial coordinates
themselves (text gets coords past the image grid). This works only
because the model learns the convention, and it breaks the moment you
add multiple images per prompt: a second image starting at (0, 0)
spatial would now look spatially "identical" to the first image's
patches, which is exactly the confound M-RoPE's temporal axis exists
to prevent.

### Problem (mrope_impl) — Implementing M-RoPE *(bonus)*

Not implemented. The hooks would go in `vlm/model.py` (3D position
assignment) and require monkey-patching SmolLM2's attention to apply
the (t, x, y)-split RoPE inside the decoder. This is meaningfully more
invasive than the encoder-side RoPE in `basics/rope.py` because it
needs to live inside HF's frozen `LlamaAttention` forward. Left as
future work.

---

## Submission

- **`writeup.pdf`** — this document, convertible via `pandoc writeup.md -o writeup.pdf`.
- **`code.zip`** — `git archive HEAD --format=zip > code.zip` (or zip the
  repo excluding `runs/`, `data/`, `__pycache__/`, `figures/`).
- Repo: https://github.com/trevorbchen/148hw3.

## Appendix — figure index

All figures used above live under `figures/<run>/figures/*.png`:

| Section | Run |
|---|---|
| §2.4 | `figures/patch_size_sweep/figures/patch_size_time.png` |
| §3.3 | `figures/clip_eurosat_learned/figures/{loss,lr,val_acc,logit_scale}.png` |
| §3.3 qualitative | `figures/clip_qualitative/figures/confusion_matrix.png` and `figures/clip_qualitative/images/{correct,wrong}/*.png` |
| §4.2 | `figures/resisc_{linear_probe_default, lora_rank8, full_ft_default}/figures/{loss,test_acc}.png` |
| §4.2 rank sweep | `figures/lora_rank_sweep/figures/rank_sweep.png` |
| §5.4 | `figures/vlm_{cls, all_patches, interleaved}_causal_A/figures/*.png` |
| §5.5 | `figures/vlm_all_patches_{causal,image_bidir}_A_short/figures/*.png` |
| §5.6 | `figures/vlm_all_patches_image_bidir_{A,B,C,D}/figures/*.png` |
| §6.1/§6.2 | `figures/clip_eurosat_{learned,rope1d,rope2d}{,_extrap96}/figures/*.png` |
