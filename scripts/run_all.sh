#!/usr/bin/env bash
# Drive every experiment for HW3. Designed for Colab (A100). Skips a phase
# if its primary metrics.json already exists, so you can resume after a crash.
#
# Usage:
#   bash scripts/run_all.sh           # everything
#   PHASES="vit clip" bash scripts/run_all.sh   # subset
#
# Phases (space-separated, in order): vit clip clip_qual resisc resisc_sweep
#                                     vlm_inject vlm_mask vlm_freeze vlm_qual
#                                     rope rope_extrap

set -euo pipefail

PHASES="${PHASES:-vit clip clip_qual resisc resisc_sweep vlm_inject vlm_mask vlm_freeze vlm_qual rope rope_extrap}"
PRETRAINED_VIT="${PRETRAINED_VIT:-runs/clip_eurosat_learned/best.pt}"

run() {
    echo
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
    "$@"
}

skippable() {
    local sentinel="$1"; shift
    if [[ -f "$sentinel" ]]; then
        echo "[skip] $sentinel exists"
        return 0
    fi
    return 1
}

# ----- §2 ViT patch-size sweep ----------------------------------------------
if [[ "$PHASES" == *vit* ]]; then
    if ! skippable runs/patch_size_sweep/metrics.json; then
        run python scripts/patch_size_sweep.py
    fi
fi

# ----- §3.3 CLIP pretraining (learned PE, default) --------------------------
if [[ "$PHASES" == *clip* ]]; then
    if ! skippable runs/clip_eurosat_learned/metrics.json; then
        run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml \
            --pos-encoding learned
    fi
fi

# ----- §3.3 zero-shot qualitative -------------------------------------------
if [[ "$PHASES" == *clip_qual* ]]; then
    if ! skippable runs/clip_qualitative/metrics.json; then
        run python scripts/clip_zeroshot_qualitative.py \
            --checkpoint "$PRETRAINED_VIT"
    fi
fi

# ----- §4.2 RESISC adaptation comparison ------------------------------------
if [[ "$PHASES" == *resisc* && "$PHASES" != *resisc_sweep* ]]; then
    if ! skippable runs/resisc_linear_probe_default/metrics.json; then
        run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
            --method linear_probe --pretrained "$PRETRAINED_VIT"
    fi
    if ! skippable runs/resisc_lora_rank8/metrics.json; then
        run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
            --method lora --rank 8 --alpha 16 --pretrained "$PRETRAINED_VIT"
    fi
    if ! skippable runs/resisc_full_ft_default/metrics.json; then
        run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
            --method full_ft --pretrained "$PRETRAINED_VIT"
    fi
fi

# ----- §4.2 LoRA rank sweep -------------------------------------------------
if [[ "$PHASES" == *resisc_sweep* ]]; then
    if ! skippable runs/lora_rank_sweep/metrics.json; then
        run python scripts/lora_rank_sweep.py \
            --config configs/lora_resisc.yaml \
            --pretrained "$PRETRAINED_VIT" \
            --ranks 1 2 4 8 16 32 64 \
            --skip-existing
    fi
fi

# ----- §5.4 injection comparison (cls / all_patches / interleaved) ----------
if [[ "$PHASES" == *vlm_inject* ]]; then
    for inj in cls all_patches interleaved; do
        sentinel="runs/vlm_${inj}_causal_A/metrics.json"
        if ! skippable "$sentinel"; then
            run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
                --pretrained-vit "$PRETRAINED_VIT" \
                --injection "$inj" --mask-mode causal --freeze-config A
        fi
    done
fi

# ----- §5.5 masking comparison (best inject; default all_patches, 500 steps) -
if [[ "$PHASES" == *vlm_mask* ]]; then
    for mask in causal image_bidir; do
        sentinel="runs/vlm_all_patches_${mask}_A_short/metrics.json"
        if ! skippable "$sentinel"; then
            run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
                --pretrained-vit "$PRETRAINED_VIT" \
                --injection all_patches --mask-mode "$mask" --freeze-config A \
                --output-dir "runs/vlm_all_patches_${mask}_A_short"
        fi
    done
fi

# ----- §5.6 freeze configs (best inject + mask, 1500 steps) -----------------
if [[ "$PHASES" == *vlm_freeze* ]]; then
    for cfg in A B C D; do
        sentinel="runs/vlm_all_patches_image_bidir_${cfg}/metrics.json"
        if ! skippable "$sentinel"; then
            run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
                --pretrained-vit "$PRETRAINED_VIT" \
                --injection all_patches --mask-mode image_bidir --freeze-config "$cfg"
        fi
    done
fi

# ----- §5.7 qualitative VLM eval --------------------------------------------
if [[ "$PHASES" == *vlm_qual* ]]; then
    BEST_VLM="${BEST_VLM:-runs/vlm_all_patches_image_bidir_A/best.pt}"
    if ! skippable runs/vlm_qualitative/metrics.json; then
        run python scripts/eval_vlm.py --checkpoint "$BEST_VLM" \
            --num-examples 10 --save-images
    fi
fi

# ----- §6.1 / §6.2 RoPE pretraining -----------------------------------------
if [[ "$PHASES" == *rope* && "$PHASES" != *rope_extrap* ]]; then
    for pe in rope1d rope2d; do
        if ! skippable "runs/clip_eurosat_${pe}/metrics.json"; then
            run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml \
                --pos-encoding "$pe"
        fi
    done
fi

# ----- §6.1 length extrapolation eval ---------------------------------------
if [[ "$PHASES" == *rope_extrap* ]]; then
    for pe in learned rope1d rope2d; do
        ckpt="runs/clip_eurosat_${pe}/best.pt"
        if [[ ! -f "$ckpt" ]]; then
            echo "[skip] missing $ckpt — train ${pe} first"
            continue
        fi
        # Re-run the val/test pass with --extrapolation-img-size 96
        # Note: we have to re-run training; simplest is to wrap the eval logic
        # standalone, but for now this serves as a TODO marker.
        echo "TODO: standalone extrapolation eval for $pe (re-run pretrain_clip with --extrapolation-img-size 96)"
    done
fi

echo
echo "============================================================"
echo "  ALL PHASES DONE. Inspect runs/*/metrics.json + figures/."
echo "============================================================"
