"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions are filled with -100 internally
                        so they're masked out of HF's loss.
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _encode_images(self, images: torch.Tensor, injection: InjectionMode) -> torch.Tensor:
        """Run ViT and projector. Returns (B, N_vis, d_decoder)."""
        if injection == "cls":
            visual_features = self.vit(images)                     # (B, d_image)
        else:
            visual_features = self.vit(images, return_all_tokens=True)  # (B, N+1, d_image)
        visual_embeds = self.projector(visual_features)            # (B, N_vis, d_decoder)
        return visual_embeds

    def _stitch_prepend(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Prepend visual tokens to text tokens (cls / all_patches modes)."""
        B, n_visual, _ = visual_embeds.shape
        text_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T_text, d)
        stitched = torch.cat([visual_embeds, text_embeds], dim=1)
        visual_attn = torch.ones(
            B, n_visual, dtype=attention_mask.dtype, device=attention_mask.device
        )
        stitched_attn = torch.cat([visual_attn, attention_mask], dim=1)
        if labels is not None:
            visual_labels = torch.full(
                (B, n_visual), -100, dtype=labels.dtype, device=labels.device
            )
            stitched_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            stitched_labels = None
        return stitched, stitched_attn, stitched_labels

    def _stitch_interleaved(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace each occurrence of self.image_token_id in input_ids with the
        full visual patch sequence."""
        if self.image_token_id is None:
            raise ValueError("image_token_id must be set for 'interleaved' injection")
        B, n_visual, d = visual_embeds.shape
        text_embeds = self.decoder.get_input_embeddings()(input_ids)

        new_embeds, new_attn, new_labels = [], [], []
        for b in range(B):
            ids = input_ids[b]
            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) != 1:
                raise ValueError(
                    f"Expected exactly one <image> token per sequence, got {len(img_pos)}"
                )
            pos = img_pos.item()

            new_embeds.append(
                torch.cat(
                    [text_embeds[b, :pos], visual_embeds[b], text_embeds[b, pos + 1 :]],
                    dim=0,
                )
            )
            new_attn.append(
                torch.cat(
                    [
                        attention_mask[b, :pos],
                        torch.ones(n_visual, dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[b, pos + 1 :],
                    ],
                    dim=0,
                )
            )
            if labels is not None:
                lb = labels[b]
                visual_labels = torch.full(
                    (n_visual,), -100, dtype=lb.dtype, device=lb.device
                )
                new_labels.append(torch.cat([lb[:pos], visual_labels, lb[pos + 1 :]], dim=0))

        stitched = torch.stack(new_embeds, dim=0)
        stitched_attn = torch.stack(new_attn, dim=0)
        stitched_labels = torch.stack(new_labels, dim=0) if labels is not None else None
        return stitched, stitched_attn, stitched_labels

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        visual_embeds = self._encode_images(images, injection)
        n_visual = visual_embeds.shape[1]

        if injection in ("cls", "all_patches"):
            stitched, stitched_attn, stitched_labels = self._stitch_prepend(
                visual_embeds, input_ids, attention_mask, labels
            )
        else:  # interleaved
            stitched, stitched_attn, stitched_labels = self._stitch_interleaved(
                visual_embeds, input_ids, attention_mask, labels
            )

        decoder_kwargs = dict(inputs_embeds=stitched, labels=stitched_labels)

        if mask_mode == "image_bidir" and injection in ("cls", "all_patches"):
            B, T_total, _ = stitched.shape
            n_text = T_total - n_visual
            mask4d = build_image_bidir_mask(
                n_visual, n_text, stitched.device, stitched.dtype
            ).expand(B, -1, -1, -1)
            decoder_kwargs["attention_mask"] = mask4d
        else:
            decoder_kwargs["attention_mask"] = stitched_attn

        output = self.decoder(**decoder_kwargs)
        return {"loss": output.loss, "logits": output.logits}

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        device = next(self.decoder.parameters()).device
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        visual_embeds = self._encode_images(images, injection)
        if injection in ("cls", "all_patches"):
            stitched, stitched_attn, _ = self._stitch_prepend(
                visual_embeds, input_ids, attention_mask, None
            )
        else:
            stitched, stitched_attn, _ = self._stitch_interleaved(
                visual_embeds, input_ids, attention_mask, None
            )

        # max_new_tokens may also be present in gen_kwargs (from a config); the
        # named parameter wins, so drop it from kwargs to avoid TypeError.
        gen_kwargs.pop("max_new_tokens", None)
        out_ids = self.decoder.generate(
            inputs_embeds=stitched,
            attention_mask=stitched_attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
