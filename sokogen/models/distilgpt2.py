"""DistilGPT-2 ablation: does natural-language pretraining transfer? (spec 6.3)

The question is sharp here because the target domain shares **zero vocabulary**
with English.  Adapting DistilGPT-2 to our 48-symbol character vocabulary means
discarding its 50257x768 input embedding matrix and its tied LM head and
re-initialising both, so what could possibly transfer is the *body* -- the
attention and MLP blocks -- not any lexical knowledge.

Either outcome is a result.  If it helps, pretrained sequence-modelling
machinery transfers across a vocabulary boundary; if it does not, the 82M
parameters are dead weight against a 10.7M from-scratch model.

The wrapper exposes exactly the ``SokobanLM`` interface -- ``forward(ids,
labels=None, past=None, use_cache=False)`` -- so the constrained decoder and the
evaluation harness treat both models identically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.vocab import MAX_LEN, PAD_ID, VOCAB_SIZE


class DistilGPT2LM(nn.Module):
    """DistilGPT-2 body with a freshly initialised character vocabulary."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.backbone = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.vocab_size = vocab_size

        # Discard the 50257-token embedding matrix and the tied LM head, and
        # re-initialise both for our 48-symbol vocabulary.  This is the whole
        # point of the ablation: nothing lexical can transfer.
        d_model = self.backbone.config.n_embd
        new_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_emb.weight[PAD_ID].zero_()
        self.backbone.set_input_embeddings(new_emb)

        new_head = nn.Linear(d_model, vocab_size, bias=False)
        new_head.weight = new_emb.weight          # keep them tied
        self.backbone.lm_head = new_head
        self.backbone.config.vocab_size = vocab_size
        self.backbone.config.pad_token_id = PAD_ID

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None,
                past=None, use_cache: bool = False):
        out = self.backbone(input_ids=ids, past_key_values=past,
                            use_cache=use_cache)
        logits = out.logits

        loss = None
        if labels is not None:
            # Same one-position shift and -100 masking as SokobanLM, computed
            # here rather than inside HF so both families share one definition.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, out.past_key_values
        return logits, loss


def build_distilgpt2(vocab_size: int = VOCAB_SIZE) -> DistilGPT2LM:
    return DistilGPT2LM(vocab_size)
