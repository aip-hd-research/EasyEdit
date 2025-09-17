import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util.globals import *

from .layer_stats import layer_stats
from .compute_mom2_inv import get_inv_mom2
from .e_rome_hparams import E_ROMEHyperParams


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: E_ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")

        if hparams.original_implementation or hparams.enable_random_prefix_keys:
            cur_repr = repr_tools.get_reprs_at_word_tokens(
                context_templates=[
                    templ.format(request["prompt"]) for templ in context_templates
                ],
                words=[word for _ in range(len(context_templates))],
                subtoken=hparams.fact_token[len("subject_") :],
                **word_repr_args,
            ).mean(0)
        else:
            cur_repr = repr_tools.get_reprs_at_word_tokens(
                subtoken=hparams.fact_token[len("subject_") :],
                context_templates=[request["prompt"]],
                words=[word],
                **word_repr_args,
            ).squeeze()

    elif hparams.fact_token == "last":
        if hparams.enable_prompt_keys:
            raise ValueError(
                "Last token projection not supported with prompt_keys. Use subject_ instead."
            )
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr

    if hparams.mom2_adjustment:
        u = get_inv_mom2(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir=hparams.stats_dir,
        )

        old_dtype = cur_repr.dtype
        old_device = cur_repr.device
        u = u @ cur_repr.unsqueeze(1).to(u.device, dtype=u.dtype)
        u = u.squeeze()
        u = u / u.norm()
        sparsity_before = ((u != 0).sum() / u.nelement()).item()
        u = u.to(old_device, dtype=old_dtype)
        sparsity_after = ((u != 0).sum() / u.nelement()).item()
        assert sparsity_after >= 0.9 * sparsity_before, (
            "Dtype conversion dropped to many nonzero values."
        )
        assert not u.isnan().any() and not u.isinf().any(), (
            "Aberrant behaviour detected. Check dtypes."
        )

    return u
