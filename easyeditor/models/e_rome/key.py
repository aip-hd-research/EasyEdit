import torch
from .e_rome_hparams import KeyMode

import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util.globals import *

from .layer_stats import layer_stats
from .compute_mom2_inv import get_inv_mom2
from .context_templates import get_context_templates

from .e_rome_hparams import E_ROMEHyperParams

def compute_key(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: E_ROMEHyperParams,       
    ) -> torch.Tensor:
    if (hparams.key_mode == KeyMode.NO_PREFIX or 
            hparams.key_mode == KeyMode.RANDOM_PREFIX):
        return compute_random_prefix_key(
            model,
            tok,
            request,
            hparams,
        )
    if hparams.key_mode == KeyMode.SEMANTIC_INTERSECTION:
        return compute_semantic_intersection_key(
            model,
            tok,
            request,
            hparams,
        )

 
def compute_random_prefix_key(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: E_ROMEHyperParams,       
    ) -> torch.Tensor:
    if not ("subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0):
        raise NotImplementedError("We only support positions relative to the subject")

    context_templates = get_context_templates(
        model,
        tok,
        hparams.context_template_length_params,
    )

    return repr_tools.get_reprs_at_word_tokens(
           context_templates=[
              templ.format(request["prompt"]) for templ in context_templates
           ],
           words=[request["subject"]] * len(context_templates),
           subtoken=hparams.fact_token[len("subject_") :],
           model=model,
           tok=tok,
           layer=hparams.layer,
           module_template=hparams.rewrite_module_tmp,
           track="in",
        ).mean(0)

def compute_semantic_intersection_key(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: E_ROMEHyperParams,       
    ) -> torch.Tensor:

    keys = repr_tools.get_reprs_at_idxs(
        model=model,
        tok=tok,
        contexts=request["key_prompts"],
        idxs=[[-1] for prompt in request["key_prompts"]], 
        layer=hparams.layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )

    intersection = (keys != 0).prod(dim=0)

    print(f"Key shape: {intersection.shape}")
    print(f"Intersection sparsity: {(intersection != 0).sum()}")

    mean = keys.mean(dim=0)

    assert mean.shape == intersection.shape
    print(f"Mean sparsity: {(mean != 0).sum()}")

    return intersection * mean



 
