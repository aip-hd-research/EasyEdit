from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .key import compute_key
from .value import compute_value
from .compute_mom2_inv import get_inv_mom2
from .e_rome_hparams import E_ROMEHyperParams


def apply_e_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: E_ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    
    if len(request) != 1:
        raise NotImplementedError("Only single requests are supported")
    request = request[0]

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    delta = execute_e_rome(model, tok, request, hparams)

    with torch.no_grad():
        delta_u, delta_v = delta
        w_name =  f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
        upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
        w = nethook.get_parameter(model, w_name)
        upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
        if return_orig_weights:
            weights_copy[w_name] = w.detach().clone()

        w[...] += upd_matrix

        print(f"New weights successfully inserted into {w_name}")

    return model, weights_copy


def execute_e_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: E_ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"][0] != ' ':
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if "{}" not in request["prompt"]:
        if not request["prompt"].count(request["subject"]) == 1:
            raise ValueError(f"Subject:{request['subject']} should exist exactly once in prompt: {request['prompt']}. Alternatively you can provide just the subject location using the format specifier '{{}}'")

        request["prompt"] = request["prompt"].replace(request["subject"], "{}")
    
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    key: torch.Tensor = compute_key(
        model,
        tok,
        request,
        hparams,
    )

    value: torch.Tensor = compute_value(
        model,
        tok,
        request,
        hparams,
    )

    # Retrieve weights that user desires to change
    weights_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
    weights = nethook.get_parameter(model, weights_name) 
    weights = upd_matrix_match_shape(weights, (value.shape[0], key.shape[0]))

    with torch.no_grad():
      if hparams.mom2_adjustment:
        mom2_inv: torch.Tensor = get_inv_mom2(
            model,
            tok,
            hparams.rewrite_module_tmp.format(hparams.layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir=hparams.stats_dir,
        )

        left_vector = numerically_checked_product(key, mom2_inv)
      else:
        left_vector = key

      print(value.dtype, weights.dtype, key.dtype, left_vector.dtype)
      right_vector = (value - weights @ key) / (left_vector @ key) 
      delta = (
         left_vector.detach(),
         right_vector.detach(),
      )

    print(f"Delta successfully computed for {hparams.rewrite_module_tmp.format(hparams.layer)}")

    return delta


def numerically_checked_product(key: torch.Tensor, mom2_inv: torch.Tensor) -> torch.Tensor:
    u = mom2_inv
    old_dtype = key.dtype
    old_device = key.device
    u = u @ key.unsqueeze(1).to(u.device, dtype=u.dtype)
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


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )

