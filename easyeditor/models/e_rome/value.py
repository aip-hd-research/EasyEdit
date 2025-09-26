from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.repr_tools import get_words_idxs_in_templates
from ...util import nethook

from .context_templates import get_context_templates
from .e_rome_hparams import E_ROMEHyperParams

def rebatch(batch, batch_size):
    n = len(next(iter(batch.values())))
    return [
        {k: v[idx : min(idx + batch_size, n)] for k, v in batch.items()}
        for idx in range(0, n, batch_size)
    ]

def compute_value(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        request: Dict,
        hparams: E_ROMEHyperParams,
        per_device_batch_size: int = 1, # Very conservative default to minimize VRAM
    ) -> torch.Tensor:

    target_pretokenized = tok.tokenize(request["target_new"])

    # strip potential bos tokens
    if target_pretokenized[0] == tok.bos_token or target_pretokenized[0] == tok.unk_token:
        target_pretokenized = target_pretokenized[1:]
 

    context_templates = get_context_templates(
        model,
        tok,
        hparams.context_template_length_params,
    )

    edit_prompt_templates = [context.format(request["prompt"]) for context in context_templates]

    edit_prompts_pretokenized = [
        (
            tok.tokenize(template.format(request["subject"]))
            + target_pretokenized[:-1]
        ) # drop last token - loss calculation is left shifted by one
        for template in edit_prompt_templates
    ]

    kl_prompt_templates = ["{} is a"]
    
    kl_prompts_pretokenized = [tok.tokenize(template.format(request["subject"])) for template in kl_prompt_templates]

    target_ids = tok(
        [target_pretokenized], 
        is_split_into_words=True, 
        return_tensors="pt"
    )["input_ids"][0].to(device=f"cuda:{hparams.device}")

    model_inputs = tok(
        edit_prompts_pretokenized + kl_prompts_pretokenized,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True
    ).to(device=f"cuda:{hparams.device}")

    target_idxs = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(edit_prompts_pretokenized), *model_inputs["input_ids"].shape[1:]
    )

    lookup_idxs = get_words_idxs_in_templates(
        tok=tok,
        context_templates=edit_prompt_templates + kl_prompt_templates,
        words=[request["subject"]]*len(edit_prompt_templates + kl_prompt_templates),
        subtoken=hparams.fact_token[len("subject_"):]
    )
    assert all(len(idxs) == 1 for idxs in lookup_idxs), "Expected exactly one lookup location per prompt"
    lookup_idxs = [idx[0] for idx in lookup_idxs]


    if tok.padding_side == "right":
      for i, idx in enumerate(target_idxs):
            sequence_len = model_inputs["attention_mask"][i].sum()
            idx[sequence_len - len(target_ids) : sequence_len] = target_ids
    elif tok.padding_side == "left":
      target_idxs[:, -len(target_ids) :] = target_ids
      for i in range(len(lookup_idxs)):
        lookup_idxs[i] += (
          (model_inputs["attention_mask"][i] != 1).sum().item()
        )  # Add padding to lookup idxs
    else:
      raise ValueError(f"Unknown padding side {tok.padding_side}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    n_embd = None
    if hasattr(model.config, "n_embd"):
        n_embd = model.config.n_embd
    elif hasattr(model.config, "hidden_size"):
        n_embd = model.config.hidden_size
    else:
        assert False, "No hidden dimension found in config"
    delta = torch.rand(
        (n_embd,), 
        requires_grad=True, 
        device=f"cuda:{hparams.device}",
    )  # ensures different trajectories on different random seeds
    delta.data *= 1e-5  # scale down to epsilon

    target_init, kl_distr_init = None, None

    # Since the nethook library works with hook functions
    # with predetermined inputs, we can't pass additional
    # per-sample data along.
    # In particular there is no direct way to access the lookup idxs.
    # The only remedy is to track the number of passes through the model
    # to determine which batch we are in. 
    exec_count = 0  # I hate everything about this

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        # This thing induces statefullness all over this entire implementation.
        nonlocal target_init, exec_count

        if cur_layer == hparams.mlp_module_tmp.format(hparams.layer):
            # Store initial value of the vector of interest
            if target_init is None:
                # Initial value is recorded for the clean sentence
                # This very tightly couples this code to the context_templates
                # Concretely it assumes that index zero is the empty prefix template
                # TODO: Decouple this!
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
    
            for i, idx in enumerate(
                lookup_idxs[exec_count : exec_count + len(cur_out)]
            ):
                cur_out[i, idx, :] += delta.to(cur_out.device)
            exec_count += len(cur_out)
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        exec_count = 0

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.mlp_module_tmp.format(hparams.layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = torch.concat(
                [
                    model(**batch).logits
                    for batch in rebatch(model_inputs, per_device_batch_size)
                ]
            )

        # Compute distribution for KL divergence
        kl_logits = torch.stack(
            [
                logits[i - len(kl_prompt_templates), idx, :]
                for i, idx in enumerate(lookup_idxs[-len(kl_prompt_templates) :])
            ],
            dim=0,
          )
        kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
        if kl_distr_init is None:
           kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(target_idxs != -100, target_idxs, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (target_idxs != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)

        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = (
            hparams.v_weight_decay
                * (torch.norm(delta) / torch.norm(target_init.to(f"cuda:{hparams.device}"))) ** 2
        )

        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        
    return target_init + delta.to(target_init.device, dtype=target_init.dtype)


    

    
