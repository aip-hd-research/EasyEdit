import argparse
import os
import sys
from argparse import BooleanOptionalAction

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path

from ...util import nethook
from ...util.globals import *
from ...util.nethook import Trace

from .layer_stats import layer_stats

os.environ["TOKENIZERS_PARALLELISM"] = "False"

CACHE_VERSION = 1  # Used to invalidate old cache files after algorithm update

# In-memory
inv_mom2_cache = {}


def retrieve_from_cache(key, stats_dir):
    global inv_mom2_cache
    if key in inv_mom2_cache:
        return inv_mom2_cache[key]

    filename = os.path.join(stats_dir, f"{key}.pt")
    if not os.path.isfile(filename):
        return None

    value = torch.load(filename, map_location=torch.device("cpu"))
    inv_mom2_cache[key] = value
    return value


def store_in_cache(key, value, stats_dir):
    global inv_mom2_cache
    inv_mom2_cache[key] = value
    os.makedirs(stats_dir, exist_ok=True)
    torch.save(value, os.path.join(stats_dir, f"{key}.pt"))


def inv(m):
    # Inverse that handles dead features gracefully
    with torch.no_grad():
        non_zero_features = torch.diagonal(m).nonzero().squeeze()
        x, y = torch.meshgrid(non_zero_features, non_zero_features)
        m_img = m[x, y]
        m_img_inverse = torch.inverse(
            m_img
        )  # Will fail if samples are linearly dependent
        m_inv = torch.zeros_like(m)
        m_inv[x, y] = m_img_inverse
    return m_inv


def assert_soundness(mom2_inv):
    assert not mom2_inv.isnan().any() and not mom2_inv.isinf().any(), (
        "Inv. cov. matrix failed nan check"
    )
    assert not (mom2_inv == 0).all(), "Inv. cov. matrix failed zero check"


def get_inv_mom2(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    stats_dir: str = "",
    overwrite_cache: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the inverse.
    Caches result for future use both on disk and in-memory.
    """
    logger = logging.getLogger("rome.compute_mom2_inv.get_inv_mom2")

    model_name = model.config._name_or_path.replace("/", "_")
    device = next(model.parameters()).device

    # Key cache of a) the results of a single forward pass up to the target layer.
    # And b) The number of samples used to estimate the second moment
    # Not exactly cryptographically secure but probably good enough for our use case.
    with Trace(
        model, layer_name, retain_input=True, retain_output=False, stop=True
    ) as tr:
        model(input_ids=torch.tensor([[42]], device=device))

    dim = max(nethook.get_parameter(model, f"{layer_name}.weight").shape)

    # This is the minimum viable number of sample to have hopes of inverting the second moment matrix.
    # It is not a sufficient condition for invertibility
    if dim > mom2_n_samples:
        logger.info(
            f"mom2_n_samples needs to be larger than layer dim to enable invertibility of covariance matrix."
            f"Continuing with minimal viable number: mom2_n_samples = dim + 1 = {dim + 1}."
        )
        mom2_n_samples = dim + 1

    # transform input to make it better suited for use as file name later on
    stats_key = (
        "".join(
            str(func(tr.input).abs().detach().cpu().numpy().item()).replace(".", "")[:2]
            for func in [torch.sum, torch.min, torch.max, torch.mean]
        )
        + str(mom2_n_samples / 1_000).replace(".", "")
        + str(sum(tr.input.shape))
    )
    inv_mom2_key = stats_key + str(CACHE_VERSION)

    logger.info(f"Stats key: {stats_key}")
    logger.info(f"Inverse matrix key: {inv_mom2_key}")

    if (
        not overwrite_cache
        and (mom2_inv := retrieve_from_cache(inv_mom2_key, stats_dir)) is not None
    ):
        assert_soundness(mom2_inv)
        return mom2_inv

    logger.info(
        f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
        f"The result will be cached to avoid repetitive computation."
    )

    stat = layer_stats(
        model,
        tok,
        layer_name,
        stats_dir,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        cache_key=stats_key,
        download=False,
    )
    dtype = getattr(torch, mom2_dtype)

    mom2 = stat.mom2.moment().detach()

    # only 32/64 bit floats supported by inverse
    mom2_inv = inv(mom2)

    store_in_cache(inv_mom2_key, mom2_inv, stats_dir)

    logger.info("Got inverse covariance matrix")

    assert_soundness(mom2_inv)
    return mom2_inv


def main(args):
    if args.layer is not None and args.expansion_factor is not None:
        transcoder_dir = os.path.join(
            args.transcoder_lookup_path, f"layer{args.layer}_ef{args.expansion_factor}"
        )
        files = os.listdir(transcoder_dir)
        transcoder = [f for f in files if "final" in f and "feature_sparsity" not in f][
            0
        ]
        args.transcoder = os.path.join(transcoder_dir, transcoder)

    if args.wandb is not None:
        wandb.init(config=vars(args))

    if args.transcoder is not None:
        model, tok = load_model_and_tokenizer_with_TC_layer(
            args.model, args.transcoder, args.layer, device_map=args.device
        )
    else:
        model, tok = load_model_and_tokenizer(args.model, device_map=args.device)

    get_inv_mom2(
        model=model,
        tok=tok,
        layer_name=args.rewrite_module_tmp.format(args.layer),
        mom2_dataset=args.mom2_dataset,
        mom2_n_samples=args.mom2_n_samples,
        mom2_dtype=args.mom2_dtype,
        stats_dir=args.stats_dir,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/nfs/data/shared/CodeLlama-7b-Instruct-hf/model", type=str
    )
    parser.add_argument("--transcoder", default=None, type=str)
    parser.add_argument(
        "--transcoder_lookup_path",
        default="/nfs/data/shared/codellama-transcoders/final",
    )
    parser.add_argument(
        "--stats_dir",
        default="/nfs/data/shared/ROME_Cache/stats",
    )
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expansion_factor", "--ef", default=None, type=int)
    parser.add_argument(
        "--rewrite_module_tmp", default="model.layers.{}.mlp.down_proj", type=str
    )
    parser.add_argument("--mom2_dataset", default="codeparrot/github-code", type=str)
    parser.add_argument("--mom2_n_samples", default=1, type=int)
    parser.add_argument("--mom2_dtype", default="float32", type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--overwrite_cache", action=BooleanOptionalAction)
    parser.add_argument("--wandb", action=BooleanOptionalAction)
    main(parser.parse_args())

