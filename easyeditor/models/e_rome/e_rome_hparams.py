from dataclasses import dataclass, field
from typing import List
import yaml
from enum import Enum

from ...util.hparams import HyperParams

class KeyMode(Enum):
    NO_PREFIX = 0
    RANDOM_PREFIX = 1
    SEMANTIC_INTERSECTION = 2 

@dataclass
class E_ROMEHyperParams(HyperParams):
    # Method
    layer: int
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    
    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: int
    model_name: str
    transcoder_path: str
    stats_dir: str

    max_length: int = 40
    model_parallel: bool = False
    fp16: bool = False

    # Key calculation
    
    context_template_length_params: List[List[int]] = field(default_factory=lambda: [[5, 10], [10, 10]])
    key_mode: KeyMode = KeyMode.RANDOM_PREFIX 

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        if config["alg_name"] != "E-ROME":
            raise ValueError(f"E_ROMEHyperParams can not load from {hparams_name_or_path}, alg_name is {config['alg_name']}")
        if "key_mode" in config:
            config["key_mode"] = KeyMode[config["key_mode"]]

        if "key_mode" in config and config["key_mode"] != KeyMode.RANDOM_PREFIX:
            config["context_template_length_params"] = []


        return cls(**config)
