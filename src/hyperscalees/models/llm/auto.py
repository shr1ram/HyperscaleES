from .tokenizer import GptTokenizer, WorldTokenizer, TinyLlamaTokenizer

from . import rwkv7, tinyllama

from huggingface_hub.constants import HF_HOME
from huggingface_hub import hf_hub_download

from transformers import AutoModelForCausalLM

from pathlib import Path

import pickle

import jax
import jax.numpy as jnp

suffix = ".model"

_default_model_class = {
    "tl1.1B": "BaseTinyLlama",
}

models = {
    "7w0.1B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth")), None),
    "7w0.4B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")), None),
    "7w1.5B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth")), None),
    "7w3B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth")), None),

    "7n0.1B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-168M-20241120-ctx4096.pth")), None),
    "7n0.4B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-421M-20241127-ctx4096.pth")), None),
    "7n1.5B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-1.47B-20241210-ctx4096.pth")), None),

    "7g0.1B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-0.1b-20260129-ctx8192.pth")), None),
    "7g0.4B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-0.4b-20260210-ctx8192.pth")), None),
    "7g1.5B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-1.5b-20260212-ctx8192.pth")), None),
    "7g2.9B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-2.9b-20260131-ctx8192.pth")), None),
    "7g7B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-7.2b-20260131-ctx8192.pth")), None),
    "7g14B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1d-13.3b-20260131-ctx8192.pth")), None),

    "tl1.1B": (tinyllama, TinyLlamaTokenizer,
                (lambda: AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="auto")),
                (lambda: {"n_layer": 22, "n_heads": 32, "n_kv_heads": 4, "head_dim": 64,
                          "hidden_size": 2048, "intermediate_size": 5632,
                          "max_seq_len": 256, "rms_norm_eps": 1e-5, "rope_theta": 10000.0})),
}

def get_model(model_name, dtype=None, model_class="BaseRWKV", verbose=False, reload_cache=False):
    rwkv, tok_cls, model_name_fn, config_fn = models[model_name]
    # Use default model class if available and caller didn't override
    if model_class == "BaseRWKV" and model_name in _default_model_class:
        model_class = _default_model_class[model_name]
    RWKV = getattr(rwkv, model_class)
    rwkv_tokenizer = tok_cls()

    if dtype is None:
        dtype = jnp.float32 if model_name.startswith('m') else jnp.bfloat16
    elif isinstance(dtype, str):
        dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
    if verbose:
        print(dtype)

    path = Path(HF_HOME, "hyperscalees_cache", f"{model_name}_{str(dtype.dtype)}.model")
    if path.is_file() and not reload_cache:
        if verbose:
            print("loading from", path)
        rwkv_full_params = load(path)
    else:
        import torch
        MODEL_NAME = model_name_fn()
        if isinstance(MODEL_NAME, torch.nn.Module):
            rwkv_full_params = MODEL_NAME.state_dict()
        else:
            rwkv_full_params = torch.load(MODEL_NAME, map_location='cpu', weights_only=True)
        config = config_fn() if config_fn is not None else None
        rwkv_full_params = RWKV.load_from_torch(rwkv_full_params, config, dtype=dtype)
        if verbose:
            print("saving to", path)
        save(rwkv_full_params, path, True)
    return RWKV, rwkv_full_params, rwkv_tokenizer

def save(model: any, path: str | Path, overwrite: bool = False):
    """
    Save the Any model as a file given a path.

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param model: The Any model you want to save
    :param path: The path to save the model to
    :param overwrite: Set to true to allow overwriting over existing file
    """
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load(path: str | Path) -> any:
    """
    Read the Any model from a file

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param path: The path to read the model from
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
