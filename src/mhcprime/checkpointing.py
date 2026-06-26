import os
import random
from importlib.resources import files
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from .features import load_processed_feature_table
from .model import get_default_model_params, init_model
from .model_utils import AminoAcidTokenizer


def get_default_checkpoint_path():
    """
    Return the packaged default MHCPrime base checkpoint path.

    Expected packaged location:
        src/mhcprime/checkpoints/mhcprime_base/model_final.pt
    """
    return files("mhcprime").joinpath(
        "checkpoints",
        "mhcprime_base",
        "model_final.pt",
    )


def resolve_device(device=None):
    """
    Resolve a user-provided device argument.

    If device is None, use CUDA when available, otherwise CPU.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def build_mhcprime_model(
    *,
    device=None,
    print_params=False,
    print_check=False,
):
    """
    Build an untrained MHCPrime model using the default released architecture.

    This reconstructs:
        - AminoAcidTokenizer
        - processed amino-acid feature table
        - default model parameter dictionary
        - initialized MHCPrime model

    This function does not load checkpoint weights.
    """
    device = resolve_device(device)

    tokenizer = AminoAcidTokenizer()
    processed_feature_table = load_processed_feature_table()
    feature_names = list(processed_feature_table.columns)

    model_params = get_default_model_params(
        tokenizer=tokenizer,
        processed_feature_table=processed_feature_table,
        feature_names=feature_names,
    )

    model = init_model(
        model_params,
        print_params=print_params,
        print_check=print_check,
        device=device,
    )

    return model, tokenizer, model_params


def extract_model_state_dict(checkpoint):
    """
    Extract a model state_dict from supported checkpoint formats.

    Supported:
        1. raw state_dict
        2. {"model": state_dict}
        3. {"model_state_dict": state_dict}
        4. {"state_dict": state_dict}

    The base MHCPrime checkpoint is expected to be a raw state_dict, but this
    helper keeps loading robust for training checkpoints or user checkpoints.
    """
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def load_state_dict_flexible(
    model,
    checkpoint_path,
    *,
    strict=True,
    map_location="cpu",
):
    """
    Load checkpoint weights into an existing model.

    Parameters
    ----------
    model:
        Initialized MHCPrime model.
    checkpoint_path:
        Path or importlib resource path to checkpoint.
    strict:
        Passed to model.load_state_dict().
    map_location:
        Passed to torch.load(). Default is CPU for safe cross-device loading.
    """
    # checkpoint = torch.load(checkpoint_path, map_location=map_location)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )
    state_dict = extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return incompatible


def load_mhcprime_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    *,
    device=None,
    strict=True,
    eval_mode=True,
    print_params=False,
    print_check=False,
):
    """
    Build the default MHCPrime architecture and load checkpoint weights.

    If checkpoint_path is None, the packaged default MHCPrime base checkpoint
    is loaded from:

        src/mhcprime/checkpoints/mhcprime_base/model_final.pt

    Users can still pass a custom checkpoint path for fine-tuned or alternative
    models.
    """
    device = resolve_device(device)

    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint_path()

    model, tokenizer, model_params = build_mhcprime_model(
        device=device,
        print_params=print_params,
        print_check=print_check,
    )

    load_state_dict_flexible(
        model,
        checkpoint_path,
        strict=strict,
        map_location="cpu",
    )

    model.to(device)

    if eval_mode:
        model.eval()

    return model, tokenizer, model_params


def get_rng_state():
    """
    Capture Python, NumPy, Torch, and CUDA RNG states.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state):
    """
    Restore Python, NumPy, Torch, and CUDA RNG states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_final_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    num_warmup_steps=None,
    num_training_steps=None,
    extra=None,
    save_rng=False,
):
    """
    Save a full training checkpoint.

    This is primarily for future training/resume support. It is distinct from
    the released base model checkpoint, which may be saved as a raw state_dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "num_warmup_steps": None if num_warmup_steps is None else int(num_warmup_steps),
        "num_training_steps": None if num_training_steps is None else int(num_training_steps),
        "param_group_info": [
            {
                "name": pg.get("name", f"group{i}"),
                "lr": pg.get("lr", None),
                "weight_decay": pg.get("weight_decay", None),
            }
            for i, pg in enumerate(optimizer.param_groups)
        ],
        "extra": extra or {},
    }

    if save_rng:
        ckpt["rng"] = get_rng_state()

    torch.save(ckpt, path)


def _move_optimizer_state_to_device(optimizer, device):
    """
    Move optimizer state tensors to a target device.

    Useful when resuming training from a checkpoint loaded on CPU.
    """
    device = resolve_device(device)

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)