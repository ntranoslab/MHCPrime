from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model_utils import PeptideMHCDataset, collate_fn
from .preprocessing import (
    _mhc_column_for_allele,
    prepare_input_dataframe,
    unprocess_output_dataframe,
)
from .ranking import (
    add_global_background_percentile_ranks,
    load_global_background_scores,
)


def _resolve_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _maybe_add_global_ranks(
    scored_df,
    *,
    score_cols: Sequence[str],
    add_rank: bool = True,
    rank_background_scores=None,
    rank_background_path: Optional[Union[str, Path]] = None,
    rank_suffix: str = "_rank",
    rank_reverse: bool = True,
    rank_dtype: str = "float32",
    disable_tqdm: bool = False,
):
    """
    Add global background percentile ranks if requested.

    By default, this loads the packaged MHCPrime global background score file:
        src/mhcprime/data/mhcprime_global_background_scores.npz

    For reverse=True:
        rank = percent of background scores <= test score
    """
    if not add_rank:
        return scored_df

    if rank_background_scores is None:
        if rank_background_path is None:
            rank_background_scores = load_global_background_scores()
        else:
            rank_background_scores = load_global_background_scores(
                rank_background_path,
                package_data=False,
            )

    return add_global_background_percentile_ranks(
        scored_df,
        background_scores=rank_background_scores,
        score_cols=score_cols,
        suffix=rank_suffix,
        reverse=rank_reverse,
        copy=False,
        dtype=rank_dtype,
        show_progress=not disable_tqdm,
    )

def _cleanup_user_facing_output(scored_df, *, original_columns):
    """
    Clean user-facing inference output.

    Behavior:
    - Drop generated n_flank/c_flank columns only if they were not present
      in the user's original input.
    - Drop accidental saved-index columns such as Unnamed: 0.

    Metadata columns supplied by the user are otherwise preserved.
    """
    out = scored_df.copy()
    original_columns = set(original_columns)

    drop_cols = []

    if "n_flank" not in original_columns and "n_flank" in out.columns:
        drop_cols.append("n_flank")

    if "c_flank" not in original_columns and "c_flank" in out.columns:
        drop_cols.append("c_flank")

    if "Unnamed: 0" in out.columns:
        drop_cols.append("Unnamed: 0")

    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out

def run_mhcprime(
    model,
    test_df,
    tokenizer,
    batch_size=1024,
    num_workers=16,
    score_col="score",
    device=None,
    return_embeddings=False,
    include_hardness_score=False,
    include_domain_id=False,
    include_prior_score=False,
    disable_tqdm=False,
):
    """
    Standard dataframe-based MHCPrime inference.

    Expects test_df to already be processed with prepare_input_dataframe or
    equivalent columns required by PeptideMHCDataset.
    """
    device = _resolve_device(device)

    model.eval()
    model.to(device)

    test_dataset = PeptideMHCDataset(
        test_df,
        tokenizer,
        include_hardness_score=include_hardness_score,
        include_domain_id=include_domain_id,
        include_prior_score=include_prior_score,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logit_scores = []
    bos_embeddings_dict = {}
    batch_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Running inference", disable=disable_tqdm)
        ):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Remove fields not consumed by the model.
            batch.pop("label", None)
            batch.pop("original_idx", None)
            batch.pop("allele", None)

            if return_embeddings:
                logits, bos_embeddings = model(batch, return_embeddings=True)

                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(test_df))
                batch_indices.extend(range(batch_start, batch_end))

                for i, embedding in enumerate(bos_embeddings):
                    if batch_start + i < len(test_df):
                        bos_embeddings_dict[batch_start + i] = embedding.cpu().numpy()
            else:
                if hasattr(model, "transformer_type") and model.transformer_type == "switch":
                    outputs = model(batch, return_embeddings=False)
                    logits = outputs[0]
                else:
                    logits = model(batch, return_embeddings=False)

            if getattr(model, "fc_output_dim", 1) == 1:
                logit_scores.extend(logits.detach().cpu().view(-1).numpy())
            elif model.fc_output_dim == 2:
                logit_scores.extend(logits[:, 1].detach().cpu().numpy())
            else:
                raise ValueError(f"Unsupported fc_output_dim={model.fc_output_dim}")

    test_df = test_df.copy()
    test_df[score_col] = logit_scores

    if return_embeddings:
        keys = test_df["seq"].astype(str) + "_" + test_df["allele"].astype(str)

        peptide_mhc_to_bos = {
            key: embedding
            for key, embedding, idx in zip(
                keys,
                bos_embeddings_dict.values(),
                bos_embeddings_dict.keys(),
            )
            if idx < len(test_df)
        }

        return test_df, peptide_mhc_to_bos

    return test_df


def build_token_cache(
    df,
    cache_out: Union[str, Path],
    tokenizer,
    pep_len: int = 34,
    mhc_len: int = 34,
    pad_id: int = 1,
    use_domain_id: bool = False,
    disable_tqdm: bool = False,
):
    """
    Build a tokenized cache for fast MHCPrime inference.

    Expects df to already be processed with prepare_input_dataframe or
    equivalent columns.
    """
    cache_out = Path(cache_out)
    cache_out.parent.mkdir(parents=True, exist_ok=True)

    n = len(df)

    pep = torch.full((n, pep_len), pad_id, dtype=torch.long)
    mhc = torch.full((n, mhc_len), pad_id, dtype=torch.long)
    lab = torch.tensor(df["label"].values, dtype=torch.float32)

    if use_domain_id:
        did = torch.tensor(df["domain_id"].values, dtype=torch.long)

    iterator = tqdm(
        enumerate(df.itertuples(index=False)),
        total=n,
        desc="Building token cache",
        disable=disable_tqdm,
    )

    for i, row in iterator:
        pep_ids = tokenizer.encode(row.n_flank + row.seq + row.c_flank)[:pep_len]
        pep[i, : len(pep_ids)] = torch.as_tensor(pep_ids, dtype=torch.long)

        mhc_col = _mhc_column_for_allele(row.allele)
        mhc_ids = tokenizer.encode(getattr(row, mhc_col))[:mhc_len]
        mhc[i, : len(mhc_ids)] = torch.as_tensor(mhc_ids, dtype=torch.long)

    if use_domain_id:
        torch.save({"pep": pep, "mhc": mhc, "lab": lab, "domain_id": did}, cache_out)
    else:
        torch.save({"pep": pep, "mhc": mhc, "lab": lab}, cache_out)

    print(f"wrote cached tensors -> {cache_out} ({pep.nbytes / 1e6:.1f} MB)")


class CachedDataset(Dataset):
    def __init__(self, cache_pt: Union[str, Path], use_domain_id: bool = False):
        data = torch.load(cache_pt, map_location="cpu")
        self.pep = data["pep"]
        self.mhc = data["mhc"]
        self.label = data["lab"]
        self.use_domain_id = use_domain_id

        if use_domain_id:
            self.domain_id = data["domain_id"]

    def __len__(self):
        return self.pep.size(0)

    def __getitem__(self, idx):
        if self.use_domain_id:
            return self.pep[idx], self.mhc[idx], self.label[idx], self.domain_id[idx]
        return self.pep[idx], self.mhc[idx], self.label[idx]


def collate_cached(batch, pad_id=1, use_domain_id=False):
    if use_domain_id:
        pep, mhc, lab, did = map(torch.stack, zip(*batch))
        pep_mask = pep.ne(pad_id)
        mhc_mask = mhc.ne(pad_id).unsqueeze(1)
        return {
            "peptide": pep,
            "peptide_mask": pep_mask,
            "mhc_list": mhc.unsqueeze(1),
            "mhc_mask_list": mhc_mask,
            "label": lab,
            "domain_id": did,
        }

    pep, mhc, lab = map(torch.stack, zip(*batch))
    pep_mask = pep.ne(pad_id)
    mhc_mask = mhc.ne(pad_id).unsqueeze(1)

    return {
        "peptide": pep,
        "peptide_mask": pep_mask,
        "mhc_list": mhc.unsqueeze(1),
        "mhc_mask_list": mhc_mask,
        "label": lab,
    }


def run_mhcprime_fast(
    model,
    cache_pt: Union[str, Path],
    batch_size: int = 3072,
    num_workers: int = 8,
    use_bf16: bool = True,
    compile_model: bool = False,
    use_domain_id: bool = False,
    device=None,
    restore_model_dtype: bool = True,
    disable_tqdm: bool = False,
):
    """
    Fast cached inference path.

    This path is intended primarily for CUDA inference. If CUDA + bf16 are used,
    the model is temporarily moved to bfloat16 and then restored to its original
    dtype by default so that users can run fast inference and then slow/fp32
    inference with the same model object.
    """
    device = _resolve_device(device)
    cache_pt = Path(cache_pt)

    ds = CachedDataset(cache_pt, use_domain_id=use_domain_id)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_cached, use_domain_id=use_domain_id),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        num_workers=num_workers,
    )

    model = model.eval().to(device)

    original_dtype = next(model.parameters()).dtype
    use_amp = device.type == "cuda" and use_bf16
    casted_to_bf16 = False

    working_model = model

    try:
        if use_amp:
            model.to(torch.bfloat16)
            casted_to_bf16 = True

        if compile_model:
            if device.type != "cuda":
                raise RuntimeError("compile_model=True is currently only recommended for CUDA.")
            working_model = torch.compile(model, mode="reduce-overhead")

        all_logits = []

        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)

        if use_amp:
            autocast_context = torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        else:
            autocast_context = nullcontext()

        with torch.inference_mode(), autocast_context:
            for batch in tqdm(loader, desc="Running fast inference", disable=disable_tqdm):
                batch = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                batch.pop("label", None)

                logits = working_model(batch, return_embeddings=False)
                all_logits.append(logits.float().detach().cpu().view(-1))

        return torch.cat(all_logits)

    finally:
        if restore_model_dtype and casted_to_bf16:
            model.to(original_dtype)


def predict_dataframe_slow(
    model,
    df,
    tokenizer,
    *,
    score_col: str = "mhcprime_score",
    batch_size: int = 1024,
    num_workers: int = 0,
    device=None,
    preprocess: bool = True,
    preprocess_kwargs: Optional[dict] = None,
    add_rank: bool = True,
    rank_background_scores=None,
    rank_background_path: Optional[Union[str, Path]] = None,
    rank_suffix: str = "_rank",
    rank_reverse: bool = True,
    rank_dtype: str = "float32",
    return_processed: bool = False,
    disable_tqdm: bool = False,
):
    """
    Public slow/standard inference wrapper.

    Takes a raw dataframe by default, preprocesses it, scores it, optionally
    adds global percentile ranks, and returns a user-facing dataframe by default.
    """
    preprocess_kwargs = preprocess_kwargs or {}
    original_columns = set(df.columns)

    if preprocess:
        processed_df = prepare_input_dataframe(df, **preprocess_kwargs)
    else:
        processed_df = df.copy()

    scored_df = run_mhcprime(
        model,
        processed_df,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        score_col=score_col,
        device=device,
        disable_tqdm=disable_tqdm,
    )

    scored_df = _maybe_add_global_ranks(
        scored_df,
        score_cols=[score_col],
        add_rank=add_rank,
        rank_background_scores=rank_background_scores,
        rank_background_path=rank_background_path,
        rank_suffix=rank_suffix,
        rank_reverse=rank_reverse,
        rank_dtype=rank_dtype,
        disable_tqdm=disable_tqdm,
    )

    if not return_processed:
        scored_df = unprocess_output_dataframe(scored_df)
        scored_df = _cleanup_user_facing_output(
            scored_df,
            original_columns=original_columns,
        )

    return scored_df


def predict_dataframe_fast(
    model,
    df,
    tokenizer,
    *,
    score_col: str = "mhcprime_score",
    batch_size: int = 3072,
    num_workers: int = 8,
    device=None,
    preprocess: bool = True,
    preprocess_kwargs: Optional[dict] = None,
    cache_path: Union[str, Path] = "dataset_cache.pt",
    build_cache: bool = True,
    remove_cache: bool = True,
    use_bf16: bool = True,
    compile_model: bool = False,
    use_domain_id: bool = False,
    restore_model_dtype: bool = True,
    add_rank: bool = True,
    rank_background_scores=None,
    rank_background_path: Optional[Union[str, Path]] = None,
    rank_suffix: str = "_rank",
    rank_reverse: bool = True,
    rank_dtype: str = "float32",
    return_processed: bool = False,
    disable_tqdm: bool = False,
):
    """
    Public fast cached inference wrapper.

    Workflow:
        optional preprocess dataframe
        build token cache if requested
        run fast cached inference
        attach scores
        optionally add global percentile ranks
        optionally delete cache
        return user-facing dataframe by default
    """
    preprocess_kwargs = preprocess_kwargs or {}
    cache_path = Path(cache_path)
    original_columns = set(df.columns)

    if preprocess:
        processed_df = prepare_input_dataframe(df, **preprocess_kwargs)
    else:
        processed_df = df.copy()

    if build_cache:
        build_token_cache(
            processed_df,
            cache_path,
            tokenizer,
            use_domain_id=use_domain_id,
            disable_tqdm=disable_tqdm,
        )

    try:
        output_scores = run_mhcprime_fast(
            model,
            cache_path,
            batch_size=batch_size,
            num_workers=num_workers,
            use_bf16=use_bf16,
            compile_model=compile_model,
            use_domain_id=use_domain_id,
            device=device,
            restore_model_dtype=restore_model_dtype,
            disable_tqdm=disable_tqdm,
        )

        scored_df = processed_df.copy()
        scored_df[score_col] = output_scores.detach().cpu().tolist()

        scored_df = _maybe_add_global_ranks(
            scored_df,
            score_cols=[score_col],
            add_rank=add_rank,
            rank_background_scores=rank_background_scores,
            rank_background_path=rank_background_path,
            rank_suffix=rank_suffix,
            rank_reverse=rank_reverse,
            rank_dtype=rank_dtype,
            disable_tqdm=disable_tqdm,
        )

    finally:
        if remove_cache and cache_path.exists():
            cache_path.unlink()

    if not return_processed:
        scored_df = unprocess_output_dataframe(scored_df)
        scored_df = _cleanup_user_facing_output(
            scored_df,
            original_columns=original_columns,
        )


    return scored_df


def predict_dataframe(
    model,
    df,
    tokenizer,
    *,
    score_col: str = "mhcprime_score",
    mode: str = "fast",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    device=None,
    preprocess: bool = True,
    preprocess_kwargs: Optional[dict] = None,
    add_rank: bool = True,
    rank_background_scores=None,
    rank_background_path: Optional[Union[str, Path]] = None,
    rank_suffix: str = "_rank",
    rank_reverse: bool = True,
    rank_dtype: str = "float32",
    return_processed: bool = False,
    disable_tqdm: bool = False,
    **kwargs,
):
    """
    Unified public inference wrapper.

    mode:
        "fast" or "cached" -> cached tensor inference
        "slow", "standard", or "dataframe" -> regular dataframe inference
    """
    mode = str(mode).lower().strip()

    if mode in {"fast", "cached"}:
        if batch_size is None:
            batch_size = 3072
        if num_workers is None:
            num_workers = 8

        return predict_dataframe_fast(
            model,
            df,
            tokenizer,
            score_col=score_col,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            preprocess=preprocess,
            preprocess_kwargs=preprocess_kwargs,
            add_rank=add_rank,
            rank_background_scores=rank_background_scores,
            rank_background_path=rank_background_path,
            rank_suffix=rank_suffix,
            rank_reverse=rank_reverse,
            rank_dtype=rank_dtype,
            disable_tqdm=disable_tqdm,
            return_processed=return_processed,
            **kwargs,
        )

    if mode in {"slow", "standard", "dataframe"}:
        if batch_size is None:
            batch_size = 1024
        if num_workers is None:
            num_workers = 0

        return predict_dataframe_slow(
            model,
            df,
            tokenizer,
            score_col=score_col,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            preprocess=preprocess,
            preprocess_kwargs=preprocess_kwargs,
            add_rank=add_rank,
            rank_background_scores=rank_background_scores,
            rank_background_path=rank_background_path,
            rank_suffix=rank_suffix,
            rank_reverse=rank_reverse,
            rank_dtype=rank_dtype,
            disable_tqdm=disable_tqdm,
            return_processed=return_processed,
        )

    raise ValueError(
        f"Unknown inference mode '{mode}'. Expected one of "
        "{'fast', 'cached', 'slow', 'standard', 'dataframe'}."
    )


def predict_with_models(
    models: Mapping[str, torch.nn.Module],
    df,
    tokenizer,
    *,
    mode: str = "fast",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    device=None,
    preprocess: bool = True,
    preprocess_kwargs: Optional[dict] = None,
    cache_dir: Union[str, Path] = ".",
    cache_name: str = "dataset_cache.pt",
    remove_cache: bool = True,
    add_rank: bool = True,
    rank_background_scores=None,
    rank_background_path: Optional[Union[str, Path]] = None,
    rank_suffix: str = "_rank",
    rank_reverse: bool = True,
    rank_dtype: str = "float32",
    disable_tqdm: bool = False,
    return_processed: bool = False,
    **kwargs,
):
    """
    Score one dataframe with multiple models and attach one score column per model.

    For fast mode, the cache is built once and reused across models.

    If add_rank=True, the same global background distribution is used to rank
    all model score columns.
    """
    preprocess_kwargs = preprocess_kwargs or {}
    original_columns = set(df.columns)

    if preprocess:
        processed_df = prepare_input_dataframe(df, **preprocess_kwargs)
    else:
        processed_df = df.copy()

    mode = str(mode).lower().strip()
    out_df = processed_df.copy()

    if mode in {"fast", "cached"}:
        if batch_size is None:
            batch_size = 3072
        if num_workers is None:
            num_workers = 8

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_name

        build_token_cache(
            processed_df,
            cache_path,
            tokenizer,
            use_domain_id=kwargs.get("use_domain_id", False),
            disable_tqdm=disable_tqdm,
        )

        try:
            for model_name, model in models.items():
                scores = run_mhcprime_fast(
                    model,
                    cache_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_bf16=kwargs.get("use_bf16", True),
                    compile_model=kwargs.get("compile_model", False),
                    use_domain_id=kwargs.get("use_domain_id", False),
                    device=device,
                    restore_model_dtype=kwargs.get("restore_model_dtype", True),
                    disable_tqdm=disable_tqdm,
                )
                out_df[model_name] = scores.detach().cpu().tolist()

            out_df = _maybe_add_global_ranks(
                out_df,
                score_cols=list(models.keys()),
                add_rank=add_rank,
                rank_background_scores=rank_background_scores,
                rank_background_path=rank_background_path,
                rank_suffix=rank_suffix,
                rank_reverse=rank_reverse,
                rank_dtype=rank_dtype,
                disable_tqdm=disable_tqdm,
            )

        finally:
            if remove_cache and cache_path.exists():
                cache_path.unlink()

        if not return_processed:
            out_df = unprocess_output_dataframe(out_df)
            out_df = _cleanup_user_facing_output(
                out_df,
                original_columns=original_columns,
            )

        return out_df

    if mode in {"slow", "standard", "dataframe"}:
        if batch_size is None:
            batch_size = 1024
        if num_workers is None:
            num_workers = 0

        for model_name, model in models.items():
            scored = run_mhcprime(
                model,
                processed_df,
                tokenizer,
                batch_size=batch_size,
                num_workers=num_workers,
                score_col=model_name,
                device=device,
                disable_tqdm=disable_tqdm,
            )
            out_df[model_name] = scored[model_name].values

        out_df = _maybe_add_global_ranks(
            out_df,
            score_cols=list(models.keys()),
            add_rank=add_rank,
            rank_background_scores=rank_background_scores,
            rank_background_path=rank_background_path,
            rank_suffix=rank_suffix,
            rank_reverse=rank_reverse,
            rank_dtype=rank_dtype,
            disable_tqdm=disable_tqdm,
        )

        if not return_processed:
            out_df = unprocess_output_dataframe(out_df)
            out_df = _cleanup_user_facing_output(
                out_df,
                original_columns=original_columns,
            )

        return out_df

    raise ValueError(
        f"Unknown inference mode '{mode}'. Expected one of "
        "{'fast', 'cached', 'slow', 'standard', 'dataframe'}."
    )