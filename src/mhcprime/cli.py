from pathlib import Path
import argparse
import sys

import pandas as pd
import torch
import warnings

from .checkpointing import load_mhcprime_model
from .inference import predict_dataframe

def _configure_cli_warnings():
    """
    Suppress noisy non-fatal warnings in command-line inference.

    These do not change model behavior. They only keep CLI output clean.
    """
    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable.*",
        category=UserWarning,
    )

def _str_to_optional_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def _read_table(path, *, index_col=None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, index_col=index_col)

    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", index_col=index_col)

    raise ValueError(
        f"Unsupported input file format: {path}. "
        "Expected .csv, .csv.gz, .tsv, or .tsv.gz."
    )


def _write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df.to_csv(path, index=False)
        return

    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        df.to_csv(path, sep="\t", index=False)
        return

    raise ValueError(
        f"Unsupported output file format: {path}. "
        "Expected .csv, .csv.gz, .tsv, or .tsv.gz."
    )

def _confirm_overwrite(path: Path, *, overwrite: bool = False) -> bool:
    """
    Return True if it is okay to write to path.

    If path exists and overwrite=False, prompt the user interactively.
    """
    path = Path(path)

    if not path.exists():
        return True

    if overwrite:
        return True

    response = input(f"Output file already exists: {path}\nOverwrite? [y/N]: ")
    response = response.strip().lower()

    return response in {"y", "yes"}

def build_parser():
    parser = argparse.ArgumentParser(
        prog="mhcprime-predict",
        description=(
            "Score peptide-MHC class I candidates with MHCPrime and optionally "
            "add global background percentile ranks."
        ),
    )

    # Positional input/output.
    parser.add_argument(
        "input",
        type=str,
        help="Input table path. Supported: .csv, .csv.gz, .tsv, .tsv.gz.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/mhcprime_scored.csv",
        help=(
            "Output table path. Supported: .csv, .csv.gz, .tsv, .tsv.gz. "
            "Default: outputs/mhcprime_scored.csv"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output file if it already exists without prompting.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to a MHCPrime checkpoint .pt file. "
            "If omitted, the packaged default MHCPrime base checkpoint is used."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )

    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Load checkpoint with strict=True. Default.",
    )

    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Load checkpoint with strict=False.",
    )

    # Input reading.
    parser.add_argument(
        "--index-col",
        type=_str_to_optional_int,
        default=None,
        help="Optional index_col passed to pandas.read_csv. Default: None.",
    )

    # Column names / preprocessing.
    parser.add_argument(
        "--seq-col",
        type=str,
        default="seq",
        help="Input peptide sequence column. Default: seq.",
    )

    parser.add_argument(
        "--allele-col",
        type=str,
        default="allele",
        help="Input allele column. Expected MHCPrime format such as A0201. Default: allele.",
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Input label column. If absent, label=0 is added. Default: label.",
    )

    parser.add_argument(
        "--n-flank-col",
        type=str,
        default="n_flank",
        help="Input N-flank column. If absent, padded empty flanks are added. Default: n_flank.",
    )

    parser.add_argument(
        "--c-flank-col",
        type=str,
        default="c_flank",
        help="Input C-flank column. If absent, padded empty flanks are added. Default: c_flank.",
    )

    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        default=True,
        help="Preprocess raw input dataframe before inference. Default.",
    )

    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Assume input dataframe is already MHCPrime-processed.",
    )

    parser.add_argument(
        "--remove-flank",
        action="store_true",
        default=False,
        help="Ignore user flanks and replace with empty padded flanks.",
    )

    parser.add_argument(
        "--drop-missing-mhc",
        dest="drop_missing_mhc",
        action="store_true",
        default=True,
        help="Drop rows with alleles missing from the MHC pseudosequence dictionary. Default.",
    )

    parser.add_argument(
        "--error-missing-mhc",
        dest="drop_missing_mhc",
        action="store_false",
        help="Raise an error if alleles are missing from the MHC pseudosequence dictionary.",
    )

    parser.add_argument(
        "--warn-missing-mhc",
        dest="warn_missing_mhc",
        action="store_true",
        default=True,
        help="Warn when rows are dropped due to missing MHC pseudosequences. Default.",
    )

    parser.add_argument(
        "--no-warn-missing-mhc",
        dest="warn_missing_mhc",
        action="store_false",
        help="Do not warn when rows are dropped due to missing MHC pseudosequences.",
    )

    # Inference.
    parser.add_argument(
        "--mode",
        choices=["fast", "cached", "slow", "standard", "dataframe"],
        default="fast",
        help="Inference mode. Default: fast.",
    )

    parser.add_argument(
        "--score-col",
        type=str,
        default="mhcprime",
        help="Name of output score column. Default: mhcprime.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults depend on mode.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults depend on mode.",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default="outputs/mhcprime_dataset_cache.pt",
        help="Cache path for fast mode. Default: outputs/mhcprime_dataset_cache.pt.",
    )

    parser.add_argument(
        "--build-cache",
        dest="build_cache",
        action="store_true",
        default=True,
        help="Build token cache before fast inference. Default.",
    )

    parser.add_argument(
        "--no-build-cache",
        dest="build_cache",
        action="store_false",
        help="Reuse an existing token cache instead of building one.",
    )

    parser.add_argument(
        "--remove-cache",
        dest="remove_cache",
        action="store_true",
        default=True,
        help="Remove token cache after fast inference. Default.",
    )

    parser.add_argument(
        "--keep-cache",
        dest="remove_cache",
        action="store_false",
        help="Keep token cache after inference.",
    )

    parser.add_argument(
        "--use-bf16",
        dest="use_bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 fast inference on CUDA. Default.",
    )

    parser.add_argument(
        "--no-bf16",
        dest="use_bf16",
        action="store_false",
        help="Do not use bfloat16 in fast mode.",
    )

    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=False,
        help="Use torch.compile in fast mode. Default: false.",
    )

    parser.add_argument(
        "--restore-model-dtype",
        dest="restore_model_dtype",
        action="store_true",
        default=True,
        help="Restore model dtype after bf16 fast inference. Default.",
    )

    parser.add_argument(
        "--no-restore-model-dtype",
        dest="restore_model_dtype",
        action="store_false",
        help="Do not restore model dtype after bf16 fast inference.",
    )

    parser.add_argument(
        "--use-domain-id",
        action="store_true",
        default=False,
        help="Include domain_id in cached batches. Mostly for internal use.",
    )

    # Ranking.
    parser.add_argument(
        "--add-rank",
        dest="add_rank",
        action="store_true",
        default=True,
        help="Add global background percentile-rank column. Default.",
    )

    parser.add_argument(
        "--no-rank",
        dest="add_rank",
        action="store_false",
        help="Do not add percentile-rank column.",
    )

    parser.add_argument(
        "--rank-background-path",
        type=str,
        default=None,
        help=(
            "Optional external .npz file with background scores. "
            "If omitted, uses packaged default background scores."
        ),
    )

    parser.add_argument(
        "--rank-suffix",
        type=str,
        default="_rank",
        help="Suffix for rank column. Default: _rank.",
    )

    parser.add_argument(
        "--rank-reverse",
        dest="rank_reverse",
        action="store_true",
        default=True,
        help="Higher scores receive higher percentile ranks. Default.",
    )

    parser.add_argument(
        "--rank-forward",
        dest="rank_reverse",
        action="store_false",
        help="Lower scores receive higher percentile ranks.",
    )

    parser.add_argument(
        "--rank-dtype",
        type=str,
        default="float32",
        help="Rank output dtype. Default: float32.",
    )

    # Output shape / verbosity.
    parser.add_argument(
        "--return-processed",
        action="store_true",
        default=False,
        help="Return/save internal processed dataframe instead of clean user-facing output.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Disable progress bars and reduce printed output.",
    )

    parser.add_argument(
        "--debug-env",
        action="store_true",
        default=False,
        help="Print Python, Torch, CUDA, and device diagnostic information.",
    )

    return parser

def _resolve_cli_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def predict_cli(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not _confirm_overwrite(output_path, overwrite=args.overwrite):
        print(f"Output file exists and overwrite was not confirmed. Exiting: {output_path}")
        return None
    
    checkpoint_msg = (
        args.checkpoint
        if args.checkpoint is not None
        else "packaged default MHCPrime base checkpoint"
    )
    
    if not args.quiet:
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Checkpoint: {checkpoint_msg}")
        print(f"Mode: {args.mode}")

    df = _read_table(input_path, index_col=args.index_col)

    if not args.quiet:
        print(f"Loaded input dataframe: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    device = torch.device(args.device) if args.device is not None else None

    model, tokenizer, model_params = load_mhcprime_model(
        args.checkpoint,
        device=device,
        strict=args.strict,
        eval_mode=True,
        print_params=False,
        print_check=False,
    )

    # actual_device = next(model.parameters()).device
    device = _resolve_cli_device(args.device)

    if not args.quiet:
        _print_runtime_info(device=device, debug_env=args.debug_env)

    preprocess_kwargs = {
        "seq_col": args.seq_col,
        "allele_col": args.allele_col,
        "label_col": args.label_col,
        "n_flank_col": args.n_flank_col,
        "c_flank_col": args.c_flank_col,
        "remove_flank": args.remove_flank,
        "drop_missing_mhc": args.drop_missing_mhc,
        "warn_missing_mhc": args.warn_missing_mhc,
    }

    predict_kwargs = {
        "model": model,
        "df": df,
        "tokenizer": tokenizer,
        "score_col": args.score_col,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "preprocess": args.preprocess,
        "preprocess_kwargs": preprocess_kwargs,
        "add_rank": args.add_rank,
        "rank_background_path": args.rank_background_path,
        "rank_suffix": args.rank_suffix,
        "rank_reverse": args.rank_reverse,
        "rank_dtype": args.rank_dtype,
        "return_processed": args.return_processed,
        "disable_tqdm": args.quiet,
    }

    # Fast-mode-specific options. These are accepted by predict_dataframe_fast
    # through predict_dataframe(..., **kwargs). They are ignored for slow mode.
    if str(args.mode).lower().strip() in {"fast", "cached"}:
        predict_kwargs.update(
            {
                "cache_path": args.cache_path,
                "build_cache": args.build_cache,
                "remove_cache": args.remove_cache,
                "use_bf16": args.use_bf16,
                "compile_model": args.compile_model,
                "use_domain_id": args.use_domain_id,
                "restore_model_dtype": args.restore_model_dtype,
            }
        )

    scored_df = predict_dataframe(**predict_kwargs)

    _write_table(scored_df, output_path)

    if not args.quiet:
        print(f"Saved scored dataframe: {output_path}")
        print(f"Output shape: {scored_df.shape[0]:,} rows, {scored_df.shape[1]:,} columns")

    return scored_df

def _print_runtime_info(*, device, debug_env=False):
    print(f"Using device: {device}")

    if debug_env:
        import sys
        import torch

        print("Python executable:", sys.executable)
        print("Torch version:", torch.__version__)
        print("Torch file:", torch.__file__)
        print("Torch CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device 0:", torch.cuda.get_device_name(0))

def main(argv=None):
    _configure_cli_warnings()

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        predict_cli(args)
    except Exception as exc:
        print(f"[mhcprime-predict] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()