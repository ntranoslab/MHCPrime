"""
Train MHCPrime on unprocessed MS training data.

Default use from repo root:

    python scripts/train_mhcprime_base.py --gpu 0

or:

    python scripts/train_mhcprime_base.py --device cuda:0

Expected default data location:

    train_test_data/ms_train_data.csv.gz

This script intentionally preserves the original base-training defaults:
    n_epochs = 120
    batch_size = 3072
    num_pos_per_epoch = 200_000
    neg_pos_ratio = 1
    num_workers = 16
    save_every_n_epochs = 60
    loss_type = logsmoothap
    loss_hp = 10.0
    encoder_lr = 2e-4
    decoder_lr = 2e-4
    seed = 42
    optimizer_type = AdamW

The trainer itself is not modified.
"""

import argparse
import gc
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch


# ---------------------------------------------------------------------
# Make src-layout imports work when running this script directly from
# the cloned repository without requiring users to manually set PYTHONPATH.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from mhcprime import (  # noqa: E402
    AminoAcidTokenizer,
    create_seq_allele_column,
    full_process_data,
    get_default_model_params,
    init_model,
    load_default_mhc_pseudosequences,
    print_data_stats,
)
from mhcprime.checkpointing import load_state_dict_flexible  # noqa: E402
from mhcprime.features import load_processed_feature_table  # noqa: E402
from mhcprime.training import train_mhcprime  # noqa: E402


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------

def save_dict_pickle(obj, path):
    """
    Save an object as a pickle file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def clear_all_gpu_memory():
    """
    Clear Python and CUDA memory after training.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resolve_training_device(args):
    """
    Resolve training device from --device or --gpu.

    Priority:
        1. --device
        2. --gpu
        3. cuda if available
        4. cpu

    Examples:
        --device cuda:0
        --device cuda
        --device cpu
        --gpu 0
    """
    if args.device is not None:
        device = torch.device(args.device)
    elif args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}', but torch.cuda.is_available() is False."
            )

        gpu_index = device.index
        if gpu_index is None:
            gpu_index = torch.cuda.current_device()

        torch.cuda.set_device(gpu_index)

        print(
            f"Using GPU: {torch.cuda.get_device_name(gpu_index)} "
            f"(ID: {torch.cuda.current_device()})"
        )
    else:
        print("Using device: CPU")

    return device


def read_training_dataframe(path):
    """
    Read CSV/CSV.GZ/TSV/TSV.GZ training data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n\n"
            "Place the downloaded training data at the default location:\n"
            "  train_test_data/ms_train_data.csv.gz\n\n"
            "or pass a custom path with:\n"
            "  --train-data path/to/ms_train_data.csv.gz"
        )

    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def confirm_or_exit(run_dir, overwrite=False):
    """
    Prompt before writing into an existing run directory unless --overwrite is set.
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        return

    if overwrite:
        print(f"Run directory exists and --overwrite was provided: {run_dir}")
        return

    print(run_dir)
    choice = input(
        "This run directory already exists. "
        "Do you want to continue and overwrite/add to it? [y/N]: "
    ).strip().lower()

    if choice not in {"y", "yes"}:
        print("Exiting.")
        sys.exit(0)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train MHCPrime base model on MS training data."
    )

    # Paths
    parser.add_argument(
        "--train-data",
        type=str,
        default=str(PROJECT_ROOT / "train_test_data" / "ms_train_data.csv.gz"),
        help=(
            "Path to unprocessed MS training data CSV/CSV.GZ/TSV/TSV.GZ. "
            "Default: train_test_data/ms_train_data.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "model_checkpoints"),
        help="Root directory for training outputs. Default: model_checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="MHCPrime_Base",
        help="Run name. Outputs are saved under output-root/run-name.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path to initialize model weights before training. "
            "This performs weight-initialized continued training, not exact optimizer resume."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory without prompting.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, e.g. cuda:0, cuda:1, cuda, or cpu.",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help=(
            "Optional GPU ID convenience argument. "
            "Equivalent to --device cuda:<gpu>. Ignored if --device is provided."
        ),
    )

    # Training hyperparameters: defaults match original base training script.
    parser.add_argument("--n-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--num-pos-per-epoch", type=int, default=200_000)
    parser.add_argument("--neg-pos-ratio", type=float, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers.",
    )

    parser.add_argument("--save-every-n-epochs", type=int, default=60)

    parser.add_argument("--loss-type", type=str, default="logsmoothap")
    parser.add_argument("--loss-hp", type=float, default=10.0)

    parser.add_argument("--encoder-lr", type=float, default=2e-4)
    parser.add_argument("--decoder-lr", type=float, default=2e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-type", type=str, default="AdamW")

    parser.add_argument("--lr-warmup-epochs", type=int, default=None)
    parser.add_argument(
        "--percentage-of-total-steps-for-warmup",
        type=float,
        default=None,
    )

    return parser


# ---------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = resolve_training_device(args)

    # -----------------------------------------------------------------
    # Load packaged MHC pseudosequences and feature table.
    # -----------------------------------------------------------------

    print("\nLoading packaged MHC pseudosequences...")
    mhc_seq_dict = load_default_mhc_pseudosequences()
    print(f"Loaded {len(mhc_seq_dict):,} MHC pseudosequences.\n")

    print("Loading processed amino-acid feature table...")
    processed_feature_table = load_processed_feature_table()
    feature_names = list(processed_feature_table.columns)
    print(f"Loaded feature table with shape: {processed_feature_table.shape}\n")

    # -----------------------------------------------------------------
    # Load and process training data.
    # -----------------------------------------------------------------

    print("Loading training data...")
    train_data_path = Path(args.train_data)
    ms_train_data = read_training_dataframe(train_data_path)
    print(f"Training data path: {train_data_path}")
    print_data_stats(ms_train_data)
    print()

    print("Processing training data...")
    ms_train_data = full_process_data(
        ms_train_data,
        mhc_seq_dict,
        remove_flank=True,
        add_pseudo_label=False,
    )
    ms_train_data = create_seq_allele_column(ms_train_data)

    # Preserve original script behavior.
    ms_train_data["method_group"] = "MS"
    method_group_map = {"MS": 0, "nonMS": 1}
    ms_train_data["domain_id"] = ms_train_data["method_group"].map(method_group_map)

    ms_train_data.reset_index(inplace=True, drop=True)
    ms_train_data["example_id"] = ms_train_data.index.tolist()

    print_data_stats(ms_train_data)
    print()

    # -----------------------------------------------------------------
    # Build model using the released default architecture.
    # -----------------------------------------------------------------

    print("Initializing tokenizer and model...")
    tokenizer = AminoAcidTokenizer()

    model_params = get_default_model_params(
        tokenizer=tokenizer,
        processed_feature_table=processed_feature_table,
        feature_names=feature_names,
    )

    model = init_model(
        model_params,
        print_check=False,
        device=device,
    )

    if args.init_checkpoint is not None:
        init_checkpoint = Path(args.init_checkpoint)
        print(f"\nLoading initialization checkpoint: {init_checkpoint}")
        load_state_dict_flexible(
            model,
            init_checkpoint,
            strict=True,
            map_location="cpu",
        )
        model.to(device)
        print("Initialization checkpoint loaded.\n")

    print()

    # -----------------------------------------------------------------
    # Output directories.
    # -----------------------------------------------------------------

    print("Setting up output directories...")
    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name
    output_dir = run_dir / "Checkpoints"

    print("Run dir:", run_dir)
    print("Checkpoint dir:", output_dir)

    confirm_or_exit(run_dir, overwrite=args.overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir set:", output_dir)
    print()

    # -----------------------------------------------------------------
    # Training parameters.
    # Defaults are intentionally matched to original base training script.
    # -----------------------------------------------------------------

    training_params = {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "num_pos_per_epoch": args.num_pos_per_epoch,
        "neg_pos_ratio": args.neg_pos_ratio,
        "num_workers": args.num_workers,
        "persistent_workers": not args.no_persistent_workers,
        "device": str(device),
        "save_every_n_epochs": args.save_every_n_epochs,
        "output_dir": output_dir,
        "loss_type": args.loss_type,
        "loss_hp": args.loss_hp,
        "encoder_lr": args.encoder_lr,
        "decoder_lr": args.decoder_lr,
        "seed": args.seed,
        "optimizer_type": args.optimizer_type,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "percentage_of_total_steps_for_warmup": args.percentage_of_total_steps_for_warmup,
    }

    print("Training parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    print()

    # -----------------------------------------------------------------
    # Train.
    # -----------------------------------------------------------------

    print("Starting training...")
    training_stats = train_mhcprime(
        model=model,
        train_df=ms_train_data,
        tokenizer=tokenizer,
        **training_params,
    )
    print("Finished training.\n")

    # -----------------------------------------------------------------
    # Save final model and metadata.
    # -----------------------------------------------------------------

    print("Saving final checkpoint and training metadata...")

    torch.save(model.state_dict(), output_dir / "model_final.pt")
    save_dict_pickle(training_stats, output_dir / "training_stats.pkl")
    save_dict_pickle(training_params, run_dir / "training_params.pkl")

    model_params_to_save = model_params.copy()
    model_params_to_save.pop("feature_table", None)
    model_params_to_save.pop("vocab_size", None)
    model_params_to_save.pop("sequence_embedder_tokenizer", None)

    save_dict_pickle(model_params_to_save, run_dir / "model_params.pkl")

    print("Saved:")
    print("  Final model:", output_dir / "model_final.pt")
    print("  Training stats:", output_dir / "training_stats.pkl")
    print("  Training params:", run_dir / "training_params.pkl")
    print("  Model params:", run_dir / "model_params.pkl")

    clear_all_gpu_memory()


if __name__ == "__main__":
    main()