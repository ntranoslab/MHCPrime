import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import GradScaler, autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .checkpointing import save_final_checkpoint
from .losses import LogSmoothAPScore, SoftSpearmanLoss
from .model_utils import PeptideMHCDataset, collate_fn
from .samplers import make_simple_negative_loader


def compute_label_stats(all_labels, all_preds, threshold=0.0, print_stats=False):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_1_preds = all_preds[all_labels == 1]
    label_0_preds = all_preds[all_labels == 0]

    stats = {
        "mean_label_1": np.mean(label_1_preds),
        "variance_label_1": np.var(label_1_preds),
        "mean_label_0": np.mean(label_0_preds),
        "variance_label_0": np.var(label_0_preds)
    }

    pred_labels = (all_preds >= threshold).astype(int)

    TP = np.sum((pred_labels == 1) & (all_labels == 1)) # True Positives
    FP = np.sum((pred_labels == 1) & (all_labels == 0)) # False Positives
    TN = np.sum((pred_labels == 0) & (all_labels == 0)) # True Negatives
    FN = np.sum((pred_labels == 0) & (all_labels == 1)) # False Negatives

    TP_mean = np.mean(all_preds[(pred_labels == 1) & (all_labels == 1)]) if TP > 0 else 0
    FP_mean = np.mean(all_preds[(pred_labels == 1) & (all_labels == 0)]) if FP > 0 else 0
    TN_mean = np.mean(all_preds[(pred_labels == 0) & (all_labels == 0)]) if TN > 0 else 0
    FN_mean = np.mean(all_preds[(pred_labels == 0) & (all_labels == 1)]) if FN > 0 else 0

    stats.update({
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "TP_mean": TP_mean,
        "FP_mean": FP_mean,
        "TN_mean": TN_mean,
        "FN_mean": FN_mean,
        "Precision": TP / (TP + FP + 1e-8),
        "Recall": TP / (TP + FN + 1e-8),
        "F1 Score": 2 * TP / (2 * TP + FP + FN + 1e-8),
        "Total Positives": TP + FN,
        "Total Negatives": FP + TN,
        "Total Samples": len(all_labels)
    })

    if print_stats:
        print("\n===== Label Statistics =====")
        print(f"Mean Prediction for Label 1: {stats['mean_label_1']:.4f}")
        print(f"Variance for Label 1: {stats['variance_label_1']:.4f}")
        print(f"Mean Prediction for Label 0: {stats['mean_label_0']:.4f}")
        print(f"Variance for Label 0: {stats['variance_label_0']:.4f}")

        print("\n===== Confusion Matrix =====")
        print(f"True Positives (TP): {TP:,} (Mean: {TP_mean:.4f})")
        print(f"False Positives (FP): {FP:,} (Mean: {FP_mean:.4f})")
        print(f"True Negatives (TN): {TN:,} (Mean: {TN_mean:.4f})")
        print(f"False Negatives (FN): {FN:,} (Mean: {FN_mean:.4f})")

        print("\n===== Sample Counts =====")
        print(f"Total Positives: {stats['Total Positives']:,}")
        print(f"Total Negatives: {stats['Total Negatives']:,}")
        print(f"Total Samples: {stats['Total Samples']:,}")

        print("\n===== Performance Metrics =====")
        print(f"Precision: {stats['Precision']:.4f}")
        print(f"Recall: {stats['Recall']:.4f}")
        print(f"F1 Score: {stats['F1 Score']:.4f}")

    return stats


# note: contains legacy parameters for backward compat
def get_param_groups(
    model,
    encoder_lr: float = 2e-4,
    decoder_lr: float = 4e-4,
    lm_head_lr: float = 0.0,
    feature_head_lr: float = 0.0,
    moe_router_lr: float = 0.0,
    moe_specialist_lr: float = 0.0,
    moe_bos_lr: float = 0.0,
    update_domain_embeddings: bool = False,
):
    encoder_params = []
    decoder_params = []
    lm_head_params = []
    feature_head_params = []
    moe_router_params = []
    moe_specialist_params = []
    moe_bos_params = []
    domain_params = []

    if hasattr(model, "peptide_conformer"):
        encoder_params.extend(list(model.peptide_conformer.parameters()))

    if hasattr(model, "mhc_conformer"):
        encoder_params.extend(list(model.mhc_conformer.parameters()))

    if hasattr(model, "peptide_embedder"):
        encoder_params.extend(list(model.peptide_embedder.parameters()))

    if hasattr(model, "mhc_embedder"):
        encoder_params.extend(list(model.mhc_embedder.parameters()))

    if hasattr(model, "bos_embedder"):
        encoder_params.extend(list(model.bos_embedder.parameters()))

    if hasattr(model, "use_segment_embedder") and model.use_segment_embedder:
        if hasattr(model, "segment_embedder"):
            encoder_params.extend(list(model.segment_embedder.parameters()))

    if hasattr(model, "peptide_transformer"):
        encoder_params.extend(list(model.peptide_transformer.parameters()))

    if hasattr(model, "mhc_transformer"):
        encoder_params.extend(list(model.mhc_transformer.parameters()))

    if hasattr(model, "peptide_transformers"):
        encoder_params.extend(list(model.peptide_transformers.parameters()))

    if hasattr(model, "mhc_transformers"):
        encoder_params.extend(list(model.mhc_transformers.parameters()))

    if hasattr(model, "enable_feature_tracks") and model.enable_feature_tracks:
        if hasattr(model, "feature_embedders") and len(model.feature_embedders) > 0:
            feature_embedder_params = list(model.feature_embedders.parameters())
            if feature_embedder_params:
                encoder_params.extend(feature_embedder_params)

    if hasattr(model, "concat_transformers"):
        decoder_params.extend(list(model.concat_transformers.parameters()))

    if hasattr(model, "fusion_conformer"):
        decoder_params.extend(list(model.fusion_conformer.parameters()))

    has_ffnn_head = hasattr(model, "ffnn_head")

    has_lm_head = (
        hasattr(model, "include_lm_head")
        and model.include_lm_head
        and hasattr(model, "lm_head")
    )

    has_feature_heads = (
        hasattr(model, "enable_feature_heads")
        and model.enable_feature_heads
        and hasattr(model, "feature_heads")
        and len(model.feature_heads) > 0
    )

    has_moe = hasattr(model, "use_moe") and model.use_moe

    if has_moe:
        if hasattr(model, "router_model") and model.router_model is not None:
            if not hasattr(model, "router_frozen") or not model.router_frozen:
                moe_router_params.extend(list(model.router_model.parameters()))

        if hasattr(model, "easy_transformers"):
            moe_specialist_params.extend(list(model.easy_transformers.parameters()))

        if hasattr(model, "hard_transformers"):
            moe_specialist_params.extend(list(model.hard_transformers.parameters()))

        if hasattr(model, "easy_ffnn_head"):
            moe_specialist_params.extend(list(model.easy_ffnn_head.parameters()))

        if hasattr(model, "hard_ffnn_head"):
            moe_specialist_params.extend(list(model.hard_ffnn_head.parameters()))

        if hasattr(model, "use_specialized_bos") and model.use_specialized_bos:
            if hasattr(model, "easy_bos_embedder"):
                moe_bos_params.extend(list(model.easy_bos_embedder.parameters()))

            if hasattr(model, "hard_bos_embedder"):
                moe_bos_params.extend(list(model.hard_bos_embedder.parameters()))

    if has_ffnn_head:
        if not (hasattr(model, "train_feature_heads_only") and model.train_feature_heads_only):
            decoder_params.extend(list(model.ffnn_head.parameters()))

    if hasattr(model, "episcan_ffnn_head"):
        decoder_params.extend(list(model.episcan_ffnn_head.parameters()))

    if has_lm_head:
        lm_head_params.extend(list(model.lm_head.parameters()))

        if hasattr(model, "enable_feature_tracks") and model.enable_feature_tracks:
            if hasattr(model, "feature_lm_heads") and len(model.feature_lm_heads) > 0:
                feature_lm_head_params = list(model.feature_lm_heads.parameters())
                if feature_lm_head_params:
                    lm_head_params.extend(feature_lm_head_params)

    if has_feature_heads:
        feature_head_params.extend(list(model.feature_heads.parameters()))

    if hasattr(model, "domain_embedder") and update_domain_embeddings:
        domain_params.extend(list(model.domain_embedder.parameters()))

    def _dedup_params(param_list):
        seen = set()
        unique = []

        for p in param_list:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                unique.append(p)

        return unique

    encoder_params = _dedup_params(encoder_params)
    decoder_params = _dedup_params(decoder_params)
    lm_head_params = _dedup_params(lm_head_params)
    feature_head_params = _dedup_params(feature_head_params)
    moe_router_params = _dedup_params(moe_router_params)
    moe_specialist_params = _dedup_params(moe_specialist_params)
    moe_bos_params = _dedup_params(moe_bos_params)
    domain_params = _dedup_params(domain_params)

    param_groups = []

    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
        })

    if decoder_params:
        param_groups.append({
            "params": decoder_params,
            "lr": decoder_lr,
        })

    if lm_head_params and has_lm_head and (not hasattr(model, "head_type") or model.head_type == "lm"):
        param_groups.append({
            "params": lm_head_params,
            "lr": lm_head_lr,
        })

    if feature_head_params:
        param_groups.append({
            "params": feature_head_params,
            "lr": feature_head_lr,
        })

    if moe_router_params:
        param_groups.append({
            "params": moe_router_params,
            "lr": moe_router_lr,
        })

    if moe_specialist_params:
        param_groups.append({
            "params": moe_specialist_params,
            "lr": moe_specialist_lr,
        })

    if moe_bos_params:
        param_groups.append({
            "params": moe_bos_params,
            "lr": moe_bos_lr,
        })

    if domain_params:
        param_groups.append({
            "params": domain_params,
            "lr": encoder_lr,
            "weight_decay": 0.0,
        })

    return param_groups

def train_mhcprime(
    model: nn.Module,
    train_df,
    tokenizer,
    n_epochs: int = 120,
    batch_size: int = 3072,
    num_pos_per_epoch: int = 200000,
    neg_pos_ratio: float = 1.0,
    num_workers: int = 16,
    persistent_workers: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_every_n_epochs: int = None,
    output_dir: str = None,
    loss_type: str = "logsmoothap",
    loss_hp: float = 10.0,
    encoder_lr: float = 2e-4,
    decoder_lr: float = 4e-4,
    lr_warmup_epochs: int = None,
    percentage_of_total_steps_for_warmup: float = None,
    seed: int = 42,
    optimizer_type: str = "AdamW",
    checkpoint_for_ft=None,
):
    model.to(device)

    scaler = GradScaler()

    if checkpoint_for_ft is not None:
        scaler.load_state_dict(checkpoint_for_ft["scaler"])

    train_dataset = PeptideMHCDataset(
        train_df,
        tokenizer,
    )

    if num_pos_per_epoch is None:
        num_pos_per_epoch = len(train_df.query("label == 1"))

    labels_array = train_df["label"].values

    loss_type = str(loss_type).lower()

    if loss_type not in {"bce", "mse", "logsmoothap", "spearman", "spearman_soft_rank"}:
        raise ValueError(
            "loss_type must be one of {'bce', 'mse', 'logsmoothap', 'spearman'}."
        )

    print("Dataset statistics:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Positive samples: {len(train_df.query('label == 1'))}")
    print(f"  Negative samples: {len(train_df.query('label == 0'))}")

    if len(train_df.query("label == 0")) > 0:
        print(
            f"  Pos/neg ratio in dataset: "
            f"{len(train_df.query('label == 1')) / len(train_df.query('label == 0')):.4f}"
        )

    if loss_type not in {"mse", "spearman", "spearman_soft_rank"}:
        print(f"  Target pos/neg ratio per epoch: {1.0 / neg_pos_ratio:.4f}")
        print(f"  Positives per epoch: {num_pos_per_epoch}")
        print(f"  Negatives per epoch: {int(num_pos_per_epoch * neg_pos_ratio)}")
        print(
            f"  Samples per epoch: "
            f"{num_pos_per_epoch + int(num_pos_per_epoch * neg_pos_ratio)}"
        )

    print("Creating SPSS data loader")

    train_loader, sampler = make_simple_negative_loader(
        dataset=train_dataset,
        labels=labels_array,
        batch_size=batch_size,
        num_pos_per_epoch=num_pos_per_epoch,
        neg_pos_ratio=neg_pos_ratio,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        seed=seed,
    )

    if loss_type == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss")

    elif loss_type == "mse":
        loss_fn = nn.MSELoss()
        print("Using MSE loss")

    elif loss_type == "logsmoothap":
        loss_fn = LogSmoothAPScore(alpha=loss_hp)
        print(f"Using LogSmoothAPScore with alpha={loss_hp}")

    elif loss_type in {"spearman", "spearman_soft_rank"}:
        loss_fn = SoftSpearmanLoss(tau=0.01)
        print("Using Soft Spearman loss with tau=0.01")

    param_groups = get_param_groups(
        model,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        lm_head_lr=0.0,
        update_domain_embeddings=False,
    )


    print(f"Using {optimizer_type} optimizer...")
    print()

    if optimizer_type != "AdamW":
        raise ValueError("Cleaned MHCPrime trainer currently supports optimizer_type='AdamW' only.")

    optimizer = optim.AdamW(param_groups)

    if checkpoint_for_ft is not None:
        optimizer.load_state_dict(checkpoint_for_ft["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)

    total_steps = len(train_loader) * n_epochs

    print("LEN train_loader", len(train_loader))
    print("n_epochs", n_epochs)
    print("total_steps", total_steps)

    if lr_warmup_epochs is not None:
        print(f"Using {lr_warmup_epochs} warmup steps for scheduler")
        num_warmup_steps = lr_warmup_epochs
    else:
        num_warmup_steps = total_steps // 10

    if percentage_of_total_steps_for_warmup is not None:
        print("Using percentage of total steps for warmup")
        num_warmup_steps = int(total_steps * percentage_of_total_steps_for_warmup)
        print("Total warmup steps:", num_warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0

    ap_history = {}
    training_history = {}
    pred_stats_history = {}
    negative_sampling_stats = {}

    if save_every_n_epochs is not None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    print("Starting MHCPrime training")

    if save_every_n_epochs is not None and output_dir is not None:
        save_path = os.path.join(output_dir, "model_e0.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at start: {save_path}")

    for epoch in range(n_epochs):
        model.train()

        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_scores = []
        all_indices = []

        train_iter = iter(train_loader)
        
        progress_bar = tqdm(
            train_iter,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{n_epochs} (spss)",
        )

        first_batch_check = True

        batch_idx = 0
        for batch in progress_bar:
            if first_batch_check:
                if "original_idx" not in batch:
                    print()
                    print("WARNING: 'original_idx' field is missing from batch.")
                    print("SPSS training can proceed, but sample-index tracking is unavailable.")
                    print(f"Available batch keys: {list(batch.keys())}")
                    print()
                first_batch_check = False

            original_indices = batch.pop("original_idx") if "original_idx" in batch else None

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            labels = batch.pop("label")

            if "allele" in batch:
                batch.pop("allele")

            with autocast(device):
                outputs = model(batch)

                if loss_type == "bce":
                    scores = outputs.view(-1)
                    loss = loss_fn(scores, labels.float())
                    preds = torch.sigmoid(scores)

                elif loss_type == "logsmoothap":
                    scores = outputs.view(-1)
                    loss = loss_fn(scores, labels.float())
                    preds = torch.sigmoid(scores)

                elif loss_type == "mse":
                    scores = outputs.view(-1)
                    loss = loss_fn(scores, labels.float())
                    preds = scores.detach()

                elif loss_type in {"spearman", "spearman_soft_rank"}:
                    scores = outputs.view(-1)
                    loss = loss_fn(scores, labels.float())
                    preds = scores.detach()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_scores.extend(scores.detach().cpu().numpy())

            if original_indices is not None:
                all_indices.extend(original_indices.cpu().numpy())

            batch_idx += 1

        avg_loss = total_loss / len(train_loader)

        if loss_type in {"mse", "spearman", "spearman_soft_rank"}:
            all_labels_t = torch.tensor(all_labels, dtype=torch.float32)
            all_preds_t = torch.tensor(all_preds, dtype=torch.float32)

            mae = torch.mean(torch.abs(all_preds_t - all_labels_t)).item()
            mse = torch.mean((all_preds_t - all_labels_t) ** 2).item()
            rmse = math.sqrt(mse)

            label_var = torch.var(all_labels_t).item()
            r2 = 1 - mse / label_var if label_var > 0 else 0.0

            errors = np.array(all_preds) - np.array(all_labels)

            pred_stats = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mean_predictions": np.mean(all_preds),
                "std_predictions": np.std(all_preds),
                "mean_targets": np.mean(all_labels),
                "std_targets": np.std(all_labels),
                "max_error": np.max(np.abs(errors)),
                "min_error": np.min(np.abs(errors)),
                "error_std": np.std(errors),
                "error_25th_percentile": np.percentile(np.abs(errors), 25),
                "error_50th_percentile": np.percentile(np.abs(errors), 50),
                "error_75th_percentile": np.percentile(np.abs(errors), 75),
                "error_90th_percentile": np.percentile(np.abs(errors), 90),
            }

            print(f"\nEpoch {epoch + 1} Regression Summary:")
            print(
                f"Loss: {avg_loss:.6f} | "
                f"MAE: {mae:.6f} | "
                f"MSE: {mse:.6f} | "
                f"RMSE: {rmse:.6f} | "
                f"R²: {r2:.6f}"
            )
            print(
                f"Predictions - Mean: {pred_stats['mean_predictions']:.4f}, "
                f"Std: {pred_stats['std_predictions']:.4f}"
            )
            print(
                f"Targets - Mean: {pred_stats['mean_targets']:.4f}, "
                f"Std: {pred_stats['std_targets']:.4f}"
            )
            print(
                f"Error Distribution - Median: "
                f"{pred_stats['error_50th_percentile']:.4f}, "
                f"Std: {pred_stats['error_std']:.4f}"
            )
            print(
                f"Error Percentiles - "
                f"25%: {pred_stats['error_25th_percentile']:.4f}, "
                f"75%: {pred_stats['error_75th_percentile']:.4f}, "
                f"90%: {pred_stats['error_90th_percentile']:.4f}"
            )

            ap_history[epoch] = r2
            training_history[epoch] = avg_loss
            pred_stats_history[epoch] = pred_stats

        else:
            threshold = 0.5

            preds_bin = (np.array(all_preds) > threshold).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                np.array(all_labels),
                preds_bin,
                average="binary",
            )

            average_precision = average_precision_score(all_labels, all_preds)

            pred_stats = compute_label_stats(
                all_labels,
                all_scores,
                threshold=0.0,
                print_stats=True,
            )

            ap_history[epoch] = average_precision
            training_history[epoch] = avg_loss
            pred_stats_history[epoch] = pred_stats

            print(f"\nEpoch {epoch + 1} Summary:")
            print(
                f"Loss: {avg_loss:.6f} | "
                f"Precision: {precision:.6f} | "
                f"Recall: {recall:.6f} | "
                f"F1 Score: {f1:.6f}"
            )
            print(f"Average Precision: {average_precision:.6f}")

            if "mean_label_1" in pred_stats and "mean_label_0" in pred_stats:
                print(
                    f"Prediction means - "
                    f"Positive: {pred_stats['mean_label_1']:.4f}, "
                    f"Negative: {pred_stats['mean_label_0']:.4f}"
                )

                separation = pred_stats["mean_label_1"] - pred_stats["mean_label_0"]
                print(f"Class separation: {separation:.4f}")

                if "FP_mean" in pred_stats and "TN_mean" in pred_stats:
                    fp_tn_ratio = pred_stats["FP_mean"] / (pred_stats["TN_mean"] + 1e-10)
                    print(f"False positive to true negative score ratio: {fp_tn_ratio:.4f}")

        if (
            save_every_n_epochs is not None
            and output_dir is not None
            and (epoch + 1) % save_every_n_epochs == 0
        ):
            save_path = os.path.join(output_dir, f"model_e{epoch + 1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved: {save_path}")

    training_stats = {
        "AP": ap_history,
        "Loss": training_history,
        "Pred_Stats": pred_stats_history,
        "Negative_Sampling": negative_sampling_stats,
    }

    if output_dir is not None:
        save_final_checkpoint(
            f"{output_dir}/final_state.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            extra=None,
            save_rng=False,
        )

    return {
        "training_stats": training_stats,
    }

