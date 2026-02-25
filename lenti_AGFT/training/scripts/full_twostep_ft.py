#!/usr/bin/env python3
"""Two-stage fine-tune of AlphaGenome on K562 LentiMPRA data.

Stage 1: Head-only training (frozen backbone) with AdamW via create_optimizer.
Stage 2: Full model fine-tuning (unfrozen backbone) with lower LR.

Uses the installed alphagenome_ft package (correct multi_transform optimizer).

Usage:
  python full_twostep_ft.py --name my_model_v1
  python full_twostep_ft.py --name my_model_v1 --lr 1e-4 --stage2-lr 1e-5 --stage2-epochs 50
  python full_twostep_ft.py --name my_model_v1  # resubmit: reloads saved args
"""

import copy
import hashlib
import os
import pickle
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
from scipy.stats import pearsonr, spearmanr

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "alphagenome_FT_MPRA"))

# Use shared HF cache (weights + token already set up in Liver_AGFT)
os.environ["HF_HOME"] = os.path.expanduser("~/Liver_AGFT/.weights/huggingface")

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from alphagenome_ft import (
    CustomHead,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    create_model_with_heads,
    custom_heads as custom_heads_module,
)
from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
from alphagenome_ft.finetune.train import create_optimizer

from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader

# ============================================================
# Constants
# ============================================================
HEAD_NAME = "mpra_head"
DIAG_HEAD = "encoder_diag"
RESULTS_BASE = REPO_ROOT / "lenti_AGFT" / "training" / "results"
MODEL_CLUB_DIR = REPO_ROOT / "lenti_AGFT" / "training" / "pearson's_model_club"
CACHE_DIR = REPO_ROOT / "lenti_AGFT" / "training" / "cache"
DATA_DIR = str(REPO_ROOT / "test_run_lenti_data")
PROMOTER_CONSTRUCT_LENGTH = 281

DEFAULTS = {
    "cell_type": "K562",
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "center_bp": 256,
    "pooling_type": "flatten",
    "nl_size": 1024,
    "dropout": 0.1,
    "activation": "relu",
    "early_stopping": 5,
    "random_shift": True,
    "random_shift_likelihood": 0.5,
    "reverse_complement": True,
    # Stage 2 (aligned with mpra_K562.json two_stage defaults)
    "stage2_lr": 1e-5,
    "stage2_epochs": 50,
    "stage2_patience": 5,
}


# ============================================================
# Diagnostic head (returns raw encoder output, same as Liver)
# ============================================================

class DiagHead(CustomHead):
    """Passthrough head that returns raw encoder embeddings."""
    def predict(self, embeddings, organism_index, **kwargs):
        if not hasattr(embeddings, 'encoder_output') or embeddings.encoder_output is None:
            raise ValueError("encoder_output not available")
        return embeddings.encoder_output

    def loss(self, predictions, batch):
        return {'loss': jnp.array(0.0)}


# ============================================================
# CLI & config
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AlphaGenome on LentiMPRA")
    parser.add_argument("--name", required=True, help="Model name (used for results directory)")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--cache-embeddings", action="store_true",
                        help="Use cached encoder embeddings (generates if not found)")

    # Hyperparameters (CLI flags override config which overrides DEFAULTS)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)

    # Head architecture
    parser.add_argument("--nl-size", type=int, nargs='+', default=None,
                        help="Hidden layer size(s). E.g. --nl-size 1024 or --nl-size 1024 512")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--pooling-type", type=str, default=None,
                        choices=["flatten", "sum", "mean", "max", "center"])
    parser.add_argument("--center-bp", type=int, default=None)

    # Augmentation
    parser.add_argument("--no-reverse-complement", action="store_true")
    parser.add_argument("--no-random-shift", action="store_true")

    # Stage 2
    parser.add_argument("--stage2-lr", type=float, default=None, help="Stage 2 learning rate (default: 1e-5)")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Stage 2 max epochs (default: 50)")
    parser.add_argument("--stage2-patience", type=int, default=None, help="Stage 2 early stopping patience")
    parser.add_argument("--skip-stage2", action="store_true", help="Run stage 1 only (head-only)")

    return parser.parse_args()


def load_config(config_path, defaults):
    """Load JSON config and merge with defaults."""
    if config_path is None:
        return dict(defaults)
    with open(config_path) as f:
        cfg = json.load(f)
    hp = dict(defaults)
    if "cell_type" in cfg:
        hp["cell_type"] = cfg["cell_type"]
    data = cfg.get("data", {})
    hp["batch_size"] = data.get("batch_size", hp["batch_size"])
    hp["random_shift"] = data.get("random_shift", hp["random_shift"])
    hp["random_shift_likelihood"] = data.get("random_shift_likelihood", hp["random_shift_likelihood"])
    hp["reverse_complement"] = data.get("reverse_complement", hp["reverse_complement"])
    model = cfg.get("model", {})
    hp["center_bp"] = model.get("center_bp", hp["center_bp"])
    hp["pooling_type"] = model.get("pooling_type", hp["pooling_type"])
    nl = model.get("nl_size", hp["nl_size"])
    hp["nl_size"] = int(nl) if isinstance(nl, str) else nl
    hp["dropout"] = model.get("do", hp["dropout"])
    hp["activation"] = model.get("activation", hp["activation"])
    training = cfg.get("training", {})
    hp["num_epochs"] = training.get("num_epochs", hp["num_epochs"])
    hp["learning_rate"] = training.get("learning_rate", hp["learning_rate"])
    hp["weight_decay"] = training.get("weight_decay", hp["weight_decay"])
    hp["early_stopping"] = training.get("early_stopping_patience", hp["early_stopping"])
    two_stage = cfg.get("two_stage", {})
    hp["stage2_lr"] = two_stage.get("second_stage_lr", hp["stage2_lr"])
    hp["stage2_epochs"] = two_stage.get("second_stage_epochs", hp["stage2_epochs"])
    hp["stage2_patience"] = two_stage.get("early_stopping_patience", hp["stage2_patience"])
    return hp


def apply_cli_overrides(hp, args):
    """CLI flags override config/defaults. Priority: CLI > config > DEFAULTS."""
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.epochs is not None:
        hp["num_epochs"] = args.epochs
    if args.patience is not None:
        hp["early_stopping"] = args.patience
    if args.nl_size is not None:
        hp["nl_size"] = args.nl_size if len(args.nl_size) > 1 else args.nl_size[0]
    if args.dropout is not None:
        hp["dropout"] = args.dropout
    if args.activation is not None:
        hp["activation"] = args.activation
    if args.pooling_type is not None:
        hp["pooling_type"] = args.pooling_type
    if args.center_bp is not None:
        hp["center_bp"] = args.center_bp
    if args.no_reverse_complement:
        hp["reverse_complement"] = False
    if args.no_random_shift:
        hp["random_shift"] = False
    if args.stage2_lr is not None:
        hp["stage2_lr"] = args.stage2_lr
    if args.stage2_epochs is not None:
        hp["stage2_epochs"] = args.stage2_epochs
    if args.stage2_patience is not None:
        hp["stage2_patience"] = args.stage2_patience


# ============================================================
# Cache generation
# ============================================================

def generate_cache(model, hp, strand_reindexing):
    """Generate encoder embedding cache. Returns cache file path."""
    cache_file = CACHE_DIR / f"{hp['cell_type']}_embeddings.pkl"
    if cache_file.exists():
        print(f"Cache already exists: {cache_file}")
        return str(cache_file)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    # Load all sequences (all folds, rev=0 — same filter as LentiMPRADataset)
    data_path = os.path.join(DATA_DIR, f"{hp['cell_type']}.tsv")
    df = pd.read_csv(data_path, sep="\t")
    df = df[df["rev"] == 0].reset_index(drop=True)

    promoter_seq = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
    rand_barcode = "AGAGACTGAGGCCAC"

    # JIT-compiled extraction via diagnostic head
    @jax.jit
    def extract_step(params, state, sequences, organism_index):
        predictions = model._predict(
            params, state, sequences, organism_index,
            negative_strand_mask=jnp.zeros(sequences.shape[0], dtype=bool),
            strand_reindexing=strand_reindexing,
        )
        return predictions[DIAG_HEAD]

    cache = {}
    batch_size = hp["batch_size"]
    n = len(df)
    print(f"Generating cache for {n} sequences (one-time cost)...")

    with model._device_context:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs, batch_hashes = [], []

            for idx in range(start, end):
                seq_str = df.iloc[idx]['seq'] + promoter_seq + rand_barcode
                batch_hashes.append(hashlib.sha256(seq_str.encode()).hexdigest())
                seq_oh = model._one_hot_encoder.encode(seq_str)
                batch_seqs.append(jnp.array(seq_oh))

            # Pad and stack
            max_len = max(s.shape[0] for s in batch_seqs)
            padded = []
            for s in batch_seqs:
                if s.shape[0] < max_len:
                    s = jnp.concatenate([s, jnp.zeros((max_len - s.shape[0], 4))], axis=0)
                padded.append(s)
            batch_seq = jnp.stack(padded)
            org_idx = jnp.zeros(len(batch_seqs), dtype=jnp.int32)

            encoder_out = extract_step(model._params, model._state, batch_seq, org_idx)
            encoder_out_np = np.array(encoder_out, dtype=np.float32)

            for i, h in enumerate(batch_hashes):
                cache[h] = encoder_out_np[i]

            done = min(end, n)
            if (start // batch_size) % 200 == 0:
                print(f"  {done}/{n} sequences cached...")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved: {cache_file} ({len(cache)} sequences)")
    return str(cache_file)


# ============================================================
# Head-only functions for cached embeddings training
# ============================================================

def create_head_only_fns(model, head_name):
    """Create JIT-compiled head-only forward and gradient functions.

    Uses the same pattern as alphagenome_FT_MPRA/src/training.py
    _create_head_only_functions. Runs ONLY the head on cached encoder
    outputs — no backbone forward pass at all.
    """
    head_config = model._head_configs[head_name]
    num_organisms = len(model._metadata)

    @hk.transform_with_state
    def head_forward_and_loss(encoder_output, organism_index, targets=None):
        embeddings = ExtendedEmbeddings(
            embeddings_1bp=None,
            embeddings_128bp=None,
            encoder_output=encoder_output,
        )
        with hk.name_scope('head'):
            head = custom_heads_module.create_custom_head(
                head_name,
                metadata=head_config.metadata,
                num_organisms=num_organisms,
            )
            predictions = head.predict(embeddings, organism_index)
            if targets is not None:
                loss_dict = head.loss(predictions, {'targets': targets})
                return predictions, loss_dict
            return predictions, None

    @jax.jit
    def forward_fn(params, state, encoder_output, organism_index):
        (predictions, _), _ = head_forward_and_loss.apply(
            params, state, None, encoder_output, organism_index, None
        )
        return predictions

    @jax.jit
    def grad_fn(params, state, encoder_output, organism_index, targets):
        def loss_inner(p):
            (_, loss_dict), _ = head_forward_and_loss.apply(
                p, state, None, encoder_output, organism_index, targets
            )
            return loss_dict['loss']
        loss_value, grads = jax.value_and_grad(loss_inner)(params)
        return grads, loss_value

    return forward_fn, grad_fn


# ============================================================
# Batch adapter (MPRADataLoader -> alphagenome_ft format)
# ============================================================

def adapt_batch(batch):
    """Convert MPRADataLoader batch to alphagenome_ft format."""
    return {
        "sequences": batch["seq"],
        "organism_index": batch["organism_index"],
        "negative_strand_mask": jnp.zeros(batch["seq"].shape[0], dtype=bool),
        f"targets_{HEAD_NAME}": batch["y"],
    }


# ============================================================
# JIT-compiled training step factory (same as Liver_AGFT)
# ============================================================

def make_train_step(model, optimizer, loss_fn, head_name, strand_reindexing):
    """Create a JIT-compiled train step. Recreated when optimizer changes."""

    @jax.jit
    def train_step(params, state, opt_state, batch):
        def loss_fn_inner(p):
            predictions = model._predict(
                p, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            head_loss_dict = loss_fn(
                predictions[head_name],
                {"targets": batch[f"targets_{head_name}"],
                 "organism_index": batch["organism_index"]},
            )
            return head_loss_dict["loss"]

        loss_value, grads = jax.value_and_grad(loss_fn_inner)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value

    return train_step


# ============================================================
# Metrics & figures
# ============================================================

def compute_metrics(preds, targets):
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    mse = float(np.mean((preds - targets) ** 2))
    return {"pearson_r": float(r), "spearman_rho": float(rho), "mse": mse}


def make_summary_figure(
    epoch1_preds, epoch1_targets,
    best_preds, best_targets, metrics_best,
    train_loss_hist, valid_loss_hist,
    best_epoch, save_path, run_name,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    all_vals = [best_targets, best_preds]
    if epoch1_preds is not None:
        all_vals.append(epoch1_preds)
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lims = [vmin, vmax]

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        axes[0].scatter(epoch1_targets, epoch1_preds, alpha=0.1, s=1, rasterized=True)
        axes[0].plot(lims, lims, "r--", linewidth=0.5)
        axes[0].set_title(f"Epoch 1: r={m1['pearson_r']:.3f}, rho={m1['spearman_rho']:.3f}")
    else:
        axes[0].set_title("Epoch 1: N/A")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)

    axes[1].scatter(best_targets, best_preds, alpha=0.1, s=1, rasterized=True)
    axes[1].plot(lims, lims, "r--", linewidth=0.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Best (ep {best_epoch}): r={metrics_best['pearson_r']:.3f}, rho={metrics_best['spearman_rho']:.3f}")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)

    epochs_range = range(1, len(train_loss_hist) + 1)
    axes[2].plot(epochs_range, train_loss_hist, label="Train")
    axes[2].plot(epochs_range, valid_loss_hist, label="Valid")
    axes[2].axvline(best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best (epoch {best_epoch})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss (MSE)")
    axes[2].set_title("Training Loss"); axes[2].legend()

    plt.suptitle(f"AlphaGenome FT -> LentiMPRA K562 [{run_name}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved to {save_path}")


def update_model_club(name, metrics_best, preds_best, targets_best, hp):
    MODEL_CLUB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODEL_CLUB_DIR / "best_models.csv"
    fields = ["name", "pearson_r", "spearman_rho", "mse", "cell_type", "timestamp"]
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    current_best = max((float(r["pearson_r"]) for r in rows), default=-1.0)
    new_r = metrics_best["pearson_r"]
    rows.append({
        "name": name, "pearson_r": f"{new_r:.6f}",
        "spearman_rho": f"{metrics_best['spearman_rho']:.6f}",
        "mse": f"{metrics_best['mse']:.6f}",
        "cell_type": hp["cell_type"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if new_r > current_best:
        print(f"New best model! Pearson r = {new_r:.4f} (previous best: {current_best:.4f})")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(targets_best, preds_best, alpha=0.1, s=1, rasterized=True)
        lims = [min(targets_best.min(), preds_best.min()), max(targets_best.max(), preds_best.max())]
        ax.plot(lims, lims, "r--", linewidth=0.5)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"Best model: {name} (r={new_r:.4f})")
        ax.text(0.05, 0.95, f"r = {new_r:.4f}\nrho = {metrics_best['spearman_rho']:.4f}\nMSE = {metrics_best['mse']:.4f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout(); plt.savefig(MODEL_CLUB_DIR / "best_model_summary.png", dpi=150); plt.close()
    else:
        print(f"Model r = {new_r:.4f} did not beat current best ({current_best:.4f})")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    results_dir = RESULTS_BASE / args.name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist args so resubmits with just --name recover original flags
    args_file = results_dir / "args.json"
    has_cli_overrides = any([
        args.config, args.cache_embeddings, args.lr, args.weight_decay,
        args.batch_size, args.epochs, args.patience, args.nl_size,
        args.dropout, args.activation, args.pooling_type, args.center_bp,
        args.no_reverse_complement, args.no_random_shift,
        args.stage2_lr, args.stage2_epochs, args.stage2_patience, args.skip_stage2,
    ])
    if has_cli_overrides or not args_file.exists():
        # First run or explicit new args — build hp and save
        hp = load_config(args.config, DEFAULTS)
        apply_cli_overrides(hp, args)
        use_cache = args.cache_embeddings
        skip_stage2 = args.skip_stage2
        saved = {"hp": hp, "config": args.config, "cache_embeddings": use_cache, "skip_stage2": skip_stage2}
        with open(args_file, "w") as f:
            json.dump(saved, f, indent=2)
    else:
        # Resubmit with just --name — reload saved args
        with open(args_file) as f:
            saved = json.load(f)
        hp = saved["hp"]
        use_cache = saved.get("cache_embeddings", False)
        skip_stage2 = saved.get("skip_stage2", False)
        print(f"Loaded saved args from {args_file}")

    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"Model name: {args.name}")
    print(f"Cell type: {hp['cell_type']}")
    print(f"Mode: {'cached embeddings' if use_cache else 'full model'}")
    print(f"Results dir: {results_dir}")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    # ---- Register heads ----
    nl_size = hp["nl_size"] if isinstance(hp["nl_size"], list) else [hp["nl_size"]]
    head_metadata = {
        "center_bp": hp["center_bp"],
        "pooling_type": hp["pooling_type"],
        "nl_size": nl_size,
        "do": hp["dropout"],
        "activation": hp["activation"],
    }
    register_custom_head(
        HEAD_NAME, EncoderMPRAHead,
        CustomHeadConfig(
            type=CustomHeadType.GENOME_TRACKS,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1, metadata=head_metadata,
        ),
    )
    # Diagnostic head for cache generation (returns raw encoder output)
    heads_list = [HEAD_NAME]
    if use_cache:
        register_custom_head(
            DIAG_HEAD, DiagHead,
            CustomHeadConfig(
                type=CustomHeadType.GENOME_TRACKS,
                output_type=dna_output.OutputType.RNA_SEQ,
                num_tracks=1536,
            ),
        )
        heads_list.append(DIAG_HEAD)

    # ---- Model ----
    from huggingface_hub import snapshot_download
    hf_path = snapshot_download("google/alphagenome-all-folds")
    print(f"Weights: {hf_path}")

    model = create_model_with_heads(
        "all_folds", heads=heads_list,
        checkpoint_path=hf_path,
        use_encoder_output=True,
        init_seq_len=PROMOTER_CONSTRUCT_LENGTH,
    )
    model.freeze_except_head(HEAD_NAME)

    head_params = model.get_head_parameters(HEAD_NAME)
    head_count = sum(p.size for p in jax.tree_util.tree_leaves(head_params))
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable (head only): {head_count:,}")

    # ---- Organism / strand setup ----
    organism_enum = dna_model.Organism.HOMO_SAPIENS
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    # ---- Generate cache if needed ----
    cache_file = None
    if use_cache:
        cache_file = generate_cache(model, hp, strand_reindexing)

    # ---- Data ----
    ds_kwargs = dict(model=model, path_to_data=DATA_DIR, cell_type=hp["cell_type"])
    cache_kwargs = dict(use_cached_embeddings=use_cache, cache_file=cache_file) if use_cache else {}

    train_ds = LentiMPRADataset(
        **ds_kwargs, split="train",
        random_shift=hp["random_shift"] if not use_cache else False,
        random_shift_likelihood=hp.get("random_shift_likelihood", 0.5),
        reverse_complement=hp["reverse_complement"] if not use_cache else False,
        **cache_kwargs,
    )
    val_ds = LentiMPRADataset(
        **ds_kwargs, split="val",
        random_shift=False, reverse_complement=False,
        **cache_kwargs,
    )
    test_ds = LentiMPRADataset(
        **ds_kwargs, split="test",
        random_shift=False, reverse_complement=False,
        **cache_kwargs,
    )

    train_loader = MPRADataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = MPRADataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False)
    test_loader = MPRADataLoader(test_ds, batch_size=hp["batch_size"], shuffle=False)

    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_ds)} samples, {len(test_loader)} batches")

    # ---- Setup mode-specific step functions ----
    if use_cache:
        # Head-only mode: no backbone, just head on cached encoder outputs
        forward_fn, grad_fn = create_head_only_fns(model, HEAD_NAME)
        optimizer = optax.adam(hp["learning_rate"])
        opt_state = optimizer.init(model._params)

        def do_train_step(raw_batch):
            nonlocal opt_state
            grads, loss_val = grad_fn(
                model._params, model._state,
                raw_batch["encoder_output"], raw_batch["organism_index"],
                raw_batch["y"],
            )
            updates, opt_state = optimizer.update(grads, opt_state, model._params)
            model._params = optax.apply_updates(model._params, updates)
            return float(loss_val)

        def do_eval_step(raw_batch):
            # Compute loss via grad_fn (discarding grads)
            _, loss_val = grad_fn(
                model._params, model._state,
                raw_batch["encoder_output"], raw_batch["organism_index"],
                raw_batch["y"],
            )
            return float(loss_val)

        def do_predict_batch(raw_batch):
            preds = forward_fn(
                model._params, model._state,
                raw_batch["encoder_output"], raw_batch["organism_index"],
            )
            while preds.ndim > 1:
                preds = preds.squeeze(-1)
            return np.array(preds, dtype=np.float32), np.array(raw_batch["y"], dtype=np.float32)

    else:
        # Full model mode 
        loss_fn = model.create_loss_fn_for_head(HEAD_NAME)
        optimizer = create_optimizer(
            model._params, trainable_head_names=[HEAD_NAME],
            learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
            heads_only=True,
        )
        opt_state = optimizer.init(model._params)
        train_step_fn = make_train_step(model, optimizer, loss_fn, HEAD_NAME, strand_reindexing)

        @jax.jit
        def eval_step(params, state, batch):
            predictions = model._predict(
                params, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            loss_dict = loss_fn(
                predictions[HEAD_NAME],
                {"targets": batch[f"targets_{HEAD_NAME}"],
                 "organism_index": batch["organism_index"]},
            )
            return loss_dict["loss"]

        @jax.jit
        def predict_step(params, state, batch):
            predictions = model._predict(
                params, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            return predictions[HEAD_NAME]

        def do_train_step(raw_batch):
            nonlocal opt_state
            batch = adapt_batch(raw_batch)
            model._params, opt_state, loss_val = train_step_fn(
                model._params, model._state, opt_state, batch
            )
            return float(loss_val)

        def do_eval_step(raw_batch):
            batch = adapt_batch(raw_batch)
            return float(eval_step(model._params, model._state, batch))

        def do_predict_batch(raw_batch):
            batch = adapt_batch(raw_batch)
            preds = predict_step(model._params, model._state, batch)
            while preds.ndim > 1:
                preds = preds.squeeze(-1)
            return np.array(preds, dtype=np.float32), np.array(raw_batch["y"], dtype=np.float32)

    # ---- Helper: collect predictions ----
    def collect_preds(loader):
        preds_list, targets_list = [], []
        for raw_batch in loader:
            p, t = do_predict_batch(raw_batch)
            preds_list.append(p)
            targets_list.append(t)
        return np.concatenate(preds_list), np.concatenate(targets_list)

    # ============================================================
    # Training loop (same for both modes)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Training: head-only ({'cached embeddings' if use_cache else 'frozen encoder'})")
    print(f"  LR={hp['learning_rate']}, WD={hp['weight_decay']}, BS={hp['batch_size']}")
    print(f"  Epochs={hp['num_epochs']}, Patience={hp['early_stopping']}")
    print(f"{'='*60}")

    best_valid_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    best_preds = None
    best_targets = None
    epoch1_preds = None
    epoch1_targets = None
    train_loss_history = []
    valid_loss_history = []
    patience = hp["early_stopping"]

    with model._device_context:
        for epoch in range(1, hp["num_epochs"] + 1):
            # Train
            train_losses = []
            for raw_batch in train_loader:
                loss_val = do_train_step(raw_batch)
                train_losses.append(loss_val)
            train_loss = np.mean(train_losses)

            # Validate
            valid_losses = []
            for raw_batch in val_loader:
                vl = do_eval_step(raw_batch)
                valid_losses.append(vl)
            valid_loss = np.mean(valid_losses)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                  f"| valid_loss={valid_loss:.4f}", end="")

            # Snapshot after epoch 1
            if epoch == 1:
                epoch1_preds, epoch1_targets = collect_preds(test_loader)

            # Early stopping + checkpointing
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                best_epoch = epoch
                best_preds, best_targets = collect_preds(test_loader)
                model.save_checkpoint(
                    str(checkpoint_dir / "best"), save_full_model=False,
                )
                print(" * (saved)")
            else:
                epochs_no_improve += 1
                print(f"  (no improve {epochs_no_improve}/{patience})")

            if patience > 0 and epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # If no checkpoint saved, use final weights
    if best_preds is None:
        with model._device_context:
            best_preds, best_targets = collect_preds(test_loader)
        best_epoch = len(train_loss_history)

    # ---- Stage 1 results ----
    s1_metrics = compute_metrics(best_preds, best_targets)
    print(f"\nStage 1 Test (best epoch {best_epoch}): "
          f"r={s1_metrics['pearson_r']:.4f}, "
          f"rho={s1_metrics['spearman_rho']:.4f}, "
          f"MSE={s1_metrics['mse']:.4f}")

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        print(f"Stage 1 Test (epoch 1): r={m1['pearson_r']:.4f}, "
              f"rho={m1['spearman_rho']:.4f}, MSE={m1['mse']:.4f}")

    make_summary_figure(
        epoch1_preds, epoch1_targets,
        best_preds, best_targets, s1_metrics,
        train_loss_history, valid_loss_history,
        best_epoch, results_dir / "summary_stage1.png", f"{args.name} (stage 1)",
    )

    # ============================================================
    # Stage 2: Full model fine-tuning (unfrozen backbone)
    # ============================================================
    s2_metrics = None
    s2_best_preds = None
    s2_best_targets = None
    s2_train_loss_history = []
    s2_valid_loss_history = []
    s2_best_epoch = 0

    if not skip_stage2:
        print(f"\n{'='*60}")
        print(f"Stage 2: Full model fine-tuning (unfrozen backbone)")
        print(f"  LR={hp['stage2_lr']}, WD={hp['weight_decay']}, BS={hp['batch_size']}")
        print(f"  Epochs={hp['stage2_epochs']}, Patience={hp['stage2_patience']}")
        print(f"{'='*60}")

        # Reload best stage 1 checkpoint
        s1_ckpt = checkpoint_dir / "best"
        if s1_ckpt.exists():
            model.load_checkpoint(str(s1_ckpt))
            print(f"Loaded stage 1 best checkpoint from {s1_ckpt}")

        # Unfreeze backbone
        model.unfreeze_parameters(
            unfreeze_prefixes=['sequence_encoder', 'transformer_tower', 'sequence_decoder']
        )
        print("Unfroze backbone parameters")

        # New optimizer for all params (heads_only=False -> plain adamw on everything)
        s2_optimizer = create_optimizer(
            model._params, trainable_head_names=[HEAD_NAME],
            learning_rate=hp["stage2_lr"], weight_decay=hp["weight_decay"],
            heads_only=False,
        )
        s2_opt_state = s2_optimizer.init(model._params)

        # Rebuild JIT-compiled train step with new optimizer
        loss_fn = model.create_loss_fn_for_head(HEAD_NAME)
        s2_train_step_fn = make_train_step(model, s2_optimizer, loss_fn, HEAD_NAME, strand_reindexing)

        @jax.jit
        def s2_eval_step(params, state, batch):
            predictions = model._predict(
                params, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            loss_dict = loss_fn(
                predictions[HEAD_NAME],
                {"targets": batch[f"targets_{HEAD_NAME}"],
                 "organism_index": batch["organism_index"]},
            )
            return loss_dict["loss"]

        s2_best_valid_loss = float("inf")
        s2_epochs_no_improve = 0
        s2_patience = hp["stage2_patience"]

        with model._device_context:
            for epoch in range(1, hp["stage2_epochs"] + 1):
                # Train
                train_losses = []
                for raw_batch in train_loader:
                    batch = adapt_batch(raw_batch)
                    model._params, s2_opt_state, loss_val = s2_train_step_fn(
                        model._params, model._state, s2_opt_state, batch
                    )
                    train_losses.append(float(loss_val))
                s2_train_loss = np.mean(train_losses)

                # Validate
                valid_losses = []
                for raw_batch in val_loader:
                    batch = adapt_batch(raw_batch)
                    vl = float(s2_eval_step(model._params, model._state, batch))
                    valid_losses.append(vl)
                s2_valid_loss = np.mean(valid_losses)

                s2_train_loss_history.append(s2_train_loss)
                s2_valid_loss_history.append(s2_valid_loss)

                print(f"S2 Epoch {epoch:03d} | train_loss={s2_train_loss:.4f} "
                      f"| valid_loss={s2_valid_loss:.4f}", end="")

                if s2_valid_loss < s2_best_valid_loss:
                    s2_best_valid_loss = s2_valid_loss
                    s2_epochs_no_improve = 0
                    s2_best_epoch = epoch
                    s2_best_preds, s2_best_targets = collect_preds(test_loader)
                    model.save_checkpoint(
                        str(checkpoint_dir / "best_stage2"), save_full_model=True,
                    )
                    print(" * (saved)")
                else:
                    s2_epochs_no_improve += 1
                    print(f"  (no improve {s2_epochs_no_improve}/{s2_patience})")

                if s2_patience > 0 and s2_epochs_no_improve >= s2_patience:
                    print(f"\nStage 2 early stopping at epoch {epoch}")
                    break

        if s2_best_preds is not None:
            s2_metrics = compute_metrics(s2_best_preds, s2_best_targets)
            print(f"\nStage 2 Test (best epoch {s2_best_epoch}): "
                  f"r={s2_metrics['pearson_r']:.4f}, "
                  f"rho={s2_metrics['spearman_rho']:.4f}, "
                  f"MSE={s2_metrics['mse']:.4f}")

            make_summary_figure(
                best_preds, best_targets,  # stage 1 best as "epoch 1" reference
                s2_best_preds, s2_best_targets, s2_metrics,
                s2_train_loss_history, s2_valid_loss_history,
                s2_best_epoch, results_dir / "summary_stage2.png", f"{args.name} (stage 2)",
            )

    # ============================================================
    # Final metrics (use stage 2 if available, else stage 1)
    # ============================================================
    final_metrics = s2_metrics if s2_metrics is not None else s1_metrics
    final_preds = s2_best_preds if s2_best_preds is not None else best_preds
    final_targets = s2_best_targets if s2_best_targets is not None else best_targets

    metrics_out = {
        "name": args.name,
        "config_path": args.config,
        "cached_embeddings": use_cache,
        "hyperparameters": hp,
        "stage1_test": s1_metrics,
        "stage1_best_epoch": best_epoch,
        "stage1_epochs_trained": len(train_loss_history),
        "stage2_test": s2_metrics,
        "stage2_best_epoch": s2_best_epoch,
        "stage2_epochs_trained": len(s2_train_loss_history),
        "best_epoch_test": final_metrics,
        "history": {
            "stage1_train_loss": [float(v) for v in train_loss_history],
            "stage1_valid_loss": [float(v) for v in valid_loss_history],
            "stage2_train_loss": [float(v) for v in s2_train_loss_history],
            "stage2_valid_loss": [float(v) for v in s2_valid_loss_history],
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {results_dir / 'metrics.json'}")

    update_model_club(args.name, final_metrics, final_preds, final_targets, hp)
    print("\nDone!")


if __name__ == "__main__":
    main()
