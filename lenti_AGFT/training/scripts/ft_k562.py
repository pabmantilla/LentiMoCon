"""
Fine-tune AlphaGenome on K562 LentiMPRA with post-training evaluation.

Runs finetune_mpra.py for training (all logic preserved there), then:
  - Collects per-sample test predictions (scatter plots)
  - Generates summary figure (scatter + loss curves)
  - Saves metrics.json
  - Updates pearson's model club (best model tracker)

Usage:
  python ft_k562.py --name my_model
  python ft_k562.py --name my_model --learning_rate 5e-4 --num_epochs 30
  python ft_k562.py --name my_model --config path/to/other_config.json

Defaults from configs/mpra_K562.json. Override with passthrough flags:
  --learning_rate FLOAT     Learning rate (default: 1e-3)
  --num_epochs INT          Max epochs (default: 100)
  --batch_size INT          Batch size (default: 32)
  --early_stopping_patience INT  Patience (default: 5)
  --pooling_type STR        sum|mean|max|center|flatten (default: flatten)
  --center_bp INT           Center window bp (default: 256)
  --nl_size STR             Hidden sizes, e.g. "1024" or "512,256" (default: 1024)
  --do FLOAT                Dropout rate (default: 0.1)
  --activation STR          relu|gelu (default: relu)
  --optimizer STR           adam|adamw (default: adam)
  --weight_decay FLOAT      L2 reg (default: 1e-6)
  --gradient_clip FLOAT     Max gradient norm (default: None)
  --lr_scheduler STR        plateau|cosine (default: None)
  --second_stage_lr FLOAT   Stage 2 LR, enables two-stage (default: 1e-5)
  --second_stage_epochs INT Stage 2 epochs (default: 50)
  --no_wandb                Disable wandb logging
  --no_freeze_backbone      Train full model (not just head)
  --base_checkpoint_path STR  Custom AlphaGenome weights path
"""

import sys
import subprocess
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_DIR = Path(__file__).resolve().parents[3]
AGFT_DIR = PROJ_DIR / 'alphagenome_FT_MPRA'
RESULTS_BASE = PROJ_DIR / 'lenti_AGFT' / 'training' / 'results'
CLUB_DIR = PROJ_DIR / 'lenti_AGFT' / 'training' / "pearson's_model_club"
PROMOTER_CONSTRUCT_LENGTH = 281
# Local HuggingFace weights (avoids Kaggle prompt)
_WEIGHTS_SUBPATH = 'huggingface/hub/models--google--alphagenome-all-folds/snapshots/a8f293a76ee73d5b57f3bf2ae146510589fcf187'
DEFAULT_WEIGHTS = Path('/grid/wsbs/home_norepl/pmantill/Liver_AGFT/.weights') / _WEIGHTS_SUBPATH


# ---------------------------------------------------------------------------
# Post-training evaluation helpers
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_dir, config_path=None):
    """Load model with best checkpoint for evaluation."""
    sys.path.insert(0, str(AGFT_DIR))
    from alphagenome.models import dna_output
    from alphagenome_ft import HeadConfig, HeadType, register_custom_head, create_model_with_custom_heads
    from src import EncoderMPRAHead

    # Read hyperparams from the config or use defaults
    center_bp, pooling_type, nl_size, do, activation = 256, 'flatten', 1024, 0.1, 'relu'
    # Use local weights by default
    base_checkpoint_path = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else None
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        m = cfg.get('model', {})
        center_bp = m.get('center_bp', center_bp)
        pooling_type = m.get('pooling_type', pooling_type)
        nl_raw = m.get('nl_size', '1024')
        nl_size = [int(x) for x in nl_raw.split(',')] if ',' in str(nl_raw) else int(nl_raw)
        do = m.get('do', do)
        activation = m.get('activation', activation)
        if cfg.get('base_checkpoint_path'):
            base_checkpoint_path = cfg['base_checkpoint_path']

    register_custom_head('mpra_head', EncoderMPRAHead, HeadConfig(
        type=HeadType.GENOME_TRACKS, name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
        metadata={'center_bp': center_bp, 'pooling_type': pooling_type,
                  'nl_size': nl_size, 'do': do, 'activation': activation}
    ))
    model = create_model_with_custom_heads(
        'all_folds', custom_heads=['mpra_head'],
        checkpoint_path=base_checkpoint_path,
        use_encoder_output=True, init_seq_len=PROMOTER_CONSTRUCT_LENGTH
    )

    # Load best checkpoint
    best_ckpt = Path(checkpoint_dir) / 'best'
    if best_ckpt.exists():
        model.load_checkpoint(str(best_ckpt))
        print(f"Loaded best checkpoint from {best_ckpt}")
    else:
        print(f"WARNING: No checkpoint at {best_ckpt}, using initial weights")

    return model


def collect_predictions(model, dataloader, head_name='mpra_head'):
    """Collect per-sample predictions and targets."""
    from alphagenome_research.model import dna_model
    all_preds, all_targets = [], []
    for batch in dataloader:
        with model._device_context:
            predictions = model._predict(
                model._params, model._state,
                batch['seq'], batch['organism_index'],
                negative_strand_mask=jnp.zeros(len(batch['seq']), dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    model._device_context._device
                ),
            )
        all_preds.append(np.array(predictions[head_name]).squeeze())
        all_targets.append(np.array(batch['y']).squeeze())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def make_scatter(ax, targets, preds, title):
    """Single scatter panel with metrics."""
    r, _ = pearsonr(targets, preds)
    rho, _ = spearmanr(targets, preds)
    mse = np.mean((targets - preds) ** 2)
    ax.scatter(targets, preds, alpha=0.1, s=1, rasterized=True)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=0.8)
    ax.text(0.05, 0.95, f'r={r:.4f}\nrho={rho:.4f}\nMSE={mse:.4f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return r, rho, mse


def fetch_wandb_history(wandb_project, run_name):
    """Pull per-epoch loss curves from wandb. Returns dict or None."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(wandb_project, filters={"display_name": run_name})
        for run in runs:
            df = run.history(keys=['train_loss', 'val_loss', 'test_loss', 'epoch'],
                            pandas=True)
            if df.empty:
                continue
            # Drop sub-epoch evals, keep one row per epoch
            df = df.dropna(subset=['train_loss'])
            return {
                'train_loss': df['train_loss'].tolist(),
                'val_loss': df['val_loss'].dropna().tolist() if 'val_loss' in df else [],
                'test_loss': df['test_loss'].dropna().tolist() if 'test_loss' in df else [],
            }
    except Exception as e:
        print(f"Could not fetch wandb history: {e}")
    return None


def make_summary_figure(results_dir, test_targets, test_preds, history=None):
    """Scatter plot + optional loss curves from wandb."""
    n_panels = 2 if history and history.get('train_loss') else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    best_r, best_rho, best_mse = make_scatter(
        axes[0], test_targets, test_preds, 'Test Set — Best Checkpoint')

    if n_panels == 2:
        ax = axes[1]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Train')
        if history.get('val_loss'):
            ax.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val')
        if history.get('test_loss'):
            ax.plot(range(1, len(history['test_loss']) + 1), history['test_loss'], label='Test')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Curves')
        ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / 'summary.png', dpi=150)
    plt.close(fig)
    return best_r, best_rho, best_mse


def update_model_club(model_name, test_pearson, results_dir, config_path=None,
                      spearman_rho=None, mse=None):
    """Update pearson's model club CSV; copy summary if new best."""
    CLUB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CLUB_DIR / 'best_models.csv'

    # Build short hyperparams description
    hp_str = ''
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        m = cfg.get('model', {})
        t = cfg.get('training', {})
        ts = cfg.get('two_stage', {})
        parts = [
            f"pool={m.get('pooling_type','?')}",
            f"nl={m.get('nl_size','?')}",
            f"do={m.get('do','?')}",
            f"act={m.get('activation','?')}",
            f"lr={t.get('learning_rate','?')}",
            f"opt={t.get('optimizer','?')}",
            f"wd={t.get('weight_decay','?')}",
        ]
        if ts.get('enabled'):
            parts.append(f"s2_lr={ts.get('second_stage_lr','?')}")
        hp_str = ' | '.join(parts)

    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))

    current_best = max((float(r['pearson_r']) for r in rows), default=-999)

    rows.append({
        'name': model_name,
        'pearson_r': f'{test_pearson:.6f}',
        'spearman_rho': f'{spearman_rho:.6f}' if spearman_rho is not None else '',
        'mse': f'{mse:.6f}' if mse is not None else '',
        'cell_type': 'K562',
        'hyperparams': hp_str,
        'timestamp': datetime.now().isoformat(),
    })
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'pearson_r', 'spearman_rho', 'mse', 'cell_type', 'hyperparams', 'timestamp'])
        w.writeheader()
        w.writerows(rows)

    if test_pearson > current_best:
        import shutil
        src = results_dir / 'summary.png'
        if src.exists():
            shutil.copy2(src, CLUB_DIR / 'best_model_summary.png')
        print(f"NEW BEST MODEL: {model_name} (r={test_pearson:.4f} > {current_best:.4f})")
    else:
        print(f"Model {model_name} (r={test_pearson:.4f}) did not beat best (r={current_best:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse --name and --config; pass everything else through to finetune_mpra.py
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--name', type=str, required=True, help='Model name for results dir')
    pre_parser.add_argument('--config', type=str,
                            default=str(AGFT_DIR / 'configs' / 'mpra_K562.json'))
    pre_args, passthrough = pre_parser.parse_known_args()

    model_name = pre_args.name
    config_path = str(Path(pre_args.config).resolve()) if pre_args.config else None
    results_dir = RESULTS_BASE / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / 'checkpoints'

    # ---- Step 1: Run finetune_mpra.py ----
    finetune_script = AGFT_DIR / 'scripts' / 'finetune_mpra.py'
    # Use local weights if available (avoids Kaggle prompt)
    weights_path = DEFAULT_WEIGHTS

    cmd = [
        sys.executable, str(finetune_script),
        '--cell_type', 'K562',
        '--checkpoint_dir', str(checkpoint_dir),
        '--wandb_name', model_name,
        '--save_test_results', str(results_dir / 'test_results.csv'),
    ]
    if weights_path.exists() and '--base_checkpoint_path' not in passthrough:
        cmd += ['--base_checkpoint_path', str(weights_path)]
    # Auto-resume stage2 if checkpoint exists from a previous run
    stage2_ckpt = checkpoint_dir / 'K562' / model_name / 'stage2'
    if stage2_ckpt.exists() and '--resume_from_stage2' not in passthrough:
        print(f"Found existing stage2 checkpoint, resuming...")
        cmd += ['--resume_from_stage2']
    if config_path:
        cmd += ['--config', config_path]
    cmd += passthrough

    print("=" * 60)
    print(f"ft_k562: {model_name}")
    print(f"Results: {results_dir}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    env = {**__import__('os').environ, 'PYTHONUNBUFFERED': '1'}
    env['PYTHONPATH'] = str(AGFT_DIR) + ':' + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, cwd=str(PROJ_DIR), env=env)
    if result.returncode != 0:
        print(f"ERROR: finetune_mpra.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # ---- Step 2: Post-training evaluation ----
    print("\n" + "=" * 60)
    print("Post-training evaluation")
    print("=" * 60)

    # Load model with best checkpoint
    model = load_model_from_checkpoint(str(checkpoint_dir), config_path)

    # Create test dataloader
    sys.path.insert(0, str(AGFT_DIR))
    from src import LentiMPRADataset, MPRADataLoader

    # Read batch_size from config or default
    batch_size = 32
    if config_path:
        with open(config_path) as f:
            batch_size = json.load(f).get('data', {}).get('batch_size', 32)

    test_dataset = LentiMPRADataset(
        model=model, cell_type='K562', split='test',
        random_shift=False, reverse_complement=False
    )
    test_loader = MPRADataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Collect per-sample predictions
    print(f"Collecting predictions on {len(test_dataset)} test samples...")
    test_targets, test_preds = collect_predictions(model, test_loader)

    test_r, _ = pearsonr(test_targets, test_preds)
    test_rho, _ = spearmanr(test_targets, test_preds)
    test_mse = float(np.mean((test_targets - test_preds) ** 2))
    print(f"Test Pearson:  {test_r:.4f}")
    print(f"Test Spearman: {test_rho:.4f}")
    print(f"Test MSE:      {test_mse:.4f}")

    # Save predictions
    np.savez(results_dir / 'test_predictions.npz',
             targets=test_targets, predictions=test_preds)

    # Fetch loss curves from wandb
    wandb_project = 'alphagenome-mpra'
    if config_path:
        with open(config_path) as f:
            wandb_project = json.load(f).get('wandb', {}).get('project', wandb_project)
    print("Fetching training curves from wandb...")
    history = fetch_wandb_history(wandb_project, model_name)

    # Generate summary figure
    print("Generating summary figure...")
    make_summary_figure(results_dir, test_targets, test_preds, history=history)
    print(f"Saved: {results_dir / 'summary.png'}")

    # Read hyperparams from config for metrics.json
    hyperparams = {}
    if config_path:
        with open(config_path) as f:
            hyperparams = json.load(f)

    metrics = {
        'model_name': model_name,
        'cell_type': 'K562',
        'timestamp': datetime.now().isoformat(),
        'config': hyperparams,
        'test_metrics': {'pearson': test_r, 'spearman': test_rho, 'mse': test_mse},
        'num_test_samples': len(test_targets),
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {results_dir / 'metrics.json'}")

    # Update pearson's model club
    update_model_club(model_name, test_r, results_dir, config_path,
                      spearman_rho=test_rho, mse=test_mse)

    print("\n" + "=" * 60)
    print(f"Done! Results in: {results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
