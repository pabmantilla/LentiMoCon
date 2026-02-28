"""Eval-only: load checkpoint, make scatter plot + metrics.json + model club entry.

Works with checkpoints from both train_k562.py (saves to 'best/') and
ft_k562.py/finetune_mpra.py (saves to 'stage1/', 'stage2/').
"""
import sys, json, csv, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_DIR = Path(__file__).resolve().parents[3]
AGFT_DIR = PROJ_DIR / 'alphagenome_FT_MPRA'
sys.path.insert(0, str(AGFT_DIR))

RESULTS_BASE = PROJ_DIR / 'lenti_AGFT' / 'training' / 'results'
CLUB_DIR = PROJ_DIR / 'lenti_AGFT' / 'training' / "pearson's_model_club"
_WEIGHTS_SUBPATH = 'huggingface/hub/models--google--alphagenome-all-folds/snapshots/a8f293a76ee73d5b57f3bf2ae146510589fcf187'
DEFAULT_WEIGHTS = Path('/grid/wsbs/home_norepl/pmantill/Liver_AGFT/.weights') / _WEIGHTS_SUBPATH

from ft_k562 import make_scatter, update_model_club


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoint', required=True,
                   help='Path to checkpoint dir containing config.json + checkpoint/')
    p.add_argument('--config', default=str(AGFT_DIR / 'configs' / 'mpra_K562.json'),
                   help='Config for model club entry (not used for architecture)')
    args = p.parse_args()

    results_dir = RESULTS_BASE / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint)

    # Fix library bug: save_checkpoint always writes use_encoder_output=false.
    # Patch it before loading so load_checkpoint creates the model correctly.
    ckpt_config_path = ckpt_dir / 'config.json'
    if ckpt_config_path.exists():
        with open(ckpt_config_path) as f:
            ckpt_cfg = json.load(f)
        if not ckpt_cfg.get('use_encoder_output', False):
            ckpt_cfg['use_encoder_output'] = True
            with open(ckpt_config_path, 'w') as f:
                json.dump(ckpt_cfg, f, indent=2)
            print(f"Patched use_encoder_output=true in {ckpt_config_path}")

    # Load using the library's load_checkpoint (handles orbax checkpoints)
    from alphagenome_ft import load_checkpoint
    from alphagenome_research.model import dna_model
    from src import EncoderMPRAHead, LentiMPRADataset, MPRADataLoader

    # EncoderMPRAHead must be registered before load_checkpoint
    from alphagenome.models import dna_output
    from alphagenome_ft import HeadConfig, HeadType, register_custom_head

    # Read head metadata from checkpoint config to register correctly
    with open(ckpt_config_path) as f:
        ckpt_cfg = json.load(f)
    head_meta = ckpt_cfg.get('head_configs', {}).get('mpra_head', {}).get('metadata', {})
    register_custom_head('mpra_head', EncoderMPRAHead, HeadConfig(
        type=HeadType.GENOME_TRACKS, name='mpra_head',
        output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
        metadata=head_meta,
    ))

    model = load_checkpoint(
        str(ckpt_dir),
        base_checkpoint_path=str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else None,
        init_seq_len=281,
    )
    print(f"Loaded checkpoint: {ckpt_dir}")

    # Test set predictions
    data_dir = str(PROJ_DIR / 'test_run_lenti_data')
    test_ds = LentiMPRADataset(
        model=model, cell_type='K562', split='test',
        path_to_data=data_dir,
        random_shift=False, reverse_complement=False,
    )
    test_loader = MPRADataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"Test set: {len(test_ds)} samples")

    organism_enum = dna_model.Organism.HOMO_SAPIENS
    all_preds, all_targets = [], []
    for batch in test_loader:
        with model._device_context:
            preds = model._predict(
                model._params, model._state,
                batch['seq'], batch['organism_index'],
                negative_strand_mask=jnp.zeros(len(batch['seq']), dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[organism_enum].strand_reindexing,
                    model._device_context._device),
            )
        # Cast to float32 to avoid bfloat16 issues with scipy
        pred_arr = np.array(preds['mpra_head'], dtype=np.float32)
        # predict() returns per-position output (batch, positions, tracks).
        # Pool over positions to get scalar per sample, matching loss function.
        if pred_arr.ndim == 3:
            pred_arr = pred_arr.sum(axis=1)  # (batch, tracks)
        all_preds.append(pred_arr.squeeze())
        all_targets.append(np.array(batch['y'], dtype=np.float32).squeeze())

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_preds)

    r, _ = pearsonr(targets, predictions)
    rho, _ = spearmanr(targets, predictions)
    mse = float(np.mean((targets - predictions) ** 2))
    print(f"Pearson: {r:.4f}  Spearman: {rho:.4f}  MSE: {mse:.4f}")

    # Save predictions
    np.savez(results_dir / 'test_predictions.npz',
             targets=targets, predictions=predictions)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    make_scatter(ax, targets, predictions, f'{args.name} — Test Set')
    plt.tight_layout()
    fig.savefig(results_dir / 'summary.png', dpi=150)
    plt.close(fig)
    print(f"Saved: {results_dir / 'summary.png'}")

    # metrics.json
    with open(args.config) as f:
        cfg = json.load(f)
    metrics = {
        'model_name': args.name, 'cell_type': 'K562',
        'timestamp': datetime.now().isoformat(),
        'config': cfg,
        'checkpoint_config': ckpt_cfg,
        'test_metrics': {'pearson': float(r), 'spearman': float(rho), 'mse': float(mse)},
        'num_test_samples': len(targets),
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {results_dir / 'metrics.json'}")

    # Model club
    update_model_club(args.name, r, results_dir, args.config,
                      spearman_rho=rho, mse=mse)


if __name__ == '__main__':
    main()
