"""Compute LegNet predictions + DeepLIFT/SHAP attributions.

For each sequence in the activity library, loads its mutagenesis library (25K mutants),
prepends WT as index 0, and computes:
  - Predictions (single test-fold model or 10-fold ensemble)
  - DeepLIFT/SHAP attributions (tangermeme, hypothetical=True for SE layers)

Default: uses only the test-fold model for each sequence (10x faster).
With --ensemble: uses all 10 models and averages (checkpointed per model).

Usage:
    python compute_legnet_attrs.py [--start START] [--end END] [--n-shuffles 20]
    python compute_legnet_attrs.py --ensemble [--start START] [--end END]

Requires: legnet_env (torch + tangermeme + lightning)
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import h5py
import glob
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'human_legnet'))

from trainer import LitModel, TrainingConfig
from tangermeme.deep_lift_shap import deep_lift_shap

# Paths
ACTIVITY_LIB = REPO_ROOT / "large_scale_swap/data/activity_lib/k562_activity_library.csv"
MUT_LIB_DIR = REPO_ROOT / "large_scale_swap/data/mutagenesis_lib"
OUT_DIR_ENSEMBLE = REPO_ROOT / "large_scale_swap/data/attributions"
OUT_DIR_TESTFOLD = REPO_ROOT / "large_scale_swap/data/attributions_testfold"
K562_MODEL_DIR = REPO_ROOT / ".weights/legnet_pretrained/final_dump/models/K562/md_shift_reverse_noavg_noch"
CONFIG_PATH = K562_MODEL_DIR / "config.json"

# Mutagenesis libs are ACGT (A=0,C=1,G=2,T=3), LegNet is AGCT (A=0,G=1,C=2,T=3)
CH_SWAP = [0, 2, 1, 3]


class WrappedModel(torch.nn.Module):
    """Wrap LegNet to return (batch, 1) for tangermeme target indexing."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).unsqueeze(-1)


def load_models():
    """Load all 10 models, indexed by test fold (1-indexed: models_by_fold[fold])."""
    train_cfg = TrainingConfig.from_json(str(CONFIG_PATH))
    models_by_fold = {}
    for test_fold in range(1, 11):
        pattern = str(K562_MODEL_DIR / f'best_model_test{test_fold}_val*.ckpt')
        cp = sorted(glob.glob(pattern))[0]
        m = LitModel.load_from_checkpoint(cp, tr_cfg=train_cfg)
        m.eval()
        m.model.cuda()
        models_by_fold[test_fold] = m.model
    print(f"Loaded {len(models_by_fold)} LegNet models")
    return models_by_fold


def compute_predictions(model_list, x_agct, batch_size=512):
    """Predictions averaged over model_list. x_agct: (N, 4, 230) numpy array."""
    ensemble = []
    with torch.no_grad():
        for m in model_list:
            preds = []
            for i in range(0, len(x_agct), batch_size):
                batch = torch.from_numpy(x_agct[i:i+batch_size]).float().cuda()
                preds.append(m(batch).cpu().numpy().flatten())
            ensemble.append(np.concatenate(preds))
    return np.mean(ensemble, axis=0).astype(np.float32)


def process_sequence(models_by_fold, fold, seq_id, activity_bin,
                     mut_lib_path, out_path, n_shuffles, use_ensemble):
    with h5py.File(mut_lib_path, 'r') as f:
        x_mut = f['sequences'][:24999]
        wt_seq = f['wt_sequence'][:]

    all_acgt = np.concatenate([wt_seq[np.newaxis], x_mut], axis=0)  # (25000, 230, 4)
    all_agct = all_acgt.transpose(0, 2, 1)[:, CH_SWAP, :]  # (25000, 4, 230)

    # Check existing progress
    has_preds = has_attrs = False
    models_done = 0
    if out_path.exists():
        with h5py.File(out_path, 'r') as f:
            has_preds = 'predictions' in f
            has_attrs = 'attributions' in f
            models_done = int(f.attrs.get('models_done', 0))

    if has_preds and has_attrs:
        return False

    # Select models
    if use_ensemble:
        model_list = [models_by_fold[f] for f in sorted(models_by_fold)]
    else:
        model_list = [models_by_fold[fold]]

    n_models = len(model_list)

    # Predictions
    if not has_preds:
        print(f"    predictions ({n_models} model{'s' if n_models > 1 else ''})...")
        predictions = compute_predictions(model_list, all_agct)
        with h5py.File(out_path, 'a') as f:
            if 'predictions' in f:
                del f['predictions']
            f.create_dataset('predictions', data=predictions)
        print(f"    -> {predictions.shape}, wt_pred={predictions[0]:.4f}")

    # Attributions with per-model checkpointing
    if not has_attrs:
        X = torch.from_numpy(all_agct).float()

        # Load checkpoint if partial progress exists
        if models_done > 0 and out_path.exists():
            with h5py.File(out_path, 'r') as f:
                if 'attrs_partial' in f:
                    partial_sum = f['attrs_partial'][:]
                    print(f"    resuming attributions from model {models_done+1}/{n_models}...")
                else:
                    models_done = 0
                    partial_sum = np.zeros((len(all_agct), 4, 230), dtype=np.float64)
        else:
            partial_sum = np.zeros((len(all_agct), 4, 230), dtype=np.float64)

        t0 = time.time()
        print(f"    attributions ({n_shuffles} shuffles, {n_models} model{'s' if n_models > 1 else ''})...")

        for mi in range(models_done, n_models):
            wrapped = WrappedModel(model_list[mi]).eval()
            attr = deep_lift_shap(wrapped, X, n_shuffles=n_shuffles, target=0,
                                  hypothetical=True, batch_size=128,
                                  device='cuda', random_state=42, verbose=False)
            partial_sum += attr.cpu().numpy()
            elapsed = time.time() - t0
            print(f"      model {mi+1}/{n_models} done ({elapsed/60:.1f}min)")

            # Checkpoint after each model (only needed for ensemble)
            if n_models > 1:
                with h5py.File(out_path, 'a') as f:
                    if 'attrs_partial' in f:
                        del f['attrs_partial']
                    f.create_dataset('attrs_partial', data=partial_sum)
                    f.attrs['models_done'] = mi + 1

        # Finalize: average over models, convert to ACGT, save
        avg_attrs = (partial_sum / n_models).astype(np.float32)
        attrs_acgt = avg_attrs[:, CH_SWAP, :].transpose(0, 2, 1)  # (N, 230, 4) ACGT

        with h5py.File(out_path, 'a') as f:
            if 'attributions' in f:
                del f['attributions']
            f.create_dataset('attributions', data=attrs_acgt,
                             compression='gzip', compression_opts=4)
            # Clean up checkpoint data
            if 'attrs_partial' in f:
                del f['attrs_partial']
            if 'models_done' in f.attrs:
                del f.attrs['models_done']

        print(f"    -> {attrs_acgt.shape} in {(time.time()-t0)/60:.1f}min")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--n-shuffles', type=int, default=20)
    parser.add_argument('--ensemble', action='store_true',
                        help='Use all 10 models (default: test-fold model only)')
    args = parser.parse_args()

    OUT_DIR = OUT_DIR_ENSEMBLE if args.ensemble else OUT_DIR_TESTFOLD
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lib_df = pd.read_csv(ACTIVITY_LIB)
    end = args.end if args.end is not None else len(lib_df)
    lib_df = lib_df.iloc[args.start:end]

    mode = "10-fold ensemble" if args.ensemble else "test-fold only"
    print(f"Processing {len(lib_df)} sequences [{args.start}:{end}] ({mode})")

    models_by_fold = load_models()

    for i, (_, row) in enumerate(lib_df.iterrows()):
        seq_id = row['seq_id']
        activity_bin = row['activity_bin']
        fold = int(row['fold'])
        mut_path = MUT_LIB_DIR / f"{activity_bin}_{seq_id}.h5"
        out_path = OUT_DIR / f"{activity_bin}_{seq_id}.h5"

        print(f"  [{i+1}/{len(lib_df)}] {activity_bin}/{seq_id} (fold={fold})")
        if not process_sequence(models_by_fold, fold, seq_id, activity_bin,
                                mut_path, out_path, args.n_shuffles, args.ensemble):
            print(f"    complete, skipping")

    print("\nDone!")


if __name__ == '__main__':
    main()
