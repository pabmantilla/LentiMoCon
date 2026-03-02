"""SEAM clustering + background separation per WT sequence.

For each of the 3000 WT sequences, loads its mutagenesis library attributions
(WT at idx 0, 25K mutants at idx 1..N), clusters flattened maps with KMeans,
then uses MetaExplainer to compute foreground/background separation.

Requires the SEAM venv: SEAM_revisions/.venv

Usage:
    python seam_clustering.py --model second_2stageig
    python seam_clustering.py --model second_2stageig --n_clusters 20

SLURM array support:
    #SBATCH --array=0-29
    python seam_clustering.py --model second_2stageig
"""

import os, sys, argparse, time
from pathlib import Path
import numpy as np
import h5py
import pandas as pd

# ---------- paths ----------
PROJ_DIR = Path(__file__).resolve().parents[4]  # LentiMoCon root
ACTIVITY_LIB_BASE = PROJ_DIR / 'lenti_AGFT' / 'motif_context_swap' / 'library_prep' / 'activity_libraries'
ATTR_LIB_BASE = PROJ_DIR / 'lenti_AGFT' / 'motif_context_swap' / 'library_prep' / 'attribution_libraries'
RESULTS_BASE = PROJ_DIR / 'lenti_AGFT' / 'motif_context_swap' / 'library_prep' / 'seam_results'

ALPHA_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
SEQ_LENGTH = 230
MUT_RATE = 0.10


def onehot_to_str(ohe):
    """Convert (L, 4) one-hot to DNA string."""
    return ''.join(ALPHA_MAP[int(np.argmax(ohe[i]))] for i in range(ohe.shape[0]))


def load_mutagenesis_and_attributions(mut_lib_path, attr_path):
    """Load mutagenesis library sequences and their attributions.

    Returns:
        attributions: (N+1, L, 4) — WT at idx 0, mutants at idx 1..N
        sequences_ohe: (N+1, L, 4) — matching one-hot sequences
        wt_ohe: (L, 4) — WT one-hot
        predictions: (N+1,) — DNN predictions if available, else None
    """
    # Load attributions (WT at idx 0, mutants at idx 1..N)
    with h5py.File(attr_path, 'r') as f:
        attributions = f['deepshap'][:]  # (N+1, L, 4)
        wt_ohe = f['wt_onehot'][:]      # (L, 4)

    # Load mutagenesis library sequences
    with h5py.File(mut_lib_path, 'r') as f:
        mutants = f['sequences'][:]      # (N, L, 4)
        wt_check = f['wt_sequence'][:]   # (L, 4)
        predictions = f['predictions'][:] if 'predictions' in f else None

    # Build sequence array: WT at idx 0, mutants at idx 1..N
    sequences_ohe = np.concatenate([wt_ohe[None, :, :], mutants], axis=0)

    # If predictions exist, prepend WT prediction (not in mutant predictions)
    # WT prediction needs to be computed or set to NaN
    if predictions is not None:
        # predictions are for mutants only; we don't have WT prediction stored
        # Use mean of mutant predictions as placeholder for WT
        wt_pred = np.array([np.mean(predictions)])
        predictions = np.concatenate([wt_pred, predictions.flatten()])

    return attributions, sequences_ohe, wt_ohe, predictions


def build_mave_df(sequences_ohe, predictions):
    """Build the DataFrame that MetaExplainer expects.

    Must have 'Sequence' and 'DNN' columns. WT is at idx 0.
    """
    seqs = [onehot_to_str(s) for s in sequences_ohe]
    if predictions is None:
        predictions = np.zeros(len(sequences_ohe))
    return pd.DataFrame({'DNN': predictions, 'Sequence': seqs})


def process_single_wt(label, mut_lib_path, attr_path, out_dir,
                      n_clusters, entropy_multiplier, gpu):
    """Run full SEAM pipeline for one WT sequence."""
    from seam import Clusterer, MetaExplainer

    print(f"\n  Loading data...")
    attributions, sequences_ohe, wt_ohe, predictions = \
        load_mutagenesis_and_attributions(mut_lib_path, attr_path)
    n_total = len(attributions)
    print(f"  Loaded {n_total} attribution maps (1 WT + {n_total-1} mutants)")

    # ---------- Cluster ----------
    print(f"  Clustering into {n_clusters} clusters (KMeans, flattened, gpu={gpu})...")
    t0 = time.time()

    clusterer = Clusterer(attributions, gpu=gpu)
    # Pass flattened maps directly — no dimensionality reduction
    cluster_labels = clusterer.cluster(
        embedding=clusterer.maps,
        method='kmeans',
        n_clusters=n_clusters,
    )
    print(f"  Clustering done in {time.time()-t0:.1f}s")

    # Save cluster labels
    np.save(out_dir / f'cluster_labels_k{n_clusters}.npy', cluster_labels)

    # ---------- Compile MAVE dataframe ----------
    print("  Building MAVE dataframe...")
    mave_df = build_mave_df(sequences_ohe, predictions)

    # Fresh clusterer for MetaExplainer (no GPU needed)
    clusterer = Clusterer(attributions, gpu=False)
    clusterer.cluster_labels = cluster_labels

    # ---------- MetaExplainer ----------
    print("  Initializing MetaExplainer (ref_idx=0 = WT)...")
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method='median',
        ref_idx=0,  # WT is always at index 0
        mut_rate=MUT_RATE,
    )

    print("  Generating MSM...")
    msm = meta.generate_msm(gpu=False)

    print("  Computing background separation...")
    meta.compute_background(
        mut_rate=MUT_RATE,
        entropy_multiplier=entropy_multiplier,
        adaptive_background_scaling=True,
        process_logos=False,
    )

    # ---------- Save outputs ----------
    # Global background
    np.save(out_dir / 'global_background.npy', meta.background)
    msm.to_csv(out_dir / 'msm.csv', index=False)

    # Sorted cluster order (same order compute_background uses internally)
    cluster_order = meta.cluster_order if meta.cluster_order is not None else list(range(n_clusters))
    sorted_mapping = {orig: sorted_idx for sorted_idx, orig in enumerate(cluster_order)}

    # Determine WT's sorted cluster
    wt_cluster_orig = cluster_labels[0]
    wt_cluster_sorted = sorted_mapping[wt_cluster_orig]

    # WT cluster foreground (like background_compute.py)
    ref_cluster_avg = np.mean(meta.get_cluster_maps(wt_cluster_sorted), axis=0)
    bg_scale = meta.background_scaling[wt_cluster_sorted] if meta.background_scaling is not None else 1.0
    foreground = ref_cluster_avg - meta.background
    foreground_scaled = ref_cluster_avg - bg_scale * meta.background

    np.save(out_dir / 'ref_cluster_avg.npy', ref_cluster_avg)
    np.save(out_dir / 'foreground.npy', foreground)
    np.save(out_dir / 'foreground_scaled.npy', foreground_scaled)
    np.save(out_dir / 'wt_attribution.npy', attributions[0])

    # Per-cluster foregrounds
    cluster_dir = out_dir / 'clusters'
    cluster_dir.mkdir(exist_ok=True)

    for sorted_idx in range(n_clusters):
        c_maps = meta.get_cluster_maps(sorted_idx)
        c_avg = np.mean(c_maps, axis=0)
        c_bg_scale = meta.background_scaling[sorted_idx] if meta.background_scaling is not None else 1.0
        c_foreground = c_avg - c_bg_scale * meta.background

        np.savez_compressed(
            cluster_dir / f'cluster_{sorted_idx}.npz',
            cluster_avg=c_avg,
            foreground=c_foreground,
            background_scaling=c_bg_scale,
            n_members=len(c_maps),
            original_label=int(cluster_order[sorted_idx]),
        )

    print(f"  WT cluster (sorted): {wt_cluster_sorted} (orig: {wt_cluster_orig}), "
          f"bg_scale={bg_scale:.4f}")
    print(f"  Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='SEAM clustering per WT sequence')
    parser.add_argument('--model', required=True, help='Model name (e.g. second_2stageig)')
    parser.add_argument('--n_clusters', type=int, default=30,
                        help='Number of KMeans clusters (default: 30)')
    parser.add_argument('--entropy_multiplier', type=float, default=0.5,
                        help='Background entropy multiplier (default: 0.5)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for clustering')
    args = parser.parse_args()

    # Load activity library CSV
    lib_path = ACTIVITY_LIB_BASE / args.model / f"k562_activity_library_{args.model}.csv"
    lib_df = pd.read_csv(lib_path)
    print(f"Activity library: {len(lib_df)} WT sequences from {lib_path.name}")

    # Paths
    mut_lib_dir = ACTIVITY_LIB_BASE / args.model / 'mutagenesis_libraries'
    attr_dir = ATTR_LIB_BASE / args.model

    # SLURM array: partition WT sequences across tasks
    array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    indices = np.array_split(range(len(lib_df)), array_count)[array_id]
    if array_count > 1:
        print(f"SLURM array {array_id}/{array_count}: processing {len(indices)} WT sequences")

    t_start = time.time()

    for count, idx in enumerate(indices):
        row = lib_df.iloc[idx]
        label = f"{row['activity_bin']}_{row['seq_id']}"

        mut_lib_path = mut_lib_dir / f"{label}.h5"
        attr_path = attr_dir / f"{label}.h5"

        # Check prerequisites
        if not mut_lib_path.exists():
            print(f"[{count+1}/{len(indices)}] SKIP {label}: no mutagenesis library")
            continue
        if not attr_path.exists():
            print(f"[{count+1}/{len(indices)}] SKIP {label}: no attributions")
            continue
        with h5py.File(attr_path, 'r') as f:
            if 'deepshap' not in f or f['deepshap'].shape[0] <= 1:
                print(f"[{count+1}/{len(indices)}] SKIP {label}: attributions incomplete "
                      f"(need WT + mutants)")
                continue

        # Output directory per WT
        out_dir = RESULTS_BASE / args.model / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already complete
        if (out_dir / 'foreground_scaled.npy').exists():
            print(f"[{count+1}/{len(indices)}] SKIP {label}: already done")
            continue

        print(f"\n{'='*60}")
        print(f"[{count+1}/{len(indices)}] {label} (activity={row['actual_activity']:.3f})")
        print(f"{'='*60}")

        process_single_wt(
            label, mut_lib_path, attr_path, out_dir,
            n_clusters=args.n_clusters,
            entropy_multiplier=args.entropy_multiplier,
            gpu=args.gpu,
        )

    elapsed = time.time() - t_start
    print(f"\nAll done! {len(indices)} WT sequences in {elapsed/60:.1f} minutes")
    print(f"Results: {RESULTS_BASE / args.model}")


if __name__ == '__main__':
    main()
