"""SEAM clustering + MetaExplainer background separation.

For each sequence, loads pre-computed predictions + attributions,
runs SEAM K-means clustering (30 clusters) and MetaExplainer to
extract scaled foreground, scaled background, and avg background.

Usage:
    python run_seam_clustering.py [--start START] [--end END]

Requires: SEAM_revisions/.venv (seam + squid)
Prereqs: compute_legnet_attrs.py must have been run first.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

from seam import Compiler, Clusterer, MetaExplainer

# Config
N_CLUSTERS = 30
MUT_RATE = 0.10
ALPHABET = ['A', 'C', 'G', 'T']

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVITY_LIB = REPO_ROOT / "large_scale_swap/data/activity_lib/k562_activity_library.csv"
MUT_LIB_DIR = REPO_ROOT / "large_scale_swap/data/mutagenesis_lib"
ATTR_DIR = REPO_ROOT / "large_scale_swap/data/attributions_testfold"
OUT_DIR = REPO_ROOT / "large_scale_swap/data/foregrounds"


def process_sequence(seq_id, activity_bin):
    prefix = f"{activity_bin}_{seq_id}"
    mut_path = MUT_LIB_DIR / f"{prefix}.h5"
    attr_path = ATTR_DIR / f"{prefix}.h5"
    seq_dir = OUT_DIR / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Load mutagenesis library: WT + 24999 mutants = 25000 total
    with h5py.File(mut_path, 'r') as f:
        x_mut_noWT = f['sequences'][:24999]
        wt_seq = f['wt_sequence'][:]
    x_mut = np.concatenate([wt_seq[np.newaxis], x_mut_noWT], axis=0)  # (25000, 230, 4)

    # Load predictions + attributions (WT at index 0, 25000 total)
    with h5py.File(attr_path, 'r') as f:
        predictions = f['predictions'][:]     # (25000,)
        attributions = f['attributions'][:]   # (25000, 230, 4)

    # K-means clustering on flattened attribution maps
    clusterer = Clusterer(attributions, gpu=False)
    cluster_labels = clusterer.cluster(
        embedding=clusterer.maps,
        method='kmeans',
        n_clusters=N_CLUSTERS
    )

    # Compile MAVE dataframe
    compiler = Compiler(
        x=x_mut,
        y=predictions,
        x_ref=wt_seq[np.newaxis],
        y_bg=None,
        alphabet=ALPHABET,
        gpu=False
    )
    mave_df = compiler.compile()

    # Fresh clusterer with labels for MetaExplainer
    clusterer = Clusterer(attributions, gpu=False)
    clusterer.cluster_labels = cluster_labels

    # MetaExplainer: sort clusters, MSM, background separation
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method='median',
        ref_idx=0,
        mut_rate=MUT_RATE
    )
    msm = meta.generate_msm(gpu=False)
    meta.compute_background(
        mut_rate=MUT_RATE,
        entropy_multiplier=0.5,
        adaptive_background_scaling=True,
        process_logos=False
    )

    # Get WT reference cluster and compute foreground
    if meta.cluster_order is not None:
        mapping = {old: new for new, old in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping)
        ref_cluster = meta.membership_df.loc[0, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[0, 'Cluster']

    ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0)
    bg_scale = meta.background_scaling[ref_cluster] if meta.background_scaling is not None else 1.0
    foreground_scaled = ref_cluster_avg - bg_scale * meta.background

    # Save outputs
    np.save(seq_dir / 'foreground_scaled.npy', foreground_scaled)
    np.save(seq_dir / 'average_background.npy', meta.background)
    np.save(seq_dir / 'average_background_scaled.npy', bg_scale * meta.background)
    np.save(seq_dir / 'wt_attribution.npy', attributions[0])
    np.save(seq_dir / 'ref_cluster_avg.npy', ref_cluster_avg)
    np.save(seq_dir / 'cluster_labels.npy', cluster_labels)

    print(f"    bg_scale={bg_scale:.4f}, ref_cluster={ref_cluster}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lib_df = pd.read_csv(ACTIVITY_LIB)
    end = args.end if args.end is not None else len(lib_df)
    lib_df = lib_df.iloc[args.start:end]
    print(f"Processing {len(lib_df)} sequences [{args.start}:{end}]")

    for i, (_, row) in enumerate(lib_df.iterrows()):
        seq_id = row['seq_id']
        activity_bin = row['activity_bin']
        seq_dir = OUT_DIR / seq_id

        if (seq_dir / 'foreground_scaled.npy').exists():
            continue

        attr_path = ATTR_DIR / f"{activity_bin}_{seq_id}.h5"
        if not attr_path.exists():
            print(f"  [{i+1}/{len(lib_df)}] {seq_id} — attributions not found, skipping")
            continue

        print(f"  [{i+1}/{len(lib_df)}] {activity_bin}/{seq_id}")
        try:
            process_sequence(seq_id, activity_bin)
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
