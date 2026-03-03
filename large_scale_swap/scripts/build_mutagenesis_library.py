"""Build mutagenesis libraries for SEAM analysis.

Generates 25K random mutants at 10% mutation rate for each of the 3000
sequences in the activity library. Saves as gzipped HDF5 files.

Usage:
    python build_mutagenesis_library.py [--start START] [--end END]

Requires SEAM_revisions/.venv (has squid + h5py).
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import squid

# Config
LIB_SIZE = 25000
MUT_RATE = 0.10
SEQ_LENGTH = 230
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVITY_LIB = REPO_ROOT / "large_scale_swap/data/activity_lib/k562_activity_library.csv"
OUT_DIR = REPO_ROOT / "large_scale_swap/data/mutagenesis_lib"

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def str_to_onehot(seq_str):
    ohe = np.zeros((len(seq_str), 4), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[j, ALPHA_MAP[base]] = 1.0
    return ohe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lib_df = pd.read_csv(ACTIVITY_LIB)
    end = args.end if args.end is not None else len(lib_df)
    lib_df = lib_df.iloc[args.start:end]
    print(f"Processing {len(lib_df)} sequences [{args.start}:{end}]")

    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=MUT_RATE, seed=SEED)

    for i, (_, row) in enumerate(lib_df.iterrows()):
        seq_id = row["seq_id"]
        activity_bin = row["activity_bin"]
        out_file = OUT_DIR / f"{activity_bin}_{seq_id}.h5"

        if out_file.exists():
            continue

        wt_onehot = str_to_onehot(row["sequence"])
        x_mut = mut_generator(wt_onehot, num_sim=LIB_SIZE)

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('sequences', data=x_mut, dtype='float32',
                             compression='gzip', compression_opts=4)
            f.create_dataset('wt_sequence', data=wt_onehot, dtype='float32')
            f.attrs['seq_id'] = seq_id
            f.attrs['activity_bin'] = activity_bin
            f.attrs['actual_activity'] = row['actual_activity']
            f.attrs['split'] = row['split']
            f.attrs['fold'] = int(row['fold'])
            f.attrs['n_mutants'] = LIB_SIZE
            f.attrs['mut_rate'] = MUT_RATE
            f.attrs['seq_length'] = SEQ_LENGTH
            f.attrs['alphabet'] = 'ACGT'

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(lib_df)}] {activity_bin}/{seq_id} saved")

    total = len(list(OUT_DIR.glob("*.h5")))
    print(f"\nDone! {total} H5 files in {OUT_DIR}")


if __name__ == '__main__':
    main()
