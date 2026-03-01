"""Compute ISM/DeepSHAP attributions for mutagenesis libraries.

For each WT sequence, loads its mutagenesis library, takes first N_MUTANTS,
computes attributions for WT (idx 0) + mutants, saves to attribution_libraries/.

Resumable via checkpointing.

Usage (on GPU node with mpra_agft venv):
    python compute_attributions.py --model second_2stageig
    python compute_attributions.py --model second_2stageig --method ism
    python compute_attributions.py --model second_2stageig --method shap

SLURM array support (split WT sequences across tasks):
    #SBATCH --array=0-29
    python compute_attributions.py --model second_2stageig
"""

import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import h5py
import pandas as pd

# ---------- paths ----------
PROJ_DIR = Path(__file__).resolve().parents[4]  # LentiMoCon root
AGFT_DIR = PROJ_DIR / 'alphagenome_FT_MPRA'
sys.path.insert(0, str(AGFT_DIR))
sys.path.insert(0, str(PROJ_DIR / 'lenti_AGFT' / 'interpreting' / 'scripts'))

RESULTS_BASE = PROJ_DIR / 'lenti_AGFT' / 'training' / 'results'
ACTIVITY_LIB_BASE = PROJ_DIR / 'lenti_AGFT' / 'motif_context_swap' / 'library_prep' / 'activity_libraries'
ATTR_LIB_BASE = PROJ_DIR / 'lenti_AGFT' / 'motif_context_swap' / 'library_prep' / 'attribution_libraries'

_WEIGHTS_SUBPATH = 'huggingface/hub/models--google--alphagenome-all-folds/snapshots/a8f293a76ee73d5b57f3bf2ae146510589fcf187'
DEFAULT_WEIGHTS = Path('/grid/wsbs/home_norepl/pmantill/Liver_AGFT/.weights') / _WEIGHTS_SUBPATH

HEAD_NAME = 'mpra_head'
PROMOTER_SEQ = 'TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG'
RAND_BARCODE = 'AGAGACTGAGGCCAC'
SEQ_LENGTH = 230  # enhancer only
CONSTRUCT_LENGTH = SEQ_LENGTH + len(PROMOTER_SEQ) + len(RAND_BARCODE)  # 281

# Defaults
N_MUTANTS = 5000
N_SHUFFLES = 20
N_STEPS = 50
CHECKPOINT_EVERY = 500


def find_checkpoint(model_name):
    """Auto-detect best checkpoint: prefer best_stage2 over best."""
    ckpt_base = RESULTS_BASE / model_name / 'checkpoints'
    for subdir in ['best_stage2', 'best']:
        candidate = ckpt_base / subdir
        if (candidate / 'config.json').exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {ckpt_base}")


def patch_checkpoint_config(ckpt_dir):
    """Fix use_encoder_output bug in saved checkpoints."""
    config_path = ckpt_dir / 'config.json'
    with open(config_path) as f:
        cfg = json.load(f)
    if not cfg.get('use_encoder_output', False):
        cfg['use_encoder_output'] = True
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"Patched use_encoder_output=true in {config_path}")
    return cfg


def load_model(model_name):
    """Load fine-tuned AlphaGenome model."""
    ckpt_dir = find_checkpoint(model_name)
    cfg = patch_checkpoint_config(ckpt_dir)

    from alphagenome_ft import load_checkpoint, register_custom_head
    from alphagenome_ft import HeadConfig, HeadType
    from alphagenome.models import dna_output
    from src import EncoderMPRAHead

    head_meta = cfg.get('head_configs', {}).get(HEAD_NAME, {}).get('metadata', {})
    register_custom_head(HEAD_NAME, EncoderMPRAHead, HeadConfig(
        type=HeadType.GENOME_TRACKS, name=HEAD_NAME,
        output_type=dna_output.OutputType.RNA_SEQ, num_tracks=1,
        metadata=head_meta,
    ))

    model = load_checkpoint(
        str(ckpt_dir),
        base_checkpoint_path=str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else None,
        init_seq_len=CONSTRUCT_LENGTH,
    )
    print(f"Loaded model: {model_name} from {ckpt_dir}")
    return model


def enhancer_to_construct(enhancer_ohe):
    """Pad 230bp enhancer one-hot with promoter + barcode to get 281bp construct."""
    alpha_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    suffix = PROMOTER_SEQ + RAND_BARCODE
    suffix_ohe = np.zeros((len(suffix), 4), dtype=np.float32)
    for i, base in enumerate(suffix):
        suffix_ohe[i, alpha_map[base]] = 1.0
    return np.concatenate([enhancer_ohe, suffix_ohe], axis=0)


def compute_ism_single(model, seq_onehot, organism_index):
    """Compute ISM for a single sequence. Returns (230, 4) numpy array."""
    ism_attr = model.compute_ism_attributions(
        sequence=seq_onehot,
        organism_index=organism_index,
        head_name=HEAD_NAME,
    )
    return np.array(ism_attr[0, :SEQ_LENGTH, :], dtype=np.float32)


def compute_deepshap_single(model, seq_onehot, organism_index, n_shuffles, n_steps, seed=42):
    """Compute DeepSHAP for a single sequence. Returns (230, 4) numpy array."""
    from deepshap import deep_lift_shap
    attr = deep_lift_shap(
        model, seq_onehot, organism_index, HEAD_NAME,
        n_shuffles=n_shuffles, n_steps=n_steps,
        random_state=seed,
    )
    return attr[0, :SEQ_LENGTH, :]


def process_wt_library(model, organism_index, mut_lib_path, out_path,
                       method, n_mutants, n_shuffles, n_steps):
    """Compute attributions for a WT + its mutagenesis library.

    Saves to out_path with datasets:
        - '{method}': (N+1, 230, 4) — WT at idx 0, mutants at idx 1..N
        - 'wt_onehot': (230, 4)
    """
    # Load mutagenesis library
    with h5py.File(mut_lib_path, 'r') as f:
        all_mutants = f['sequences'][:]   # (25000, 230, 4)
        wt_enh = f['wt_sequence'][:]      # (230, 4)
        meta_attrs = dict(f.attrs)

    # Take first n_mutants
    mutants = all_mutants[:n_mutants]
    total = n_mutants + 1  # WT + mutants

    # Check if already complete
    if out_path.exists():
        with h5py.File(out_path, 'r') as f:
            if method in f and f[method].shape[0] == total:
                print(f"  {method} already complete ({total} maps)")
                return

    # Check for checkpoint
    checkpoint_path = out_path.parent / (out_path.stem + f'_{method}_checkpoint.h5')
    start_idx = 0

    if checkpoint_path.exists():
        with h5py.File(checkpoint_path, 'r') as f:
            if method in f:
                start_idx = int(f.attrs.get('last_completed', 0))
        print(f"  Resuming {method} from idx {start_idx}")

    print(f"  Computing {method} for {total} sequences (1 WT + {n_mutants} mutants)")
    t0 = time.time()

    # Build suffix one-hot once
    suffix_ohe = enhancer_to_construct(np.zeros((SEQ_LENGTH, 4)))[SEQ_LENGTH:]

    with h5py.File(checkpoint_path, 'a') as ckpt:
        if method not in ckpt:
            ckpt.create_dataset(method, shape=(total, SEQ_LENGTH, 4),
                                dtype='float32', compression='gzip', compression_opts=4)
        if 'wt_onehot' not in ckpt:
            ckpt.create_dataset('wt_onehot', data=wt_enh, dtype='float32')
        ckpt.attrs['last_completed'] = start_idx
        for k, v in meta_attrs.items():
            ckpt.attrs[k] = v
        ckpt.attrs['n_mutants'] = n_mutants
        ckpt.attrs['method'] = method

        for i in range(start_idx, total):
            # WT is idx 0, mutants are idx 1..N
            enh_ohe = wt_enh if i == 0 else mutants[i - 1]

            # Pad to construct length
            construct = np.concatenate([enh_ohe, suffix_ohe], axis=0)
            seq_jax = jnp.array(construct)[None, ...]  # (1, 281, 4)

            if method == 'ism':
                attr = compute_ism_single(model, seq_jax, organism_index)
            else:
                attr = compute_deepshap_single(
                    model, seq_jax, organism_index,
                    n_shuffles=n_shuffles, n_steps=n_steps, seed=42,
                )

            ckpt[method][i] = attr

            if (i + 1) % CHECKPOINT_EVERY == 0 or i == total - 1:
                ckpt.attrs['last_completed'] = i + 1
                ckpt.flush()
                elapsed = time.time() - t0
                rate = (i - start_idx + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    [{i+1}/{total}] {elapsed/60:.1f}min, ~{eta/60:.1f}min left")

    # Merge checkpoint into final output (append mode to preserve other methods)
    with h5py.File(checkpoint_path, 'r') as ckpt:
        with h5py.File(out_path, 'a') as out:
            if method in out:
                del out[method]
            out.create_dataset(method, data=ckpt[method][:],
                               dtype='float32', compression='gzip', compression_opts=4)
            if 'wt_onehot' not in out:
                out.create_dataset('wt_onehot', data=ckpt['wt_onehot'][:], dtype='float32')
            for k, v in ckpt.attrs.items():
                out.attrs[k] = v

    os.remove(checkpoint_path)
    elapsed = time.time() - t0
    print(f"  Done {method}: {total} maps in {elapsed/60:.1f}min -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute attributions for mutagenesis libraries')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--method', choices=['both', 'ism', 'shap'], default='both',
                        help='Which method(s) to compute (default: both)')
    parser.add_argument('--library_csv', default=None,
                        help='Path to activity library CSV (default: full library)')
    parser.add_argument('--n_mutants', type=int, default=N_MUTANTS,
                        help=f'Number of mutants per WT (default: {N_MUTANTS})')
    parser.add_argument('--n_shuffles', type=int, default=N_SHUFFLES)
    parser.add_argument('--n_steps', type=int, default=N_STEPS)
    args = parser.parse_args()

    do_ism = args.method in ('both', 'ism')
    do_shap = args.method in ('both', 'shap')

    # Load activity library
    if args.library_csv:
        lib_path = Path(args.library_csv)
    else:
        lib_path = ACTIVITY_LIB_BASE / args.model / f"k562_activity_library_{args.model}.csv"
    lib_df = pd.read_csv(lib_path)
    print(f"Activity library: {len(lib_df)} WT sequences")

    # Mutagenesis library directory
    mut_lib_dir = ACTIVITY_LIB_BASE / args.model / 'mutagenesis_libraries'
    if not mut_lib_dir.exists():
        raise FileNotFoundError(f"Mutagenesis libraries not found: {mut_lib_dir}")

    # SLURM array: partition WT sequences across tasks
    array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    indices = np.array_split(range(len(lib_df)), array_count)[array_id]
    if array_count > 1:
        print(f"SLURM array {array_id}/{array_count}: processing {len(indices)} WT sequences")

    # Output directory
    out_dir = ATTR_LIB_BASE / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model)
    organism_index = jnp.array([0])

    # HuggingFace patch
    from alphagenome_research.model import dna_model
    dna_model.create_from_kaggle = dna_model.create_from_huggingface

    # Process each WT
    for count, idx in enumerate(indices):
        row = lib_df.iloc[idx]
        label = f"{row['activity_bin']}_{row['seq_id']}"

        mut_lib_path = mut_lib_dir / f"{label}.h5"
        if not mut_lib_path.exists():
            print(f"\n[{count+1}/{len(indices)}] SKIP {label}: no mutagenesis library")
            continue

        out_path = out_dir / f"{label}.h5"

        print(f"\n{'='*60}")
        print(f"[{count+1}/{len(indices)}] {label} (activity={row['actual_activity']:.3f})")
        print(f"{'='*60}")

        if do_ism:
            process_wt_library(
                model, organism_index, mut_lib_path, out_path,
                method='ism', n_mutants=args.n_mutants,
                n_shuffles=args.n_shuffles, n_steps=args.n_steps,
            )

        if do_shap:
            process_wt_library(
                model, organism_index, mut_lib_path, out_path,
                method='deepshap', n_mutants=args.n_mutants,
                n_shuffles=args.n_shuffles, n_steps=args.n_steps,
            )

    print(f"\nAll done! Output: {out_dir}")


if __name__ == '__main__':
    main()
