"""Compute ISM and DeepSHAP attributions for all sequences in an activity library.

Saves per-sequence HDF5 files to attribution_libraries/{MODEL_NAME}/.
Resumable: skips sequences whose output files already exist.

Usage (on GPU node with mpra_agft venv):
    python compute_attributions.py --model second_2stageig
    python compute_attributions.py --model second_2stageig --method ism   # ISM only
    python compute_attributions.py --model second_2stageig --method shap  # DeepSHAP only

SLURM array support (split across N tasks):
    #SBATCH --array=0-9
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

# DeepSHAP defaults
N_SHUFFLES = 20
N_STEPS = 50


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


def seq_to_model_input(seq_str, model):
    """Convert 230bp enhancer string to model-ready one-hot (1, 281, 4)."""
    full_seq = seq_str + PROMOTER_SEQ + RAND_BARCODE
    assert len(full_seq) == CONSTRUCT_LENGTH, f"Expected {CONSTRUCT_LENGTH}bp, got {len(full_seq)}"
    ohe = model._one_hot_encoder.encode(full_seq)
    return jnp.array(ohe)[None, ...]  # (1, 281, 4)


def compute_ism(model, seq_onehot, organism_index):
    """Run ISM using the built-in method. Returns (1, 281, 4) numpy array."""
    ism_attr = model.compute_ism_attributions(
        sequence=seq_onehot,
        organism_index=organism_index,
        head_name=HEAD_NAME,
    )
    return np.array(ism_attr, dtype=np.float32)


def compute_deepshap(model, seq_onehot, organism_index, seed=42):
    """Run our custom DeepSHAP. Returns (1, 281, 4) numpy array."""
    from deepshap import deep_lift_shap
    attr = deep_lift_shap(
        model, seq_onehot, organism_index, HEAD_NAME,
        n_shuffles=N_SHUFFLES, n_steps=N_STEPS,
        random_state=seed,
    )
    return attr


def main():
    parser = argparse.ArgumentParser(description='Compute ISM/DeepSHAP attributions for activity library')
    parser.add_argument('--model', required=True, help='Model name (e.g. second_2stageig)')
    parser.add_argument('--method', choices=['both', 'ism', 'shap'], default='both',
                        help='Which attribution method(s) to compute')
    parser.add_argument('--n_shuffles', type=int, default=N_SHUFFLES)
    parser.add_argument('--n_steps', type=int, default=N_STEPS)
    args = parser.parse_args()

    do_ism = args.method in ('both', 'ism')
    do_shap = args.method in ('both', 'shap')

    # Load activity library
    lib_path = ACTIVITY_LIB_BASE / args.model / f"k562_activity_library_{args.model}.csv"
    lib_df = pd.read_csv(lib_path)
    print(f"Activity library: {len(lib_df)} sequences from {lib_path.name}")

    # SLURM array: partition sequences
    array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    indices = np.array_split(range(len(lib_df)), array_count)[array_id]
    if array_count > 1:
        print(f"SLURM array {array_id}/{array_count}: processing {len(indices)} sequences")

    # Output directory
    out_dir = ATTR_LIB_BASE / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model)
    from alphagenome_research.model import dna_model
    organism_index = jnp.array([0])  # HOMO_SAPIENS

    # Process sequences
    total = len(indices)
    t_start = time.time()

    for count, idx in enumerate(indices):
        row = lib_df.iloc[idx]
        seq_id = row['seq_id']
        activity_bin = row['activity_bin']
        out_file = out_dir / f"{activity_bin}_{seq_id}.h5"

        # Check if already done
        if out_file.exists():
            with h5py.File(out_file, 'r') as f:
                has_ism = 'ism' in f
                has_shap = 'deepshap' in f
            if (not do_ism or has_ism) and (not do_shap or has_shap):
                continue

        # Encode: 230bp enhancer -> 281bp construct
        seq_onehot = seq_to_model_input(row['sequence'], model)

        # Open file for writing (append mode to preserve existing attrs)
        with h5py.File(out_file, 'a') as f:
            # ISM
            if do_ism and 'ism' not in f:
                ism_attr = compute_ism(model, seq_onehot, organism_index)
                # Store only enhancer region (first 230 positions)
                f.create_dataset('ism', data=ism_attr[:, :SEQ_LENGTH, :],
                                 dtype='float32', compression='gzip', compression_opts=4)

            # DeepSHAP
            if do_shap and 'deepshap' not in f:
                shap_attr = compute_deepshap(model, seq_onehot, organism_index, seed=42)
                f.create_dataset('deepshap', data=shap_attr[:, :SEQ_LENGTH, :],
                                 dtype='float32', compression='gzip', compression_opts=4)

            # Metadata (write once)
            if 'wt_onehot' not in f:
                wt_np = np.array(seq_onehot[0, :SEQ_LENGTH, :], dtype=np.float32)
                f.create_dataset('wt_onehot', data=wt_np, dtype='float32')
                f.attrs['seq_id'] = seq_id
                f.attrs['activity_bin'] = activity_bin
                f.attrs['actual_activity'] = float(row['actual_activity'])
                f.attrs['split'] = row['split']
                f.attrs['fold'] = int(row['fold'])
                f.attrs['seq_length'] = SEQ_LENGTH
                f.attrs['construct_length'] = CONSTRUCT_LENGTH
                f.attrs['alphabet'] = 'ACGT'
                f.attrs['model'] = args.model
                if do_shap:
                    f.attrs['n_shuffles'] = args.n_shuffles
                    f.attrs['n_steps'] = args.n_steps

        if (count + 1) % 10 == 0 or count == 0:
            elapsed = time.time() - t_start
            rate = (count + 1) / elapsed
            eta = (total - count - 1) / rate if rate > 0 else 0
            print(f"  [{count+1}/{total}] {activity_bin}/{seq_id} "
                  f"({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)")

    elapsed = time.time() - t_start
    print(f"\nDone! {total} sequences in {elapsed/60:.1f} minutes")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()
