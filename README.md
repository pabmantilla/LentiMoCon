# LentiMoCon

Fine-tuning generalist genomic models (AlphaGenome, Enformer) on LentiMPRA data to predict and interpret cis-regulatory activity in human enhancers across K562, HepG2, and WTC11 cell lines.

## Overview

This project leverages the [LentiMPRA dataset](https://doi.org/10.1101/2023.03.05.531189) (Agarwal et al. 2025), which systematically characterizes transcriptional regulatory elements via massively parallel reporter assays. We fine-tune large pretrained sequence-to-function models by freezing their backbones and training task-specific prediction heads on MPRA readouts, then interpret the learned regulatory logic through in silico mutagenesis and DeepSHAP.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `human_legnet/` | [MPRA-LegNet](https://github.com/autosome-ru/LegNet/) baseline model and LentiMPRA training data |
| `alphagenome_FT_MPRA/` | Fine-tuning framework for AlphaGenome (JAX) and Enformer (PyTorch) |
| `lenti_AGFT/` | Training runs, results, and model checkpoints |

## Quick Start

```bash
# Install AlphaGenome fine-tuning framework
cd alphagenome_FT_MPRA
pip install git+https://github.com/google-deepmind/alphagenome_research.git
pip install -e .

# Fine-tune on K562 LentiMPRA
python scripts/finetune_mpra.py --config configs/mpra_K562.json
```

See `alphagenome_FT_MPRA/README.md` and `human_legnet/README.md` for detailed documentation.
