# QSVQ-Net: A Morphology-Conditioned Spiking–Variational Quantum Network for Interpretable Brain Tumour MRI Classification

This repository contains the source code for the QSVQ-Net model, a hybrid quantum–classical
framework for brain tumour MRI classification. It couples an EfficientNet-B0 backbone and a
complex-valued spiking neural network with a morphology-conditioned variational quantum circuit
(VQC), in which morphological and radiomic descriptors parameterise the qubit rotations.

## Overview

QSVQ-Net is evaluated as a transparent, rigorously characterised design rather than a claim of
accuracy superiority. On the BRISC-2025 dataset it performs comparably to strong classical CNN
baselines; a matched, multi-seed ablation shows the morphology-conditioned quantum branch performs
within confidence-interval overlap of a parameter-matched classical control, so no quantum accuracy
advantage is claimed. External cross-dataset validation reveals an honestly reported limitation of
the mask-conditioned design.

## Repository structure

| File | Description |
|------|-------------|
| `config.py`   | Configuration (`CFG` class): dataset paths, hyperparameters, seeds, device. Edit paths before running. |
| `data.py`     | Dataset indexing, preprocessing, radiomic/morphological descriptor extraction, leakage-safe pruning and scaling, and PyTorch datasets/loaders. |
| `model.py`    | Model components: complex-valued spiking neural network, morphology-conditioned VQC (with a parameter-matched classical control), auxiliary U-Net localiser, and the full QSVQ-Net classifier. |
| `train.py`    | Focal loss with label smoothing, the shared training loop, and the two-stage training procedure for the proposed model. |
| `evaluate.py` | Test-time augmentation and metric computation (accuracy, macro-F1, AUC, ECE, Brier). |

## Requirements

The code was developed and tested with:

- Python 3.11
- PyTorch and TorchVision
- PennyLane (quantum circuit simulation, `default.qubit` state-vector backend)
- NumPy, pandas, scikit-learn, scikit-image, OpenCV (`opencv-python`), SciPy

Install with:

```bash
pip install torch torchvision pennylane numpy pandas scikit-learn scikit-image opencv-python scipy
```

## Dataset

The primary dataset is **BRISC-2025** (classification and segmentation tasks). The external
evaluation set is the publicly available **Nickparvar Brain Tumour MRI Dataset**
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

Update the paths in `config.py` (`CFG.base`, `CFG.ext_base`, and the segmentation/classification
sub-paths) to point to your local copies before running.

## Usage

The modules are organised so they can be imported and composed. A typical workflow:

```python
from config import CFG, DEVICE, SEED, seed_everything
from data import train_loader, val_loader, test_loader, MORPH_DIM
from model import QSVQNet
from train import train_proposed        # two-stage training procedure
from evaluate import evaluate           # metric computation

seed_everything(SEED)
model = QSVQNet(MORPH_DIM).to(DEVICE)
# train and evaluate using the provided functions
```

The quantum circuit is simulated with an exact state-vector backend (PennyLane `default.qubit`,
adjoint differentiation, no finite-shot sampling and no hardware noise model). Near-term hardware
deployment is discussed as future work, not a demonstrated capability.

## Reproducibility

- Random seeds are fixed (`SEED = 42`; multi-seed experiments use {42, 7, 123}).
- Morphological descriptors are pruned and scaled within each fold to prevent leakage.
- Data splits are performed at the slice level; BRISC-2025 does not release subject identifiers,
  so a patient-level partition is not possible with the available metadata (stated as a limitation
  in the paper).

## Citation

If you use this code, please cite the associated paper:

> Abinash P., Krishnamoorthy N. "QSVQ-Net: A Morphology-Conditioned Spiking–Variational Quantum
> Network for Interpretable Brain Tumour MRI Classification." *Scientific Reports* (under review).

## License

Released for academic and research use.
