"""Configuration for QSVQ-Net (BRISC-2025).

Edit the paths in CFG to point to your local dataset copies before running.
"""
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


class CFG:
    base = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025"
    cls_train = f"{base}/classification_task/train"
    cls_test  = f"{base}/classification_task/test"
    seg_train_img = f"{base}/segmentation_task/train/images"
    seg_train_mask= f"{base}/segmentation_task/train/masks"
    seg_test_img  = f"{base}/segmentation_task/test/images"
    seg_test_mask = f"{base}/segmentation_task/test/masks"
    # external mask-free 4-class set: <ext_base>/<glioma|meningioma|notumor|pituitary>/*.jpg
    ext_base = "/kaggle/input/brain-tumor-mri-dataset/Testing"   # EDIT to your external set
    ext_class_map = {"glioma":"glioma","meningioma":"meningioma","notumor":"no_tumor",
                     "no_tumor":"no_tumor","pituitary":"pituitary"}
    img_size = 224
    classes = ["glioma","meningioma","no_tumor","pituitary"]; n_classes = 4
    val_frac = 0.15
    batch_size = 32
    backbone_epochs = 12; head_epochs = 15; seg_epochs = 10
    lr_backbone_A = 5e-5; lr_other_A = 3e-4; lr_backbone_B = 2e-5; lr_head_B = 1e-3; lr_seg = 1e-3
    weight_decay = 1e-4; grad_clip = 2.0; focal_gamma = 1.5; label_smoothing = 0.05; dropout = 0.3
    SNN_T = 8; snn_dim = 256; snn_beta = 0.9; snn_vth = 1.0; snn_surrogate_slope = 10.0; encoding = "rate"
    use_vqc = True; n_qubits = 8; VQC_LAYERS = 2
    morph_cond = True; vqc_as_mlp = False; disable_all_morph = False
    expr_pairs = 400; expr_bins = 50; ent_samples = 200
    n_folds = 5; seeds = [42, 7, 123]
    # ---- compute honesty switch ----
    FAST_ABLATION = True          # True: reduced epochs for ablation/CV (relative comparison).
    fast_epochs = 8               # set FAST_ABLATION=False for final paper numbers.
cfg = CFG()
