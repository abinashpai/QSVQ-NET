"""Evaluation utilities for QSVQ-Net: test-time augmentation, metric computation
(accuracy, macro-F1, AUC, ECE, Brier) and calibration.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from config import CFG, DEVICE

cfg = CFG()

res=evaluate(model,test_loader)
json.dump({k:float(res[k]) for k in ["acc","macro_f1","auc","ece","brier"]},open(f"{OUT}/test_metrics.json","w"),indent=2)

# Test-Time Augmentation
@torch.no_grad()
def tta_eval(model,df,M):
    model.eval(); tfs=[eval_tf,
        transforms.Compose([transforms.Resize((cfg.img_size,)*2),transforms.RandomHorizontalFlip(1.0),transforms.ToTensor(),_norm])]
    probs=np.zeros((len(df),cfg.n_classes))
    for tf in tfs:
        L=[]
        for img,m,y in make_loader(df,M,tf,False):
            lg,_=model(img.to(DEVICE),m.to(DEVICE)); L.append(F.softmax(lg,1).cpu().numpy())
        probs+=np.concatenate(L)
    probs/=len(tfs); preds=probs.argmax(1); yt=df.label.values
tta_eval(model,test_df,M_test)
