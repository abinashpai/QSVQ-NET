"""Training utilities for QSVQ-Net: focal loss with label smoothing, shared epoch loop,
and the two-stage training procedure for the proposed model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, DEVICE, SEED
from model import QSVQNet

cfg = CFG()

def run_epoch(model,loader,opt=None):
    train=opt is not None; model.train() if train else model.eval()
    ce=nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing,reduction="none"); tot=correct=0; loss_sum=0.0
    for img,m,y in loader:
        img,m,y=img.to(DEVICE),m.to(DEVICE),y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits,_=model(img,m); logp=F.log_softmax(logits,1)
            pt=torch.exp(logp.gather(1,y[:,None])).squeeze(1); loss=((1-pt)**cfg.focal_gamma*ce(logits,y)).mean()
        if train:
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step()
        loss_sum+=loss.item()*y.size(0); tot+=y.size(0); correct+=(logits.argmax(1)==y).sum().item()
    return loss_sum/tot,correct/tot
@torch.no_grad()
def evaluate(model,loader):
    model.eval(); L=[]; Y=[]
    for img,m,y in loader:
        logits,_=model(img.to(DEVICE),m.to(DEVICE)); L.append(logits.cpu()); Y.append(y)
    logits=torch.cat(L); y=torch.cat(Y); probs=F.softmax(logits,1).numpy(); preds=probs.argmax(1); yt=y.numpy()
    acc=accuracy_score(yt,preds); f1m=f1_score(yt,preds,average="macro")
    try: auc=roc_auc_score(F.one_hot(y,cfg.n_classes).numpy(),probs,average="macro",multi_class="ovr")
    except Exception: auc=float("nan")
    conf=probs.max(1); cor=(preds==yt).astype(float); bins=np.linspace(0,1,11); ece=0.0
    for b in range(10):
        msk=(conf>bins[b])&(conf<=bins[b+1])
        if msk.sum()>0: ece+=msk.mean()*abs(cor[msk].mean()-conf[msk].mean())
    brier=np.mean(np.sum((probs-F.one_hot(y,cfg.n_classes).numpy())**2,1))
    return dict(acc=acc,macro_f1=f1m,auc=auc,ece=ece,brier=brier,preds=preds,probs=probs,y=yt,logits=logits.numpy())


cfg.use_vqc,cfg.morph_cond,cfg.vqc_as_mlp,cfg.disable_all_morph=True,True,False,False
seed_everything(SEED); model=QSVQNet(MORPH_DIM).to(DEVICE)
history={"tr_loss":[],"tr_acc":[],"val_loss":[],"val_acc":[]}; best_val=0.0; best_state=None
for p in model.vqc.parameters(): p.requires_grad_(False)
optA=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":cfg.lr_backbone_A},
    {"params":[p for n,p in model.named_parameters() if "backbone" not in n and "vqc" not in n],"lr":cfg.lr_other_A}],
    weight_decay=cfg.weight_decay)
schA=torch.optim.lr_scheduler.CosineAnnealingLR(optA,cfg.backbone_epochs)
for ep in range(cfg.backbone_epochs):
    trl,tra=run_epoch(model,train_loader,optA); vll,vla=run_epoch(model,val_loader); schA.step()
    for k,v in zip(history,[trl,tra,vll,vla]): history[k].append(v)
    if vla>best_val: best_val=vla; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
for p in model.vqc.parameters(): p.requires_grad_(True)
optB=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":cfg.lr_backbone_B},
    {"params":model.vqc.parameters(),"lr":cfg.lr_head_B},{"params":model.head.parameters(),"lr":cfg.lr_head_B},
    {"params":model.fusion.parameters(),"lr":cfg.lr_head_B}],weight_decay=cfg.weight_decay)
schB=torch.optim.lr_scheduler.CosineAnnealingLR(optB,cfg.head_epochs)
for ep in range(cfg.head_epochs):
    trl,tra=run_epoch(model,train_loader,optB); vll,vla=run_epoch(model,val_loader); schB.step()
    for k,v in zip(history,[trl,tra,vll,vla]): history[k].append(v)
    if vla>best_val: best_val=vla; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
if best_state: model.load_state_dict(best_state)
def cfg_to_dict(c): return {k:getattr(c,k) for k in dir(c) if not k.startswith("__") and not callable(getattr(c,k))}
ckpt=dict(state_dict=model.state_dict(),cfg=cfg_to_dict(cfg),keep=KEEP,morph_names=NAMES,
          scaler_mean=SCALER.mean_.tolist(),scaler_scale=SCALER.scale_.tolist(),versions=VERSIONS,best_val=best_val)
torch.save(ckpt,f"{OUT}/qsvqnet_final.pt"); json.dump(history,open(f"{OUT}/history.json","w"))
