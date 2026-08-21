"""Data pipeline for QSVQ-Net: dataset indexing, preprocessing, radiomic/morphological
descriptor extraction, leakage-safe pruning/scaling, and PyTorch datasets/loaders.
"""
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label, regionprops, perimeter as sk_perimeter
from skimage.feature import graycomatrix, graycoprops

from config import CFG, SEED

cfg = CFG()

def parse_meta(fname):
    plane = "unk"; mm = re.search(r"_(ax|co|sa)_", fname)
    if mm: plane = mm.group(1)
    return plane, os.path.splitext(fname)[0]   # slice_id: unique per slice, NOT a patient id

def index_brisc(root):
    rows=[]
    for ci,c in enumerate(cfg.classes):
        d=os.path.join(root,c)
        if not os.path.isdir(d):
            alt={"no_tumor":"notumor"}.get(c)
            if alt and os.path.isdir(os.path.join(root,alt)): d=os.path.join(root,alt)
        for f in os.listdir(d):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                pl,sid=parse_meta(f)
                rows.append(dict(path=os.path.join(d,f),label=ci,cls=c,plane=pl,slice_id=sid,fname=f))
    return pd.DataFrame(rows).reset_index(drop=True)

train_df_all = index_brisc(cfg.cls_train); test_df = index_brisc(cfg.cls_test)
# transparency: BRISC exposes no subject id -> slice-level split, reported as a limitation.
strat = train_df_all[["label","plane"]].astype(str).agg("-".join, axis=1)
train_df, val_df = train_test_split(train_df_all, test_size=cfg.val_frac, stratify=strat, random_state=SEED)
train_df=train_df.reset_index(drop=True); val_df=val_df.reset_index(drop=True)


clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def roi_mask(g8):
    try: t=threshold_otsu(g8)
    except Exception: t=127
    bw=g8>t; lbl=cc_label(bw)
    if lbl.max()==0: return np.zeros_like(bw,bool)
    p=max(regionprops(lbl),key=lambda r:r.area); return lbl==p.label
def build_mask_index():
    idx={}
    for mdir in [cfg.seg_train_mask,cfg.seg_test_mask]:
        if os.path.isdir(mdir):
            for f in os.listdir(mdir):
                if f.lower().endswith(".png"): idx[os.path.splitext(f)[0]]=os.path.join(mdir,f)
    return idx
MASK_IDX=build_mask_index(); print("verified masks indexed:",len(MASK_IDX))
def load_mask_for(path):
    mp=MASK_IDX.get(os.path.splitext(os.path.basename(path))[0])
    if mp is None: return None
    m=cv2.resize(cv2.imread(mp,cv2.IMREAD_GRAYSCALE),(cfg.img_size,)*2); return m>127
def preprocess_gray(path):
    raw=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(cfg.img_size,)*2); return raw,clahe.apply(raw)

# VISUAL: preprocessing + mask overlay
p=train_df[train_df.cls=="glioma"].path.iloc[0]; raw,enh=preprocess_gray(p); mk=load_mask_for(p)
axes[0].imshow(raw,cmap="gray"); axes[0].set_title("raw"); axes[0].axis("off")
axes[1].imshow(enh,cmap="gray"); axes[1].set_title("CLAHE"); axes[1].axis("off")
axes[2].imshow(enh,cmap="gray")
if mk is not None: axes[2].imshow(mk,alpha=0.4,cmap="autumn")
axes[2].set_title("verified mask"); axes[2].axis("off")
axes[3].imshow(roi_mask(enh),cmap="gray"); axes[3].set_title("Otsu fallback"); axes[3].axis("off")


# Discretisation: g//16 -> 16 grey levels. GLCM distances=[1], angles=[0,45,90,135deg],
# levels=16, symmetric+normed, averaged over angles. Shape from largest CC. 16-bin first-order.
GLCM_LEVELS=16; GLCM_DIST=[1]; GLCM_ANGLES=[0,np.pi/4,np.pi/2,3*np.pi/4]
def first_order(v):
    if v.size==0: v=np.zeros(1)
    h=np.histogram(v,16,density=True)[0]
    return [v.mean(),v.std(),v.min(),v.max(),np.median(v),
            float(((v-v.mean())**3).mean()/(v.std()**3+1e-6)),
            float(((v-v.mean())**4).mean()/(v.std()**4+1e-6)),
            float(-np.sum(h*np.log(h+1e-12)))]
def glcm_feats(g8,m):
    g=g8.copy(); g[~m]=0; g=(g//GLCM_LEVELS).astype(np.uint8)
    G=graycomatrix(g,GLCM_DIST,GLCM_ANGLES,levels=GLCM_LEVELS,symmetric=True,normed=True)
    return [float(graycoprops(G,p).mean()) for p in
            ["contrast","correlation","homogeneity","energy","dissimilarity","ASM"]]
def glrlm(g8,m):
    g=(g8//GLCM_LEVELS).astype(np.uint8); g[~m]=0; sre=lre=0.0; runs=0
    for row in g:
        i=0
        while i<len(row):
            j=i
            while j+1<len(row) and row[j+1]==row[i]: j+=1
            rl=j-i+1
            if row[i]>0: sre+=1/rl**2; lre+=rl**2; runs+=1
            i=j+1
    runs=max(runs,1); return [sre/runs,lre/runs,runs]
def shape_feats(m):
    lbl=cc_label(m)
    if lbl.max()==0: return [0]*8
    p=max(regionprops(lbl),key=lambda r:r.area); area=p.area; per=sk_perimeter(m)+1e-6
    return [area,per,4*np.pi*area/per**2,p.solidity,p.eccentricity,p.extent,
            p.major_axis_length/(p.minor_axis_length+1e-6),p.convex_area/(area+1e-6)]
RAW_NAMES=(["area","perimeter","circularity","solidity","eccentricity","extent","axis_ratio","convexity"]
           +["glcm_contrast","glcm_corr","glcm_homog","glcm_energy","glcm_dissim","glcm_asm"]
           +["glrlm_sre","glrlm_lre","glrlm_runs"]
           +["fo_mean","fo_std","fo_min","fo_max","fo_median","fo_skew","fo_kurt","fo_entropy"])
def extract_morph(path,use_predicted_mask=False,unet=None):
    raw,enh=preprocess_gray(path)
    if use_predicted_mask and unet is not None: m=predict_mask_unet(unet,path)
    else:
        m=load_mask_for(path)
        if m is None or m.sum()<20: m=roi_mask(enh)
    v=enh[m].astype(np.float32)
    return np.array(shape_feats(m)+glcm_feats(enh,m)+glrlm(enh,m)+first_order(v),dtype=np.float32)


def fit_prune_scale(M_raw):
    corr=np.corrcoef(np.nan_to_num(M_raw).T); drop=set()
    for i in range(corr.shape[0]):
        for j in range(i+1,corr.shape[0]):
            if i not in drop and j not in drop and abs(corr[i,j])>0.95: drop.add(j)
    keep=[k for k in range(corr.shape[0]) if k not in drop]
    scaler=StandardScaler().fit(np.nan_to_num(M_raw)[:,keep]); return keep,scaler
def apply_prune_scale(M_raw,keep,scaler):
    return scaler.transform(np.nan_to_num(M_raw)[:,keep]).astype(np.float32)
def build_raw_cache(df,tag):
    fp=f"{CACHE}/morph_raw_{tag}.npy"
    if os.path.exists(fp):
        arr=np.load(fp)
        if len(arr)==len(df): return arr
    arr=np.stack([extract_morph(p) for p in df.path.tolist()]); np.save(fp,arr); return arr

t0=time.time()
M_train_raw=build_raw_cache(train_df,"train"); M_val_raw=build_raw_cache(val_df,"val"); M_test_raw=build_raw_cache(test_df,"test")
KEEP,SCALER=fit_prune_scale(M_train_raw); NAMES=[RAW_NAMES[k] for k in KEEP]
M_train=apply_prune_scale(M_train_raw,KEEP,SCALER); M_val=apply_prune_scale(M_val_raw,KEEP,SCALER); M_test=apply_prune_scale(M_test_raw,KEEP,SCALER)
MORPH_DIM=M_train.shape[1]; print("features after prune:",MORPH_DIM,"->",NAMES)


_norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_tf=transforms.Compose([transforms.Resize((cfg.img_size,)*2),transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),transforms.ColorJitter(0.1,0.1),transforms.ToTensor(),_norm])
eval_tf=transforms.Compose([transforms.Resize((cfg.img_size,)*2),transforms.ToTensor(),_norm])
class MRIDataset(Dataset):
    def __init__(self,df,morph,tf): self.df=df.reset_index(drop=True); self.M=morph; self.tf=tf
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        x=self.tf(Image.open(self.df.path.iloc[i]).convert("RGB"))
        return x,torch.tensor(self.M[i],dtype=torch.float32),int(self.df.label.iloc[i])
def make_loader(df,morph,tf,shuffle):
    return DataLoader(MRIDataset(df,morph,tf),batch_size=cfg.batch_size,shuffle=shuffle,
                      num_workers=2,pin_memory=True,drop_last=shuffle)
train_loader=make_loader(train_df,M_train,train_tf,True)
val_loader=make_loader(val_df,M_val,eval_tf,False)
test_loader=make_loader(test_df,M_test,eval_tf,False)
