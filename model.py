"""QSVQ-Net model definition: complex-valued spiking neural network, morphology-conditioned
variational quantum circuit (with parameter-matched classical control), auxiliary U-Net localiser,
and the full hybrid classifier.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import CFG, DEVICE

cfg = CFG()

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): ctx.save_for_backward(x); return (x>=0).float()
    @staticmethod
    def backward(ctx,g):
        (x,)=ctx.saved_tensors; return g/(cfg.snn_surrogate_slope*torch.abs(x)+1)**2
spike_fn=SurrogateSpike.apply
class ComplexLIFLayer(nn.Module):
    def __init__(self,i,o,beta=cfg.snn_beta,vth=cfg.snn_vth):
        super().__init__(); self.Wr=nn.Linear(i,o,bias=False); self.Wi=nn.Linear(i,o,bias=False)
        self.beta=beta; self.vth=vth
    def forward(self,spikes):
        B=spikes[0].shape[0]; zr=torch.zeros(B,self.Wr.out_features,device=spikes[0].device); zi=torch.zeros_like(zr); out=[]
        for s in spikes:
            zr=self.beta*zr+self.Wr(s); zi=self.beta*zi+self.Wi(s)
            amp=torch.sqrt(zr**2+zi**2+1e-8); sp=spike_fn(amp-self.vth)
            pr=zr/(amp+1e-8); pi=zi/(amp+1e-8); zr=zr-self.vth*pr*sp; zi=zi-self.vth*pi*sp; out.append(sp)
        return out
class SpikeEncoder(nn.Module):
    def __init__(self,T=cfg.SNN_T,mode=cfg.encoding): super().__init__(); self.T=T; self.mode=mode
    def forward(self,x):
        x=torch.sigmoid(x); out=[]
        if self.mode=="rate":
            for _ in range(self.T): out.append((torch.rand_like(x)<x).float())
        else:
            ts=((1-x)*(self.T-1)).round()
            for t in range(self.T): out.append((ts==t).float())
        return out
class ComplexSNN(nn.Module):
    def __init__(self,out_dim=cfg.snn_dim):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(),
                                nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool2d(1); self.proj=nn.Linear(64,out_dim); self.enc=SpikeEncoder()
        self.l1=ComplexLIFLayer(out_dim,out_dim); self.l2=ComplexLIFLayer(out_dim,out_dim); self.res=ComplexLIFLayer(out_dim,out_dim)
        self.attn_map=None
    def forward(self,img):
        fm=self.stem(img); self.attn_map=fm.mean(1,keepdim=True).detach()
        v=self.proj(self.pool(fm).flatten(1)); sp=self.enc(v); s2=self.l2(self.l1(sp)); sr=self.res(s2)
        merged=[a+b for a,b in zip(s2,sr)]; return torch.stack(merged,0).mean(0),merged


#           (clean quantum-vs-morphology isolation)  (R2 §3.1)
class MorphFusion(nn.Module):   # pathway (i): SE-style gating
    def __init__(self,fd,md):
        super().__init__(); self.gate=nn.Sequential(nn.Linear(md,fd),nn.ReLU(),nn.Linear(fd,fd),nn.Sigmoid())
    def forward(self,f,m): return f*self.gate(m)
dev=qml.device("default.qubit",wires=cfg.n_qubits)     # exact state-vector, NO shots
@qml.qnode(dev,interface="torch",diff_method="adjoint")
def vqc_circuit(inputs,rz,weights):
    for i in range(cfg.n_qubits): qml.RY(inputs[i],wires=i)
    for i in range(cfg.n_qubits): qml.RZ(rz[i],wires=i)
    for l in range(cfg.VQC_LAYERS):
        for i in range(cfg.n_qubits): qml.RY(weights[l,i,0],wires=i); qml.RZ(weights[l,i,1],wires=i)
        for i in range(cfg.n_qubits): qml.CNOT(wires=[i,(i+1)%cfg.n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]
class QuantumOrMLP(nn.Module):
    def __init__(self,in_dim,md):
        super().__init__()
        self.reduce=nn.Linear(in_dim,cfg.n_qubits)       # W_r angle-encoding map
        self.m2rz=nn.Linear(md,cfg.n_qubits)             # pathway (ii): morphology -> RZ
        self.weights=nn.Parameter(0.1*torch.randn(cfg.VQC_LAYERS,cfg.n_qubits,2))
        self.mlp=nn.Sequential(nn.Linear(cfg.n_qubits*2,32),nn.ReLU(),nn.Linear(32,cfg.n_qubits))
    def forward(self,f,m):
        enc=torch.tanh(self.reduce(f))*math.pi
        rz=(self.m2rz(m)*math.pi) if cfg.morph_cond else torch.zeros_like(enc)
        if cfg.vqc_as_mlp or not cfg.use_vqc:
            return torch.tanh(self.mlp(torch.cat([enc,rz],1)))    # classical control, same inputs
        enc_c,rz_c,w_c=enc.cpu(),rz.cpu(),self.weights.cpu()
        out=torch.stack([torch.stack(vqc_circuit(enc_c[b],rz_c[b],w_c)) for b in range(f.shape[0])]).float()
        return out.to(f.device)


class TinyUNet(nn.Module):
    def __init__(self,ch=32):
        super().__init__()
        b=lambda i,o:nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU())
        self.e1=b(3,ch); self.e2=b(ch,ch*2); self.e3=b(ch*2,ch*4); self.pool=nn.MaxPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.d2=b(ch*4+ch*2,ch*2); self.d1=b(ch*2+ch,ch); self.out=nn.Conv2d(ch,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2))
        d2=self.d2(torch.cat([self.up(e3),e2],1)); d1=self.d1(torch.cat([self.up(d2),e1],1)); return self.out(d1)
def dice_score(p,g,eps=1e-6):
    p=(torch.sigmoid(p)>0.5).float(); inter=(p*g).sum((1,2,3)); union=p.sum((1,2,3))+g.sum((1,2,3))
    return ((2*inter+eps)/(union+eps))
class SegDataset(Dataset):
    def __init__(self,idir,mdir):
        self.items=[f for f in os.listdir(idir) if f.endswith(".jpg")]; self.idir=idir; self.mdir=mdir
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        f=self.items[i]; img=cv2.resize(cv2.imread(os.path.join(self.idir,f)),(cfg.img_size,)*2)
        img=torch.tensor(img/255.,dtype=torch.float32).permute(2,0,1)
        mp=os.path.join(self.mdir,f.replace(".jpg",".png"))
        m=cv2.resize(cv2.imread(mp,cv2.IMREAD_GRAYSCALE),(cfg.img_size,)*2) if os.path.exists(mp) else np.zeros((cfg.img_size,)*2)
        return img,torch.tensor((m>127)[None],dtype=torch.float32),f
@torch.no_grad()
def predict_mask_unet(unet,path):
    unet.eval(); img=cv2.resize(cv2.imread(path),(cfg.img_size,)*2)
    x=torch.tensor(img/255.,dtype=torch.float32).permute(2,0,1)[None].to(DEVICE)
    return (torch.sigmoid(unet(x))[0,0]>0.5).cpu().numpy()


class QSVQNet(nn.Module):
    def __init__(self,md,nc=cfg.n_classes):
        super().__init__()
        bb=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone=bb.features; self.bpool=nn.AdaptiveAvgPool2d(1); self.bproj=nn.Linear(1280,cfg.snn_dim)
        self.snn=ComplexSNN(cfg.snn_dim); self.fusion=MorphFusion(cfg.snn_dim,md); self.vqc=QuantumOrMLP(cfg.snn_dim,md)
        self.head=nn.Sequential(nn.Linear(cfg.snn_dim+md+cfg.n_qubits,256),nn.BatchNorm1d(256),
                                nn.ReLU(),nn.Dropout(cfg.dropout),nn.Linear(256,nc))
        self.last_spikes=None; self.last_fq=None
    def forward(self,img,m):
        bb=self.bproj(self.bpool(self.backbone(img)).flatten(1))
        fs,sp=self.snn(img); self.last_spikes=sp; fs=fs+bb
        m_eff=torch.zeros_like(m) if cfg.disable_all_morph else m      # no-M-anywhere ablation arm
        ff=self.fusion(fs,m_eff); fq=self.vqc(ff,m_eff); self.last_fq=fq.detach()
        return self.head(torch.cat([ff,m_eff,fq],1)),fq
def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
seed_everything(SEED); model=QSVQNet(MORPH_DIM).to(DEVICE)
