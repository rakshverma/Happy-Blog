import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from linformer import LinformerSelfAttention      # pip install linformer

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# --------------------------Main------------------------------- #
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        num_channel = 31
        num_feature = 48
        ####################
        self.T_E = Transformer_E(num_feature)
        self.T_D = Transformer_D(num_feature)
        self.Embedding = nn.Sequential(
            nn.Linear(num_channel + 3, num_feature),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        ################LR-HSI###################
        UP_LRHSI = F.interpolate(HSI, scale_factor=4, mode='bicubic')
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)
        sz = UP_LRHSI.size(2)
        Data = torch.cat((UP_LRHSI, MSI), 1)
        E = rearrange(Data, 'B c H W -> B (H W) c', H=sz)
        E = self.Embedding(E)
        Code = self.T_E(E)
        Highpass = self.T_D(Code)
        Highpass = rearrange(Highpass, 'B (H W) C -> B C H W', H=sz)
        Highpass = self.refine(Highpass)
        output = Highpass + UP_LRHSI
        output = output.clamp_(0, 1)
        return output, UP_LRHSI, Highpass

# -----------------Transformer-----------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ----------------Linformer wrapper----------------
class LinformerAttention(nn.Module):
    """
    Drop-in replacement for the original Attention class
    using LinformerSelfAttention; preserves call signature.
    """
    def __init__(self, dim, seq_len, heads=3, k=256, dropout=0.):
        super().__init__()
        self.attn = LinformerSelfAttention(
            dim       = dim,
            seq_len   = seq_len,
            heads     = heads,
            k         = k,
            one_kv_head = True,
            share_kv    = True,
            dropout   = dropout
        )
    def forward(self, x, mask=None):
        return self.attn(x, mask)

# -------------Encoder--------------------------
class Transformer_E(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16,
                 mlp_dim=48, sp_sz=64*64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,
                         LinformerAttention(dim, seq_len=sp_sz,
                                            heads=heads, k=256,
                                            dropout=dropout))),
                Residual(PreNorm(dim,
                         FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

# -------------Decoder--------------------------
class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16,
                 mlp_dim=48, sp_sz=64*64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,
                         LinformerAttention(dim, seq_len=sp_sz,
                                            heads=heads, k=256,
                                            dropout=dropout))),
                Residual(PreNorm(dim,
                         LinformerAttention(dim, seq_len=sp_sz,
                                            heads=heads, k=256,
                                            dropout=dropout))),
                Residual(PreNorm(dim,
                         FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask=None):
        for attn1, attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x
