"""VBT-xAI model: Gated Cross-Attention CBAM-EfficientNet + Vision Transformer.

Standalone version of the model defined in Cell 7 of
notebooks/VBT-xAI_Br35H_end_to_end.ipynb. Layer names and structure are identical,
so a checkpoint saved by the notebook (`best_model.pth`, key 'state') loads directly:

    import torch
    from model import HybridModel
    model = HybridModel(pretrained=False)
    ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
"""
import torch
import torch.nn as nn
import timm

# Defaults used in the paper (LOW_MEMORY_MODE = True in the notebook)
CNN_BACKBONE = "efficientnet_b0"
VIT_BACKBONE = "vit_small_patch16_224"
IMG_SIZE = 224
FUSION_DIM = 512
DROPOUT = 0.4
NUM_CLASSES = 2


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention, then spatial attention."""

    def __init__(self, ch, r=16, k=7):
        super().__init__()
        r = max(1, min(r, ch))
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp = nn.Sequential(nn.Flatten(),
                                    nn.Linear(ch, ch // r, bias=False), nn.ReLU(),
                                    nn.Linear(ch // r, ch, bias=False))
        self.sp_conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ch = self.sig((self.ch_mlp(self.ch_avg(x)) + self.ch_mlp(self.ch_max(x))).unsqueeze(2).unsqueeze(3))
        x = x * ch
        sp = self.sig(self.sp_conv(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))
        return x * sp


class CNNBranch(nn.Module):
    """Local anatomical pathway: EfficientNet-B0 -> CBAM -> GAP -> 512-d projection."""

    def __init__(self, name=CNN_BACKBONE, out=FUSION_DIM, dropout=DROPOUT,
                 img_size=IMG_SIZE, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        with torch.no_grad():
            ch = self.backbone.forward_features(torch.zeros(1, 3, img_size, img_size)).shape[1]
        self.cbam = CBAM(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.LayerNorm(ch),
                                  nn.Linear(ch, out), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = self.cbam(f)
        return self.proj(self.pool(f))

    def last_conv_layer(self):
        """Final conv layer of the backbone (Grad-CAM++ / Score-CAM target)."""
        for attr in ("conv_head", "conv_head_2", "conv_final"):
            if hasattr(self.backbone, attr):
                return getattr(self.backbone, attr)
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError("No Conv2d layer found in the CNN backbone.")
        return convs[-1]


class ViTBranch(nn.Module):
    """Global context pathway: ViT-Small/16 [CLS] embedding -> 512-d projection."""

    def __init__(self, name=VIT_BACKBONE, out=FUSION_DIM, dropout=DROPOUT, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        vit_dim = self.vit.embed_dim
        self.proj = nn.Sequential(nn.LayerNorm(vit_dim),
                                  nn.Linear(vit_dim, out), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.proj(self.vit(x))


class CrossAttnFusion(nn.Module):
    """8-head cross-attention: CNN feature = query, ViT feature = key and value (Eqs. 4-5)."""

    def __init__(self, dim=FUSION_DIM, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(dim * 4, dim))
        self.n3 = nn.LayerNorm(dim)

    def forward(self, cnn, vit):
        q = cnn.unsqueeze(1)
        k = vit.unsqueeze(1)
        out, w = self.attn(self.n1(q), self.n2(k), self.n2(k))
        out = q + out
        out = out + self.ffn(self.n3(out))
        return out.squeeze(1), w


class HybridModel(nn.Module):
    """VBT-xAI: z = g1*cf + g2*vf + z_attn  ->  classification head (Eq. 6).

    forward(x) returns (main_logits, cnn_aux_logits, vit_aux_logits).
    forward(x, return_feats=True) returns (main_logits, fused_embedding, attention_weights).
    """

    def __init__(self, nc=NUM_CLASSES, dim=FUSION_DIM, cnn_backbone=CNN_BACKBONE,
                 vit_backbone=VIT_BACKBONE, dropout=DROPOUT, img_size=IMG_SIZE, pretrained=True):
        super().__init__()
        self.cnn = CNNBranch(cnn_backbone, dim, dropout, img_size, pretrained)
        self.vit = ViTBranch(vit_backbone, dim, dropout, pretrained)
        self.fusion = CrossAttnFusion(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, 2), nn.Softmax(dim=-1))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, nc))
        self.cnn_aux = nn.Linear(dim, nc)
        self.vit_aux = nn.Linear(dim, nc)

    def forward(self, x, return_feats=False):
        cf = self.cnn(x)
        vf = self.vit(x)
        fused, attn = self.fusion(cf, vf)
        g = self.gate(torch.cat([cf, vf], dim=-1))
        z = g[:, 0:1] * cf + g[:, 1:2] * vf + fused
        logits = self.head(z)
        if return_feats:
            return logits, z, attn
        return logits, self.cnn_aux(cf), self.vit_aux(vf)


class ModelWrapper(nn.Module):
    """Returns only the main logits; used by Grad-CAM, Captum, SHAP and LIME."""

    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        logits, _, _ = self.model(x)
        return logits


if __name__ == "__main__":
    m = HybridModel(pretrained=False)
    n = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {n:,}")  # about 30.02 M
    out = m(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE))
    print([o.shape for o in out])
