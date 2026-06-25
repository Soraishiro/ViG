import math
import torch
from torch import nn

def build_pairwise_geo_features(boxes: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    B, N, _ = boxes.shape
    cx, cy, w, h = (boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3])
    dcx = cx.unsqueeze(2) - cx.unsqueeze(1)
    dcy = cy.unsqueeze(2) - cy.unsqueeze(1)
    w_i = w.unsqueeze(2).clamp_min(eps)
    h_i = h.unsqueeze(2).clamp_min(eps)
    w_j = w.unsqueeze(1).clamp_min(eps)
    h_j = h.unsqueeze(1).clamp_min(eps)
    dx_norm = dcx / w_i
    dy_norm = dcy / h_i
    dw_log = torch.log(w_j / w_i)
    dh_log = torch.log(h_j / h_i)
    x1_i = cx.unsqueeze(2) - w_i / 2
    y1_i = cy.unsqueeze(2) - h_i / 2
    x2_i = cx.unsqueeze(2) + w_i / 2
    y2_i = cy.unsqueeze(2) + h_i / 2
    x1_j = cx.unsqueeze(1) - w_j / 2
    y1_j = cy.unsqueeze(1) - h_j / 2
    x2_j = cx.unsqueeze(1) + w_j / 2
    y2_j = cy.unsqueeze(1) + h_j / 2
    inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp_min(0)
    inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp_min(0)
    inter = inter_w * inter_h
    area_i = (w.unsqueeze(2) * h.unsqueeze(2)).clamp_min(eps)
    area_j = (w.unsqueeze(1) * h.unsqueeze(1)).clamp_min(eps)
    union = area_i + area_j - inter
    iou = inter / union.clamp_min(eps)
    center_dist = torch.sqrt(dcx ** 2 + dcy ** 2 + eps)
    sin_theta = dcy / center_dist.clamp_min(eps)
    cos_theta = dcx / center_dist.clamp_min(eps)
    area_ratio_ji = area_j / area_i
    area_ratio_ij = area_i / area_j
    phi_geo = torch.stack([dx_norm, dy_norm, dw_log, dh_log, iou, center_dist, sin_theta, cos_theta, area_ratio_ji, area_ratio_ij], dim=-1)
    return phi_geo

def build_pairwise_vis_features(projected_features: torch.Tensor) -> torch.Tensor:
    p_i = projected_features.unsqueeze(2)
    p_j = projected_features.unsqueeze(1)
    hadamard = p_i * p_j
    abs_diff = torch.abs(p_i - p_j)
    return torch.cat([hadamard, abs_diff], dim=-1)

class TrigonometricEmbedding(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, wave_len: float=1000.0):
        super().__init__()
        self.out_dim = out_dim
        self.wave_len = wave_len
        bands = wave_len ** (torch.arange(0, out_dim // 2).float() / (out_dim // 2))
        self.register_buffer('bands', bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1) * self.bands
        sin_emb = torch.sin(x_exp)
        cos_emb = torch.cos(x_exp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb.flatten(-2)

class RelationGeometryBias(nn.Module):

    def __init__(self, n_heads: int=8, geo_dim: int=10, d_model: int=512, d_p: int=64, geo_mlp_hidden: int=128, dropout: float=0.2, lambda_geo: float=1.0, lambda_vis: float=1.0, use_trigonometric_embedding: bool=False, trig_wave_len: float=1000.0):
        super().__init__()
        self.n_heads = n_heads
        self.lambda_geo = lambda_geo
        self.lambda_vis = lambda_vis
        self._has_geo = abs(lambda_geo) > 1e-08
        self._has_vis = abs(lambda_vis) > 1e-08
        if not self._has_geo and (not self._has_vis):
            raise ValueError(f'[RRM] RelationGeometryBias: FATAL — both lambda_geo={lambda_geo} and lambda_vis={lambda_vis} are zero. At least one must be non-zero.')
        _status_geo = 'active' if self._has_geo else 'SKIPPED (lambda_geo=0, geo_mlp frozen)'
        _status_vis = 'active' if self._has_vis else 'SKIPPED (lambda_vis=0, vis_proj frozen)'
        print(f'[RRM] RelationGeometryBias: lambda_geo={lambda_geo} ({_status_geo}), lambda_vis={lambda_vis} ({_status_vis})')
        if use_trigonometric_embedding:
            trig_out = geo_dim * d_p
            self.trig_emb = TrigonometricEmbedding(geo_dim, d_p, trig_wave_len)
            geo_in = trig_out
        else:
            self.trig_emb = None
            geo_in = geo_dim
        self.geo_mlp = nn.Sequential(nn.Linear(geo_in, geo_mlp_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(geo_mlp_hidden, n_heads))
        self.vis_norm = nn.LayerNorm(d_model)
        self.vis_pair_proj = nn.Linear(d_model, d_p)
        vis_in = d_p * 2
        self.vis_proj = nn.Sequential(nn.Linear(vis_in, d_p * 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_p * 2, d_p), nn.ReLU(inplace=True), nn.Linear(d_p, n_heads))
        if not self._has_geo:
            self._freeze_module(self.geo_mlp, 'geo_mlp')
            if self.trig_emb is not None:
                self._freeze_module(self.trig_emb, 'trig_emb')
        if not self._has_vis:
            self._freeze_module(self.vis_norm, 'vis_norm')
            self._freeze_module(self.vis_pair_proj, 'vis_pair_proj')
            self._freeze_module(self.vis_proj, 'vis_proj')

    @staticmethod
    def _freeze_module(module, tag):
        n_params = sum((p.numel() for p in module.parameters()))
        for p in module.parameters():
            p.requires_grad = False
        print(f'[RRM] RelationGeometryBias: {tag} frozen ({n_params} params, requires_grad=False)')

    def forward(self, boxes: torch.Tensor, region_features: torch.Tensor) -> torch.Tensor:
        biases = []
        if self._has_geo:
            phi_geo = build_pairwise_geo_features(boxes)
            if self.trig_emb is not None:
                phi_geo = self.trig_emb(phi_geo)
            geo_bias = self.geo_mlp(phi_geo)
            geo_bias = geo_bias.permute(0, 3, 1, 2)
            biases.append(self.lambda_geo * geo_bias)
        if self._has_vis:
            projected = self.vis_pair_proj(self.vis_norm(region_features))
            phi_vis = build_pairwise_vis_features(projected)
            vis_bias = self.vis_proj(phi_vis)
            vis_bias = vis_bias.permute(0, 3, 1, 2)
            biases.append(self.lambda_vis * vis_bias)
        if not biases:
            raise RuntimeError('[RRM] RelationGeometryBias.forward: FATAL — no active branches. This should have been caught at init. Check _has_geo / _has_vis.')
        return biases[0] if len(biases) == 1 else biases[0] + biases[1]

class RelationEncoderLayer(nn.Module):

    def __init__(self, d_model: int=512, n_heads: int=8, dropout: float=0.2, ffn_dim: int=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.geo_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, R: torch.Tensor, geo_bias: torch.Tensor, objectness: torch.Tensor=None, return_attn: bool=False):
        B, N, D = R.shape
        H, Dh = (self.n_heads, self.head_dim)
        q = self.q_proj(R).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(R).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(R).view(B, N, H, Dh).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        logits = logits + self.geo_scale * geo_bias
        if objectness is not None:
            conf_bias = torch.log(torch.sigmoid(objectness).clamp_min(1e-06))
            logits = logits + conf_bias[:, None, None, :]
        attn = logits.softmax(dim=-1)
        attn = self.dropout(attn)
        msg = torch.matmul(attn, v)
        msg = msg.transpose(1, 2).contiguous().view(B, N, D)
        msg = self.out_proj(msg)
        R1 = self.norm1(R + self.dropout(msg))
        R2 = self.norm2(R1 + self.dropout(self.ffn(R1)))
        if return_attn:
            return (R2, attn)
        return R2

class RelationMemory(nn.Module):

    def __init__(self, config):
        super().__init__()
        cfg = config.model_ext.rrm
        self.d_model = config.model.d_model
        self.n_heads = cfg.n_heads
        self.dropout = getattr(cfg, 'dropout', 0.2)
        ffn_dim = getattr(config.model, 'd_ff', 2048)
        self.geo_bias = RelationGeometryBias(n_heads=self.n_heads, geo_dim=cfg.phi_geo_dim, d_model=self.d_model, d_p=cfg.d_p, geo_mlp_hidden=cfg.geo_mlp_hidden, dropout=self.dropout, lambda_geo=cfg.lambda_geo, lambda_vis=cfg.lambda_vis, use_trigonometric_embedding=cfg.use_trigonometric_embedding, trig_wave_len=cfg.trig_wave_len)
        self.layers = nn.ModuleList([RelationEncoderLayer(self.d_model, self.n_heads, self.dropout, ffn_dim) for _ in range(cfg.n_layers)])
        self.alpha_rel = nn.Parameter(torch.tensor(cfg.alpha_rel_init))

    def _init_ext_weights(self):
        pass

    def forward(self, R: torch.Tensor, boxes: torch.Tensor, objectness: torch.Tensor=None, return_aux: bool=False):
        geo_bias = self.geo_bias(boxes, R)
        X = R
        attn_list = []
        for layer in self.layers:
            if return_aux:
                X, attn = layer(X, geo_bias, objectness=objectness, return_attn=True)
                attn_list.append(attn)
            else:
                X = layer(X, geo_bias, objectness=objectness)
        R_rel = R + self.alpha_rel * (X - R)
        if return_aux:
            return (R_rel, {'relation_attn': attn_list, 'alpha_rel': self.alpha_rel.detach()})
        return R_rel
