import torch
from torch import nn
from ..common.attention import MultiHeadAttention

class CulturalQueryMemory(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        d_model: int = getattr(config, 'd_model', 512)
        n_heads: int = getattr(config, 'n_heads', 8)
        k_loc: int = getattr(config, 'k_loc', 16)
        dropout: float = getattr(config, 'dropout', 0.1)
        ff_hidden: int = getattr(config, 'ff_hidden', 2048)
        slot_std: float = getattr(config, 'slot_init_std', 0.02)
        type_std: float = getattr(config, 'type_emb_std', 0.02)
        out_scale: float = getattr(config, 'out_scale', 0.01)
        self.k_loc = k_loc
        self.d_model = d_model
        self.out_scale = out_scale
        self.q_loc = nn.Parameter(torch.empty(k_loc, d_model))
        self.type_emb_rel = nn.Parameter(torch.empty(1, 1, d_model))
        self.type_emb_gri = nn.Parameter(torch.empty(1, 1, d_model))
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ff_hidden, d_model))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_mem = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.W_c = nn.Linear(d_model, 768)
        self.dropout = nn.Dropout(dropout)
        self._init_weights(slot_std, type_std)

    def _init_weights(self, slot_std: float, type_std: float) -> None:
        nn.init.normal_(self.q_loc, std=slot_std)
        nn.init.normal_(self.type_emb_rel, std=type_std)
        nn.init.normal_(self.type_emb_gri, std=type_std)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.weight.data.mul_(self.out_scale)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.W_c.weight)
        nn.init.zeros_(self.W_c.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_key_mask(self, B: int, N: int, M: int, grid_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        reg_mask = torch.zeros(B, 1, 1, N, dtype=torch.bool, device=device)
        return torch.cat([reg_mask, grid_mask], dim=-1)

    def forward(self, region_features: torch.Tensor, grid_features: torch.Tensor, grid_mask: torch.Tensor, return_attn: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = region_features.shape
        M = grid_features.shape[1]
        device = region_features.device
        rel_typed = region_features + self.type_emb_rel
        gri_typed = grid_features + self.type_emb_gri
        M_loc = torch.cat([rel_typed, gri_typed], dim=1)
        key_mask = self._build_key_mask(B, N, M, grid_mask, device)
        Q = self.q_loc.unsqueeze(0).expand(B, -1, -1)
        Q = self.ln_q(Q)
        M_loc = self.ln_mem(M_loc)
        result = self.cross_attn(Q, M_loc, M_loc, attention_mask=key_mask, return_attn=return_attn)
        if return_attn:
            attn_out, p_attn = result
        else:
            attn_out = result
            p_attn = None
        attn_out = self.dropout(attn_out)
        ffn_in = self.ln_ffn(attn_out + Q)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.dropout(ffn_out)
        hidden = self.ln_out(ffn_out + ffn_in)
        C_loc = self.out_proj(hidden)
        C_proj = self.W_c(C_loc)
        if return_attn:
            return (C_loc, C_proj, p_attn)
        return (C_loc, C_proj)

    def _init_vig_weights(self) -> None:
        self._init_weights(0.02, 0.02)
