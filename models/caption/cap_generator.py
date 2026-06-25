import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from models.common.attention import MultiHeadAttention, Attention
from models.common.pos_embed import sinusoid_encoding_table, FeedForward
from models.caption.containers import Module, ModuleList

class GeneratorLayer(Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__()
        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_ff, dropout)

class ParallelAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, activation='sigmoid', n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.activation = activation
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        self_att = self.self_att(x, x, x, mask_x)
        self_att = self_att * mask_pad
        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2) / np.sqrt(2)
        enc_att = enc_att * mask_pad
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class ConcatAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

    def forward(self, x, y, mask_pad, mask_x, mask_y):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att(out, y, y, mask_y) * mask_pad
        out = self.pwff(out) * mask_pad
        return out

class SequentialAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att1(out, y1, y1, mask_y1) * mask_pad
        out = self.vis_att2(out, y2, y2, mask_y2) * mask_pad
        ff = self.pwff(out)
        ff = ff * mask_pad
        return ff

class ParallelAttentionWithRelationLayer(ParallelAttentionLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, use_sqrt3_norm=False, gate_bias_init=-5.0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.vis_att3 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha3.bias, gate_bias_init)
        self.norm_divisor = np.sqrt(3) if use_sqrt3_norm else np.sqrt(2)

    def forward(self, x, y1, y2, y3, mask_pad, mask_x, mask_y1, mask_y2, mask_y3):
        self_att = self.self_att(x, x, x, mask_x) * mask_pad
        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad
        enc_att3 = self.vis_att3(self_att, y3, y3, mask_y3) * mask_pad
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))
        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / self.norm_divisor
        enc_att = enc_att * mask_pad
        ff = self.pwff(enc_att) * mask_pad
        return ff

class ParallelAttentionWithLocalCulturalLayer(ParallelAttentionWithRelationLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, use_sqrt3_norm=False, gate_bias_init=-5.0, use_eta_modulator=False, b_eta=-3.0, gate_bias_init_lcqm=-1.0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, use_sqrt3_norm=use_sqrt3_norm, gate_bias_init=gate_bias_init)
        self.vis_att4 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False)
        self.fc_alpha4 = nn.Linear(d_model + d_model, d_model)
        nn.init.xavier_uniform_(self.fc_alpha4.weight)
        nn.init.constant_(self.fc_alpha4.bias, gate_bias_init_lcqm)
        self.norm_divisor = np.sqrt(4) if use_sqrt3_norm else np.sqrt(3)
        self.register_buffer('gate_bias_offset', torch.tensor(0.0))
        self.use_eta_modulator = use_eta_modulator
        if use_eta_modulator:
            self.sqrt_d = d_model ** 0.5
            self.w_u = nn.Linear(4, 1)
            self.w_h = nn.Linear(d_model, 1)
            self.b_eta = nn.Parameter(torch.tensor(b_eta))

    def forward(self, x, y1, y2, y3, y4, mask_pad, mask_x, mask_y1, mask_y2, mask_y3, mask_y4):
        self_att = self.self_att(x, x, x, mask_x) * mask_pad
        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad
        enc_att3 = self.vis_att3(self_att, y3, y3, mask_y3) * mask_pad
        enc_att4_out, a_loc = self.vis_att4(self_att, y4, y4, mask_y4, return_attn=True)
        enc_att4 = enc_att4_out * mask_pad
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))
        alpha4 = torch.sigmoid(self.fc_alpha4(torch.cat([self_att, enc_att4], -1)) + self.gate_bias_offset)
        if self.use_eta_modulator:
            d_gr = (enc_att1 - enc_att2).norm(dim=-1, keepdim=True) / self.sqrt_d
            d_rr = (enc_att2 - enc_att3).norm(dim=-1, keepdim=True) / self.sqrt_d
            a_loc_mean = a_loc.mean(dim=1)
            h_a = -(a_loc_mean * a_loc_mean.clamp(min=1e-08).log()).sum(dim=-1, keepdim=True)
            m_a = a_loc_mean.amax(dim=-1, keepdim=True)
            u_t = torch.cat([d_gr, d_rr, h_a, m_a], dim=-1)
            eta_loc = torch.sigmoid(self.w_u(u_t) + self.w_h(self_att) + self.b_eta)
            enc_att4 = enc_att4 * eta_loc
        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3 + enc_att4 * alpha4) / self.norm_divisor
        enc_att = enc_att * mask_pad
        ff = self.pwff(enc_att) * mask_pad
        if self.use_eta_modulator:
            return (ff, {'eta_loc': eta_loc, 'g_loc': alpha4, 'alpha4': alpha4})
        return (ff, {'alpha4': alpha4})

class CaptionGenerator(Module):
    GENERATOR_LAYER = {'concat': ConcatAttentionLayer, 'parallel': ParallelAttentionLayer, 'sequential': SequentialAttentionLayer}

    def __init__(self, vocab_size, max_len, n_layers, pad_idx, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, decoder_name='parallel', cfg=None, use_relation_branch=False, use_sqrt3_norm=False, use_local_cultural_branch=False, use_eta_modulator=False, b_eta=-3.0, gate_bias_init_lcqm=-1.0):
        super().__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.cfg = cfg
        self.decoder_name = decoder_name
        self.use_relation_branch = use_relation_branch and decoder_name == 'parallel'
        self.use_local_cultural_branch = use_local_cultural_branch and decoder_name == 'parallel'
        gate_bias_init = float(getattr(cfg, 'gate_bias_init', -5.0)) if cfg else -5.0
        if self.use_local_cultural_branch:
            self.layers = ModuleList([ParallelAttentionWithLocalCulturalLayer(d_model, n_heads, d_ff, dropout, use_sqrt3_norm=use_sqrt3_norm, gate_bias_init=gate_bias_init, use_eta_modulator=use_eta_modulator, b_eta=b_eta, gate_bias_init_lcqm=gate_bias_init_lcqm) for _ in range(n_layers)])
        elif self.use_relation_branch:
            self.layers = ModuleList([ParallelAttentionWithRelationLayer(d_model, n_heads, d_ff, dropout, use_sqrt3_norm=use_sqrt3_norm, gate_bias_init=gate_bias_init) for _ in range(n_layers)])
        else:
            generator_layer = self.GENERATOR_LAYER[self.decoder_name]
            self.layers = ModuleList([generator_layer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers
        self.register_state('running_mask_x', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def get_seq_inputs(self, input):
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_x = mask_x.gt(0)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        x = self.word_emb(input) + self.pos_emb(seq)
        return (x, mask_x, mask_pad)

    def forward(self, input, vis_inputs, capture_all_layers=False):
        x, mask_x, mask_pad = self.get_seq_inputs(input)
        if self.decoder_name == 'concat':
            y = torch.cat([vis_inputs['gri_feat'], vis_inputs['reg_feat']], dim=1)
            mask_y = torch.cat([vis_inputs['gri_mask'], vis_inputs['reg_mask']], dim=3)
            for layer in self.layers:
                x = layer(x, y, mask_pad, mask_x, mask_y)
        if self.decoder_name == 'sequential':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']
            for layer in self.layers:
                x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        if self.decoder_name == 'parallel':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']
            rel_feat = vis_inputs.get('rel_feat')
            loc_feat = vis_inputs.get('loc_feat')
            if self.use_local_cultural_branch:
                if rel_feat is None or loc_feat is None:
                    raise RuntimeError('LCQM branch is enabled but vis_inputs is missing rel_feat or loc_feat.')
                mask_y3 = vis_inputs.get('rel_mask', vis_inputs['reg_mask'])
                mask_y4 = vis_inputs.get('loc_mask')
                layer_diag = {}
                all_layer_diags = []
                for layer_idx, layer in enumerate(self.layers):
                    result = layer(x, y1, y2, rel_feat, loc_feat, mask_pad, mask_x, mask_y1, mask_y2, mask_y3, mask_y4)
                    if isinstance(result, tuple):
                        x, diag = result
                        diag['layer_idx'] = layer_idx
                        layer_diag = diag
                        all_layer_diags.append(diag)
                    else:
                        x = result
            elif self.use_relation_branch:
                if rel_feat is None:
                    raise RuntimeError('RRM relation branch is enabled but vis_inputs is missing rel_feat.')
                mask_y3 = vis_inputs.get('rel_mask', vis_inputs['reg_mask'])
                for layer in self.layers:
                    x = layer(x, y1, y2, rel_feat, mask_pad, mask_x, mask_y1, mask_y2, mask_y3)
            else:
                for layer in self.layers:
                    x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        x = self.fc(x)
        log_probs = F.log_softmax(x, dim=-1)
        if self.use_local_cultural_branch:
            if capture_all_layers and all_layer_diags:
                return (log_probs, layer_diag, all_layer_diags)
            elif layer_diag:
                return (log_probs, layer_diag)
        return log_probs
