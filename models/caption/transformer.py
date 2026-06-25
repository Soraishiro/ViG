import torch
from torch import nn
from einops import rearrange, repeat
from engine.utils import NestedTensor
from models.common.attention import MemoryAttention
from models.caption.base import BaseCaptioner
from models.caption.grid_net import GridFeatureNetwork
from models.caption.cap_generator import CaptionGenerator

class Transformer(BaseCaptioner):

    def __init__(self, detector, config=None):
        super(Transformer, self).__init__()
        self.grid_net = GridFeatureNetwork(n_layers=config.model.grid_net.n_layers, d_in=config.model.grid_feat_dim, dropout=config.model.dropout, n_memories=getattr(config.model.grid_net, 'n_memories', 0))
        _rrm_enabled = hasattr(config, 'model_vig') and hasattr(config.model_vig, 'rrm') and config.model_vig.rrm.enabled
        _use_sqrt3 = getattr(config.model_vig.rrm, 'use_sqrt3_norm', False) if _rrm_enabled else False
        _lcqm_enabled = hasattr(config, 'model_vig') and hasattr(config.model_vig, 'lcqm') and config.model_vig.lcqm.enabled
        self.cap_generator = CaptionGenerator(n_layers=config.model.cap_generator.n_layers, vocab_size=config.model.vocab_size, max_len=config.model.max_len, pad_idx=config.model.pad_idx, dropout=config.model.dropout, cfg=config.model.cap_generator, use_relation_branch=_rrm_enabled, use_sqrt3_norm=_use_sqrt3, use_local_cultural_branch=_lcqm_enabled, use_eta_modulator=_lcqm_enabled and getattr(config.model_vig.lcqm, 'modulator', None) is not None and getattr(config.model_vig.lcqm.modulator, 'enabled', False), b_eta=float(getattr(getattr(config.model_vig.lcqm, 'modulator', None), 'b_eta', -3.0) if _lcqm_enabled else -3.0), gate_bias_init_lcqm=float(getattr(config.model_vig.lcqm, 'gate_bias_init', -1.0) if _lcqm_enabled else -1.0))
        self.config = config
        self.bos_idx = config.model.bos_idx
        self.use_reg_feat = config.model.use_reg_feat
        self.use_gri_feat = config.model.use_gri_feat
        self.cached_features = False
        self._capture_diagnostics = False
        self._diag_buffer = {}
        if _rrm_enabled and (not self.use_reg_feat):
            raise RuntimeError('model_vig.rrm.enabled=true requires model.use_reg_feat=true.')
        if _rrm_enabled and config.model.cap_generator.decoder_name != 'parallel':
            raise RuntimeError("model_vig.rrm.enabled=true is only supported with cap_generator.decoder_name='parallel'.")
        if self.use_gri_feat:
            self.register_state('gri_feat', None)
            self.register_state('gri_mask', None)
        if self.use_reg_feat:
            self.register_state('reg_feat', None)
            self.register_state('reg_mask', None)
            self.register_state('reg_boxes', None)
        if _rrm_enabled:
            self.register_state('rel_feat', None)
            self.register_state('rel_mask', None)
        self.init_weights()
        if _rrm_enabled:
            from models.caption.relation_memory import RelationMemory
            self.rrm = RelationMemory(config)
            self.rrm._init_vig_weights()
        else:
            self.rrm = None
        if _lcqm_enabled:
            from models.caption.cultural_memory import CulturalQueryMemory
            self.lcqm = CulturalQueryMemory(config.model_vig.lcqm)
            self.lcqm._init_vig_weights()
            self.register_state('loc_feat', None)
            self.register_state('loc_mask', None)
        else:
            self.lcqm = None
        self.detector = detector

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, use_beam_search=False, max_len=20, eos_idx=3, beam_size=5, out_size=1, return_probs=False, return_aux=False, **kwargs):
        if not use_beam_search:
            if not self.cached_features:
                vis_inputs = self.detector(images)
            else:
                vis_inputs = images
            if self.config.model.use_gri_feat:
                gri_feat, _ = self.grid_net(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                vis_inputs['gri_feat'] = gri_feat[:, -1]
            self._add_relation_features(vis_inputs, context='non-beam forward')
            self._add_local_cultural_features(vis_inputs, context='non-beam forward', return_attn=self._capture_diagnostics)
            dec_output = self.cap_generator(seq, vis_inputs, capture_all_layers=self._capture_diagnostics)
            if return_aux:
                aux = {}
                if self.lcqm is not None:
                    aux['loc_feat'] = vis_inputs.get('loc_feat')
                    aux['loc_feat_proj'] = vis_inputs.get('loc_feat_proj')
                    aux['p_attn_lcqm'] = vis_inputs.get('p_attn_lcqm')
                if self.rrm is not None and hasattr(self.rrm, 'alpha_rel'):
                    aux['alpha_rel'] = self.rrm.alpha_rel.detach().item()
                if isinstance(dec_output, tuple):
                    if len(dec_output) == 3:
                        log_probs, layer_diag, all_layers = dec_output
                        aux.update(layer_diag)
                        aux['all_layer_diags'] = all_layers
                    else:
                        log_probs, layer_diag = dec_output
                        aux.update(layer_diag)
                    return (log_probs, aux)
                return (dec_output, aux)
            if isinstance(dec_output, tuple):
                return dec_output[0]
            return dec_output
        else:
            batch_size, device = self.get_bs_device(images)
            self.seq_mask = torch.ones((batch_size, beam_size, 1), device=device)
            self.seq_logprob = torch.zeros((batch_size, 1, 1), device=device)
            self.log_probs = []
            self.selected_words = None
            if return_probs:
                self.all_log_probs = []
            outputs = []
            with self.statefulness(batch_size):
                for timestep in range(max_len):
                    images, outputs = self.iter(timestep=timestep, samples=images, outputs=outputs, return_probs=return_probs, batch_size=batch_size, beam_size=beam_size, eos_idx=eos_idx, **kwargs)
            seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
            outputs = torch.cat(outputs, -1)
            outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            log_probs = torch.cat(self.log_probs, -1)
            log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            if return_probs:
                all_log_probs = torch.cat(self.all_log_probs, 2)
                all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, max_len, all_log_probs.shape[-1]))
            outputs = outputs.contiguous()[:, :out_size]
            log_probs = log_probs.contiguous()[:, :out_size]
            if out_size == 1:
                outputs = outputs.squeeze(1)
                log_probs = log_probs.squeeze(1)
            if return_probs:
                return (outputs, log_probs, all_log_probs)
            else:
                return (outputs, log_probs)

    def step(self, timestep, prev_output, samples, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if timestep == 0:
                if not self.cached_features:
                    vis_inputs = self.detector(samples)
                else:
                    vis_inputs = samples
                if self.config.model.use_gri_feat:
                    self.gri_feat, self.gri_mask = self.grid_net(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                    self.gri_feat = self.gri_feat[:, -1]
                if self.config.model.use_reg_feat:
                    self.reg_feat = vis_inputs['reg_feat']
                    self.reg_mask = vis_inputs['reg_mask']
                    self.reg_boxes = self._require_reg_boxes(vis_inputs, context='beam step t=0')
                    if self.rrm is not None:
                        self.rel_feat = self.rrm(self.reg_feat, self.reg_boxes)
                        self.rel_mask = self.reg_mask
                    if self.lcqm is not None:
                        self.loc_feat, self.loc_mask = self._compute_lcqm_beam()
                _feat = getattr(self, 'gri_feat', self.reg_feat)
                it = _feat.data.new_full((_feat.shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
        vis_inputs = {}
        if self.config.model.use_gri_feat:
            vis_inputs['gri_feat'] = self.gri_feat
            vis_inputs['gri_mask'] = self.gri_mask
        if self.config.model.use_reg_feat:
            vis_inputs['reg_feat'] = self.reg_feat
            vis_inputs['reg_mask'] = self.reg_mask
            if self.reg_boxes is not None:
                vis_inputs['reg_boxes'] = self.reg_boxes
        if self.rrm is not None:
            vis_inputs['rel_feat'] = self.rel_feat
            vis_inputs['rel_mask'] = self.rel_mask
        if self.lcqm is not None:
            vis_inputs['loc_feat'] = self.loc_feat
            vis_inputs['loc_mask'] = self.loc_mask
            if self.rrm is None:
                vis_inputs['rel_feat'] = self.reg_feat
                vis_inputs['rel_mask'] = self.reg_mask
        result = self.cap_generator(it, vis_inputs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def _require_reg_boxes(self, vis_inputs, context):
        if self.rrm is None:
            return vis_inputs.get('reg_boxes') if isinstance(vis_inputs, dict) else None
        if 'reg_boxes' not in vis_inputs or vis_inputs['reg_boxes'] is None:
            raise RuntimeError(f"model_vig.rrm.enabled=true requires vis_inputs['reg_boxes']; missing in {context}. Cached-feature runs must include reg_boxes.")
        return vis_inputs['reg_boxes']

    def _add_relation_features(self, vis_inputs, context):
        if self.rrm is None:
            return
        reg_boxes = self._require_reg_boxes(vis_inputs, context=context)
        vis_inputs['rel_feat'] = self.rrm(vis_inputs['reg_feat'], reg_boxes)
        vis_inputs['rel_mask'] = vis_inputs['reg_mask']

    def _add_local_cultural_features(self, vis_inputs, context, return_attn=False):
        if self.lcqm is None:
            return
        if 'rel_feat' not in vis_inputs:
            vis_inputs['rel_feat'] = vis_inputs['reg_feat']
            vis_inputs['rel_mask'] = vis_inputs['reg_mask']
        result = self._compute_lcqm_inner(vis_inputs, return_attn=return_attn)
        if return_attn:
            region_in, loc_feat, loc_proj, p_attn_lcqm = result
            vis_inputs['p_attn_lcqm'] = p_attn_lcqm
        else:
            region_in, loc_feat, loc_proj = result
        vis_inputs['loc_feat_proj'] = loc_proj
        vis_inputs['loc_feat'] = loc_feat
        B, K, _ = loc_feat.shape
        vis_inputs['loc_mask'] = loc_feat.new_zeros(B, 1, 1, K).bool()

    def _compute_lcqm_inner(self, vis_inputs, return_attn=False):
        cfg = self.config.model_vig.lcqm
        use_rel = getattr(cfg, 'use_rel_input', False)
        region_in = vis_inputs.get('rel_feat') if use_rel and 'rel_feat' in vis_inputs else vis_inputs['reg_feat']
        if use_rel and getattr(cfg, 'stop_grad_rel', False):
            region_in = region_in.detach()
        result = self.lcqm(region_in, vis_inputs['gri_feat'], vis_inputs['gri_mask'], return_attn=return_attn)
        if return_attn:
            loc_feat, loc_proj, p_attn_lcqm = result
            return (region_in, loc_feat, loc_proj, p_attn_lcqm)
        loc_feat, loc_proj = result
        return (region_in, loc_feat, loc_proj)

    def _compute_lcqm_beam(self, return_attn=False):
        vis_inputs = {'reg_feat': self.reg_feat, 'gri_feat': self.gri_feat, 'gri_mask': self.gri_mask}
        if self.rrm is not None:
            vis_inputs['rel_feat'] = self.rel_feat
        result = self._compute_lcqm_inner(vis_inputs, return_attn=return_attn)
        if return_attn:
            region_in, loc_feat, loc_proj, p_attn_lcqm = result
        else:
            region_in, loc_feat, loc_proj = result
        B, K, _ = loc_feat.shape
        loc_mask = loc_feat.new_zeros(B, 1, 1, K).bool()
        if return_attn:
            return (loc_feat, loc_mask, loc_proj, p_attn_lcqm)
        return (loc_feat, loc_mask)

    def enable_diagnostics(self):
        self._capture_diagnostics = True
        self._diag_buffer.clear()

    def disable_diagnostics(self):
        self._capture_diagnostics = False

    def get_diagnostics(self):
        return dict(self._diag_buffer)

    def get_bs_device(self, samples):
        if isinstance(samples, dict):
            key = 'gri_feat' if 'gri_feat' in samples else 'reg_feat'
            batch_size = samples[key].shape[0]
            device = samples[key].device
        elif isinstance(samples, NestedTensor):
            batch_size = samples.tensors.shape[0]
            device = samples.tensors.device
        return (batch_size, device)

    def init_state(self, batch_size, device):
        return [torch.zeros((batch_size, 0), dtype=torch.long, device=device), None, None]

    def select(self, t, candidate_logprob, beam_size, **kwargs):
        candidate_logprob = rearrange(candidate_logprob, 'B Beam V -> B (Beam V)')
        selected_logprob, selected_idx = torch.sort(candidate_logprob, -1, descending=True)
        selected_logprob, selected_idx = (selected_logprob[:, :beam_size], selected_idx[:, :beam_size])
        return (selected_idx, selected_logprob)

    def _expand_state(self, selected_beam, cur_beam_size, batch_size, beam_size):

        def fn(tensor):
            shape = [int(sh) for sh in tensor.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            tensor = torch.gather(tensor.view(*[batch_size, cur_beam_size] + shape[1:]), 1, beam.expand(*[batch_size, beam_size] + shape[1:]))
            tensor = tensor.view(*[-1] + shape[1:])
            return tensor
        return fn

    def iter(self, timestep, samples, outputs, return_probs, batch_size, beam_size=5, eos_idx=3, **kwargs):
        cur_beam_size = 1 if timestep == 0 else beam_size
        word_logprob = self.step(timestep, self.selected_words, samples, None, mode='feedback', **kwargs)
        if isinstance(word_logprob, tuple):
            word_logprob = word_logprob[0]
        word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob
        if timestep > 0:
            _selected_words = self.selected_words.view(batch_size, cur_beam_size)
            mask = repeat((_selected_words != eos_idx).float(), 'B Beam -> B Beam V', V=1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)
        selected_idx, selected_logprob = self.select(timestep, candidate_logprob, beam_size, **kwargs)
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        self.apply_to_states(self._expand_state(selected_beam, cur_beam_size, batch_size, beam_size))
        self.seq_logprob = repeat(selected_logprob, 'B Beam -> B Beam L', L=1)
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.seq_mask = torch.gather(self.seq_mask, 1, beam_exp)
        outputs = [torch.gather(o, 1, beam_exp) for o in outputs]
        outputs.append(repeat(selected_words, 'B Beam -> B Beam L', L=1))
        if return_probs:
            if timestep == 0:
                self.all_log_probs.append(word_logprob.expand((batch_size, beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob.shape[-1])
        this_word_logprob = torch.gather(word_logprob, 1, beam_exp)
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.log_probs = [torch.gather(o, 1, beam_exp) for o in self.log_probs]
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)
        return (samples, outputs)
