from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainingRegime:
    lane_id: str
    init_mode: str
    checkpoint_kind: str
    phase_plan: str
    freeze_scope: str
    selection_split: str
    required_checkpoint_field: str
    forbidden_checkpoint_field: str
    xe_epochs: int = 10
    scst_epochs: int = 0
    freezing_xe_epochs: int = 0
    xe_lr: str = '5e-6'
    xe_detector_lr: str = '5e-6'
    sc_lr: str = '5e-6'
    sc_detector_lr: str = '5e-6'
    beam_size: int = 5
    beam_len: int = 20

    def _freeze_scope_overrides(self) -> list[str]:
        if self.freeze_scope == 'detector':
            return ['optimizer.freeze_detector=true', 'optimizer.freeze_backbone=false', 'optimizer.freeze_grid=false']
        if self.freeze_scope == 'backbone':
            return ['optimizer.freeze_detector=false', 'optimizer.freeze_backbone=true', 'optimizer.freeze_grid=false']
        if self.freeze_scope == 'detector_grid':
            return ['optimizer.freeze_detector=true', 'optimizer.freeze_backbone=false', 'optimizer.freeze_grid=true']
        return ['optimizer.freeze_detector=false', 'optimizer.freeze_backbone=false', 'optimizer.freeze_grid=false']

    def _checkpoint_clear_overrides(self) -> list[str]:
        if self.init_mode == 'detector_only':
            return ['exp.checkpoint=']
        if self.init_mode == 'full_model':
            return ['model.detector.checkpoint=']
        if self.init_mode == 'none':
            return ['exp.checkpoint=', 'model.detector.checkpoint=']
        return []

    def to_overrides(self) -> list[str]:
        overrides = [f'exp.training_regime={self.lane_id}', f'exp.init_mode={self.init_mode}', f'exp.phase_plan={self.phase_plan}', f'exp.selection_split={self.selection_split}', f'optimizer.finetune_xe_epochs={self.xe_epochs}', f'optimizer.finetune_sc_epochs={self.scst_epochs}', f'optimizer.freezing_xe_epochs={self.freezing_xe_epochs}', f'optimizer.xe_lr={self.xe_lr}', f'optimizer.xe_detector_lr={self.xe_detector_lr}', f'optimizer.sc_lr={self.sc_lr}', f'optimizer.sc_detector_lr={self.sc_detector_lr}', f'model.beam_size={self.beam_size}', f'model.beam_len={self.beam_len}']
        overrides += self._freeze_scope_overrides()
        overrides += self._checkpoint_clear_overrides()
        return overrides

    def to_contract_overrides(self) -> list[str]:
        overrides = [f'exp.training_regime={self.lane_id}', f'exp.init_mode={self.init_mode}', f'exp.phase_plan={self.phase_plan}', f'exp.selection_split={self.selection_split}']
        overrides += self._freeze_scope_overrides()
        overrides += self._checkpoint_clear_overrides()
        return overrides
REGIMES = {'full_xe': TrainingRegime(lane_id='full_xe', init_mode='full_model', checkpoint_kind='full_grit_caption_checkpoint', phase_plan='xe_only', freeze_scope='none', selection_split='valid', required_checkpoint_field='exp.checkpoint', forbidden_checkpoint_field='model.detector.checkpoint'), 'full_freeze_then_xe_backbone': TrainingRegime(lane_id='full_freeze_then_xe_backbone', init_mode='full_model', checkpoint_kind='full_grit_caption_checkpoint', phase_plan='freeze_then_xe', freeze_scope='backbone', selection_split='valid', required_checkpoint_field='exp.checkpoint', forbidden_checkpoint_field='model.detector.checkpoint', xe_epochs=10, freezing_xe_epochs=5), 'full_freeze_then_xe_detector': TrainingRegime(lane_id='full_freeze_then_xe_detector', init_mode='full_model', checkpoint_kind='full_grit_caption_checkpoint', phase_plan='freeze_then_xe', freeze_scope='detector', selection_split='valid', required_checkpoint_field='exp.checkpoint', forbidden_checkpoint_field='model.detector.checkpoint', xe_epochs=10, freezing_xe_epochs=5), 'full_frozen_visual_xe': TrainingRegime(lane_id='full_frozen_visual_xe', init_mode='full_model', checkpoint_kind='full_grit_caption_checkpoint', phase_plan='frozen_visual_xe', freeze_scope='detector_grid', selection_split='valid', required_checkpoint_field='exp.checkpoint', forbidden_checkpoint_field='model.detector.checkpoint', xe_epochs=15), 'detector_stage2_xe': TrainingRegime(lane_id='detector_stage2_xe', init_mode='detector_only', checkpoint_kind='stage1_detector_checkpoint', phase_plan='xe_only', freeze_scope='none', selection_split='valid', xe_epochs=15, required_checkpoint_field='model.detector.checkpoint', forbidden_checkpoint_field='exp.checkpoint'), 'detector_frozen_visual_xe': TrainingRegime(lane_id='detector_frozen_visual_xe', init_mode='detector_only', checkpoint_kind='stage1_detector_checkpoint', phase_plan='frozen_visual_xe', freeze_scope='detector', selection_split='valid', xe_epochs=15, required_checkpoint_field='model.detector.checkpoint', forbidden_checkpoint_field='exp.checkpoint'), 'detector_partial_visual_xe': TrainingRegime(lane_id='detector_partial_visual_xe', init_mode='detector_only', checkpoint_kind='stage1_detector_checkpoint', phase_plan='partial_visual_xe', freeze_scope='backbone', selection_split='valid', required_checkpoint_field='model.detector.checkpoint', forbidden_checkpoint_field='exp.checkpoint'), 'scst_from_xe': TrainingRegime(lane_id='scst_from_xe', init_mode='full_model', checkpoint_kind='full_grit_caption_checkpoint_or_recovery_seed', phase_plan='scst_only', freeze_scope='none', selection_split='valid', required_checkpoint_field='exp.scst_seed_checkpoint', forbidden_checkpoint_field='model.detector.checkpoint', xe_epochs=0, scst_epochs=3, sc_lr='2e-6', sc_detector_lr='2e-6')}
PROFILE_TO_REGIME = {'xe_seed_baseline': 'full_xe', 'xe_seed_vg_anchor': 'full_xe', 'vg_anchor_low_lr': 'full_xe', 'vg_anchor_asym_backbone': 'full_xe', 'vg_anchor_backbone_high': 'full_xe', 'vg_warmup_real': 'full_xe', 'vg_detector_low': 'full_xe', 'vg_warmup_detector': 'full_xe', 'vg_cosine_10ep': 'full_xe', 'vg_flat_15ep': 'full_xe', 'vg_cosine_15ep': 'full_xe', 'vg_cosine_15ep_min2e6': 'full_xe', 'vg_flat_25ep': 'full_xe', 'xe_seed_xe_lr_half': 'full_xe', 'xe_seed_backbone_quarter': 'full_xe', 'xe_seed_batch8': 'full_xe', 'freeze_then_xe_backbone': 'full_freeze_then_xe_backbone', 'freeze_then_xe_detector': 'full_freeze_then_xe_detector', 'freeze_visual_xe': 'full_frozen_visual_xe', 'frozen_stage3': 'full_xe', 'detector_stage2_xe': 'detector_stage2_xe', 'detector_frozen_visual_xe': 'detector_frozen_visual_xe', 'detector_partial_visual_xe': 'detector_partial_visual_xe', 'scst_cider': 'scst_from_xe', 'scst_asym': 'scst_from_xe', 'scst_lowlr': 'scst_from_xe', 'scst_short': 'scst_from_xe'}

def get_regime(lane_id: str) -> TrainingRegime:
    try:
        return REGIMES[lane_id]
    except KeyError as exc:
        raise ValueError(f'Unsupported training regime: {lane_id}') from exc

def regime_for_profile(profile: str) -> TrainingRegime:
    try:
        lane_id = PROFILE_TO_REGIME[profile]
    except KeyError as exc:
        raise ValueError(f'No training regime mapping for profile: {profile}') from exc
    return get_regime(lane_id)

def infer_regime_id(config) -> str:
    explicit = str(getattr(config.exp, 'training_regime', '') or '').strip()
    if explicit:
        return explicit
    init_mode = str(getattr(config.exp, 'init_mode', 'full_model') or 'full_model')
    phase_plan = str(getattr(config.exp, 'phase_plan', 'xe_only') or 'xe_only')
    freeze_detector = _as_bool(getattr(config.optimizer, 'freeze_detector', False))
    freeze_backbone = _as_bool(getattr(config.optimizer, 'freeze_backbone', False))
    if phase_plan == 'scst_only':
        return 'scst_from_xe'
    if init_mode == 'detector_only' and phase_plan == 'frozen_visual_xe':
        return 'detector_frozen_visual_xe'
    if init_mode == 'detector_only' and phase_plan == 'partial_visual_xe':
        return 'detector_partial_visual_xe'
    if init_mode == 'detector_only':
        return 'detector_stage2_xe'
    if phase_plan == 'frozen_visual_xe':
        return 'full_frozen_visual_xe'
    if phase_plan == 'freeze_then_xe' and freeze_detector:
        return 'full_freeze_then_xe_detector'
    if phase_plan == 'freeze_then_xe' and freeze_backbone:
        return 'full_freeze_then_xe_backbone'
    return 'full_xe'

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)

def validate_regime_consistency(config, regime: TrainingRegime) -> None:
    init_mode = str(getattr(config.exp, 'init_mode', '') or '')
    phase_plan = str(getattr(config.exp, 'phase_plan', '') or '')
    freeze_detector = _as_bool(getattr(config.optimizer, 'freeze_detector', False))
    freeze_backbone = _as_bool(getattr(config.optimizer, 'freeze_backbone', False))
    errors = []
    if init_mode != regime.init_mode:
        errors.append(f'init_mode={init_mode} does not match regime {regime.lane_id} init_mode={regime.init_mode}')
    if phase_plan != regime.phase_plan:
        errors.append(f'phase_plan={phase_plan} does not match regime {regime.lane_id} phase_plan={regime.phase_plan}')
    expected_freeze_detector = regime.freeze_scope in ('detector', 'detector_grid')
    expected_freeze_backbone = regime.freeze_scope == 'backbone'
    expected_freeze_grid = regime.freeze_scope == 'detector_grid'
    if freeze_detector != expected_freeze_detector:
        errors.append(f'freeze_detector={freeze_detector} does not match regime {regime.lane_id} freeze_detector={expected_freeze_detector}')
    if freeze_backbone != expected_freeze_backbone:
        errors.append(f'freeze_backbone={freeze_backbone} does not match regime {regime.lane_id} freeze_backbone={expected_freeze_backbone}')
    freeze_grid = _as_bool(getattr(config.optimizer, 'freeze_grid', False))
    if freeze_grid != expected_freeze_grid:
        errors.append(f'freeze_grid={freeze_grid} does not match regime {regime.lane_id} freeze_grid={expected_freeze_grid}')
    if errors:
        raise RuntimeError('Training regime contract mismatch: ' + '; '.join(errors))

def build_training_regime_contract(config, init_contract: dict | None=None) -> dict:
    lane_id = infer_regime_id(config)
    regime = get_regime(lane_id)
    validate_regime_consistency(config, regime)
    init_contract = dict(init_contract or {})
    return {**asdict(regime), 'resolved_lane_id': lane_id, 'configured_training_regime': str(getattr(config.exp, 'training_regime', '') or ''), 'configured_init_mode': str(getattr(config.exp, 'init_mode', '')), 'configured_phase_plan': str(getattr(config.exp, 'phase_plan', '')), 'full_checkpoint_path': init_contract.get('full_checkpoint_path'), 'detector_checkpoint_path': init_contract.get('detector_checkpoint_path'), 'backbone_pretrained_source': init_contract.get('backbone_pretrained_source')}
_ABLATION_VG_PROFILES = frozenset({'freeze_then_xe_backbone', 'freeze_then_xe_detector', 'freeze_visual_xe'})

def checkpoint_for_regime(regime: TrainingRegime, checkpoint_root: Path, profile: str) -> str:
    if regime.init_mode == 'detector_only':
        return str(checkpoint_root / 'detector_checkpoint_vg.pth')
    if profile.startswith('vg_') or profile == 'xe_seed_vg_anchor' or profile in _ABLATION_VG_PROFILES:
        return str(checkpoint_root / 'grit_checkpoint_vg.pth')
    return str(checkpoint_root / 'grit_checkpoint_4ds.pth')

def _cmd_launcher_overrides(args) -> int:
    checkpoint_root = Path(args.checkpoint_root)
    regime = regime_for_profile(args.profile)
    print(f'CHECKPOINT_PATH\t{checkpoint_for_regime(regime, checkpoint_root, args.profile)}')
    overrides = regime.to_overrides() if args.full_defaults else regime.to_contract_overrides()
    for override in overrides:
        print(f'ARG\t{override}')
    if regime.init_mode == 'detector_only':
        print(f'ARG\tmodel.detector.checkpoint={checkpoint_for_regime(regime, checkpoint_root, args.profile)}')
    return 0

def _cmd_describe(args) -> int:
    regime = get_regime(args.regime)
    for key, value in asdict(regime).items():
        print(f'{key}\t{value}')
    return 0

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='GRIT/KTVIC training regime registry.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    launcher = subparsers.add_parser('launcher-overrides')
    launcher.add_argument('--profile', required=True)
    launcher.add_argument('--checkpoint-root', required=True)
    launcher.add_argument('--full-defaults', action='store_true')
    launcher.set_defaults(func=_cmd_launcher_overrides)
    describe = subparsers.add_parser('describe')
    describe.add_argument('--regime', required=True)
    describe.set_defaults(func=_cmd_describe)
    args = parser.parse_args(argv)
    return args.func(args)
if __name__ == '__main__':
    raise SystemExit(main())
