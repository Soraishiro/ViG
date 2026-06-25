import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from engine.utils import nested_tensor_from_tensor_list
from models.caption import Transformer
from models.caption.detector import build_detector
from train import CONTENT_STOP_TOKENS, ExperimentTracker, RunObserver, collect_row_fieldnames, collect_system_info, compute_caption_quality_metrics, contains_vietnamese_diacritic, decode_prediction, extract_model_state_dict, filter_content_tokens, get_dataloaders, iso_timestamp, load_trusted_checkpoint, mean, resolve_vocab_from_dataset, select_best_reference_tokens, stddev, tokenize_caption, write_csv, write_json
from datasets.caption import metrics as metrics

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)

def write_result_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def parse_args():
    parser = argparse.ArgumentParser(description='Run standalone official-test inference with rich per-sample analysis artifacts.')
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--source-train-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--tracking-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--checkpoint-preference', default='best', choices=['auto', 'best', 'final'])
    parser.add_argument('--run-name-suffix', default='official-test-rich')
    parser.add_argument('--max-test-items', type=int, default=0)
    parser.add_argument('--metrics-scope', default='full', choices=['full', 'light'])
    parser.add_argument('--comparison-details-json', default='')
    parser.add_argument('--comparison-label', default='comparison')
    return parser.parse_args()

def pick_checkpoint(source_train_dir: Path, manifest: dict, preference: str):
    artifacts = manifest.get('artifacts', {})
    candidates = []

    def add_candidate(name, relpath):
        if not relpath:
            return
        candidates.append((name, source_train_dir / relpath))
    add_candidate('best', artifacts.get('best_valid_checkpoint'))
    add_candidate('final', artifacts.get('final_checkpoint'))
    for fallback in ('checkpoints/checkpoint_best_valid_xe.pth', 'checkpoints/checkpoint_best_valid_scst.pth', 'model_stage1_xe_final.pth', 'model_stage2_scst_final.pth', 'model_epoch10.pth', 'model_epoch12.pth', 'model_epoch13.pth', 'model_epoch14.pth', 'model_epoch15.pth', 'model_epoch20.pth', 'model_epoch25.pth'):
        candidates.append(('fallback', source_train_dir / fallback))
    if preference == 'best':
        ordered = [item for item in candidates if item[0] == 'best'] + [item for item in candidates if item[0] != 'best']
    elif preference == 'final':
        ordered = [item for item in candidates if item[0] == 'final'] + [item for item in candidates if item[0] != 'final']
    else:
        ordered = candidates
    for candidate_type, path in ordered:
        if path.exists():
            return (candidate_type, path)
    raise FileNotFoundError(f'No usable checkpoint found for source run: {source_train_dir}')

def load_source_contract(source_train_dir: Path):
    source_run_root = source_train_dir.parent
    preflight_path = source_run_root / 'preflight' / 'trace' / 'preflight_report.json'
    if preflight_path.exists():
        preflight_report = load_json(preflight_path)
        return {'preflight_path': str(preflight_path), 'resolved_paths': preflight_report.get('resolved_paths', {}), 'caption_field': preflight_report.get('caption_field'), 'raw_caption_field': preflight_report.get('raw_caption_field'), 'tokenizer_backend': preflight_report.get('tokenizer_backend'), 'vocab_size': preflight_report.get('vocab_size')}
    shared_vocab = source_run_root / 'shared_cache' / 'vi_captions_train_only.json'
    return {'preflight_path': None, 'resolved_paths': {'vocab_source': str(shared_vocab)}, 'caption_field': 'segment_caption', 'raw_caption_field': 'caption', 'tokenizer_backend': 'rdrsegmenter_segment_caption', 'vocab_size': None}

def configure_dataset_environment(source_contract: dict, dataset_root: Path):
    resolved_paths = dict(source_contract.get('resolved_paths', {}))
    train_images = resolved_paths.get('train_images') or str(dataset_root / 'train-images' / 'train-images')
    train_json = resolved_paths.get('train_json') or str(dataset_root / 'train_data.json')
    vocab_source = resolved_paths.get('vocab_source') or str(dataset_root / 'vi_captions_train_only.json')
    test_images = str(dataset_root / 'public-test-images' / 'public-test-images')
    test_json = str(dataset_root / 'test_data.json')
    os.environ['KTVIC_ROOT'] = str(dataset_root)
    os.environ['DATA_ROOT'] = str(dataset_root)
    os.environ['KTVIC_TRAIN_IMAGES'] = train_images
    os.environ['KTVIC_VALID_IMAGES'] = test_images
    os.environ['KTVIC_TEST_IMAGES'] = test_images
    os.environ['KTVIC_TRAIN_JSON'] = train_json
    os.environ['KTVIC_VALID_JSON'] = test_json
    os.environ['KTVIC_TEST_JSON'] = test_json
    os.environ['KTVIC_VOCAB_SOURCE_JSON'] = vocab_source
    os.environ['KTVIC_EVAL_SPLIT_NAME'] = 'test'
    caption_field = source_contract.get('caption_field') or 'segment_caption'
    raw_caption_field = source_contract.get('raw_caption_field') or 'caption'
    tokenizer_backend = source_contract.get('tokenizer_backend') or 'rdrsegmenter_segment_caption'
    os.environ['KTVIC_CAPTION_FIELD'] = str(caption_field)
    os.environ['KTVIC_RAW_CAPTION_FIELD'] = str(raw_caption_field)
    os.environ['KTVIC_TOKENIZER_BACKEND'] = str(tokenizer_backend)
    return {'train_images': train_images, 'valid_images': test_images, 'test_images': test_images, 'train_json': train_json, 'valid_json': test_json, 'test_json': test_json, 'vocab_source': vocab_source, 'caption_field': caption_field, 'raw_caption_field': raw_caption_field, 'tokenizer_backend': tokenizer_backend}

def build_eval_config(source_config: dict, checkpoint_path: Path, run_dir: Path, tracking_mode: str, run_name_suffix: str):
    config = OmegaConf.create(source_config)
    config.exp.eval = True
    config.exp.eval_skip_valid = True
    config.exp.resume = False
    config.exp.resume_checkpoint = ''
    config.exp.eval_checkpoint = str(checkpoint_path)
    config.exp.checkpoint = str(checkpoint_path)
    config.exp.preflight_only = False
    config.exp.run_dir = str(run_dir)
    config.exp.ngpus_per_node = 1
    config.exp.world_size = 1
    config.exp.rank = 0
    config.exp.selection_split = 'test'
    config.optimizer.num_workers = 0
    config.dataset.max_valid_items = 0
    config.dataset.max_test_items = 0
    config.tracking.run_name_suffix = run_name_suffix
    if tracking_mode == 'online':
        config.tracking.use_wandb = True
        os.environ.pop('WANDB_MODE', None)
    elif tracking_mode == 'offline':
        config.tracking.use_wandb = True
        os.environ['WANDB_MODE'] = 'offline'
    else:
        config.tracking.use_wandb = False
        os.environ['WANDB_MODE'] = 'disabled'
    return config

def unwrap_dataset(dataset):
    current = dataset
    while hasattr(current, 'dataset'):
        current = current.dataset
    return current

def build_image_meta_index(dataset):
    dataset = unwrap_dataset(dataset)
    images = dataset.data.get('images', [])
    annotations = dataset.data.get('annotations', [])
    annotation_counts = {}
    for ann in annotations:
        key = str(ann['image_id'])
        annotation_counts[key] = annotation_counts.get(key, 0) + 1
    meta_index = {}
    for image in images:
        image_id = str(image['id'])
        filename = image.get('filename')
        image_path = None
        if filename is not None:
            image_path = str(Path(dataset.root_dir) / filename)
        meta_index[image_id] = {'image_filename': filename, 'image_path': image_path, 'image_width': image.get('width'), 'image_height': image.get('height'), 'reference_count': annotation_counts.get(image_id, 0)}
    return meta_index

def build_dataset_index(dataset):
    dataset = unwrap_dataset(dataset)
    return {str(image_id): index for index, image_id in enumerate(dataset.img_ids)}

def build_reference_tensor(reference, vocab):
    token_ids = [vocab.stoi['<bos>']]
    token_ids.extend(vocab.numericalize(str(reference)))
    token_ids.append(vocab.stoi['<eos>'])
    return torch.tensor(token_ids, dtype=torch.long)

def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))

def summarize_numeric(values, prefix):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {f'{prefix}_count': 0, f'{prefix}_mean': 0.0, f'{prefix}_median': 0.0, f'{prefix}_std': 0.0, f'{prefix}_min': 0.0, f'{prefix}_max': 0.0, f'{prefix}_p90': 0.0, f'{prefix}_p95': 0.0}
    return {f'{prefix}_count': len(values), f'{prefix}_mean': mean(values), f'{prefix}_median': percentile(values, 50), f'{prefix}_std': stddev(values), f'{prefix}_min': float(min(values)), f'{prefix}_max': float(max(values)), f'{prefix}_p90': percentile(values, 90), f'{prefix}_p95': percentile(values, 95)}

def extract_generation_trace(token_ids, token_log_probs, vocab):
    generated_token_ids = []
    generated_tokens = []
    generated_token_logprobs = []
    for token_id, token_logprob in zip(token_ids, token_log_probs):
        token_id = int(token_id)
        token = vocab.itos[token_id]
        generated_token_ids.append(token_id)
        generated_tokens.append(token)
        generated_token_logprobs.append(float(token_logprob))
        if token == '<eos>':
            break
    sequence_logprob = float(sum(generated_token_logprobs))
    mean_token_logprob = float(sequence_logprob / len(generated_token_logprobs)) if generated_token_logprobs else 0.0
    return {'generated_token_ids': generated_token_ids, 'generated_tokens': generated_tokens, 'generated_token_logprobs': generated_token_logprobs, 'beam_sequence_logprob': sequence_logprob, 'beam_mean_token_logprob': mean_token_logprob}

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + '\n')

def compute_row_alignment_fields(prediction, references):
    prediction_tokens = tokenize_caption(prediction)
    prediction_token_set = set(prediction_tokens)
    reference_token_lists = [tokenize_caption(reference) for reference in references]
    reference_union = set((token for tokens in reference_token_lists for token in tokens))
    best_oracle = select_best_reference_tokens(prediction_tokens, references)
    prediction_content_tokens = filter_content_tokens(prediction_tokens)
    prediction_content_set = set(prediction_content_tokens)
    best_content = select_best_reference_tokens(prediction_content_tokens, references, transform=filter_content_tokens)
    best_reference_token_set = set(best_oracle['tokens'])
    extra_tokens = sorted(prediction_token_set - reference_union)
    missing_tokens = sorted(best_reference_token_set - prediction_token_set)
    reference_content_union = {token for token in reference_union if token not in CONTENT_STOP_TOKENS}
    extra_content_tokens = sorted(prediction_content_set - reference_content_union)
    missing_content_tokens = sorted(set(best_content['tokens']) - prediction_content_set)
    exact_match = any((str(prediction) == str(reference) for reference in references))
    return {'oracle_unigram_precision': float(best_oracle['precision']), 'oracle_unigram_recall': float(best_oracle['recall']), 'oracle_unigram_f1': float(best_oracle['f1']), 'content_unigram_precision': float(best_content['precision']), 'content_unigram_recall': float(best_content['recall']), 'content_unigram_f1': float(best_content['f1']), 'extra_tokens': extra_tokens, 'missing_tokens': missing_tokens, 'extra_content_tokens': extra_content_tokens, 'missing_content_tokens': missing_content_tokens, 'exact_match_any_ref': bool(exact_match)}

def compute_row_reference_lengths(references):
    reference_lengths = [len(str(reference).split()) for reference in references]
    reference_len_mean = mean(reference_lengths)
    return {'reference_token_len_mean': reference_len_mean, 'reference_token_len_min': min(reference_lengths) if reference_lengths else 0, 'reference_token_len_max': max(reference_lengths) if reference_lengths else 0}

def collect_predictions(model, dataloader, config):
    model.eval()
    results = {}
    pred_captions = {}
    gt_captions = {}
    rows = []
    vocab = resolve_vocab_from_dataset(dataloader.dataset)
    meta_index = build_image_meta_index(dataloader.dataset)
    decode_started = time.perf_counter()
    with tqdm(desc='official_test decode', unit='it', total=len(dataloader)) as progress:
        for batch_index, batch in enumerate(iter(dataloader)):
            with torch.no_grad():
                outputs, token_log_probs = model(batch['samples'], seq=None, use_beam_search=True, max_len=config.model.beam_len, eos_idx=config.model.eos_idx, beam_size=config.model.beam_size, out_size=1, return_probs=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs_np = outputs.detach().cpu().numpy()
            token_log_probs_np = token_log_probs.detach().cpu().numpy()
            for sample_index, token_ids in enumerate(outputs_np):
                decoded = decode_prediction(token_ids, vocab)
                image_id = str(batch['image_ids'][sample_index])
                references = list(batch['captions'][sample_index])
                row = {'image_id': image_id, 'prediction': decoded['prediction'], 'references': references, 'prediction_token_len': decoded['prediction_token_len'], 'ended_with_eos': decoded['ended_with_eos'], 'contains_unk': decoded['contains_unk'], 'duplicate_ngram_flag': decoded['duplicate_ngram_flag'], 'prediction_has_vietnamese_diacritic': contains_vietnamese_diacritic(decoded['prediction']), **meta_index.get(image_id, {}), **compute_row_reference_lengths(references), **extract_generation_trace(token_ids, token_log_probs_np[sample_index], vocab), **compute_row_alignment_fields(decoded['prediction'], references)}
                results[image_id] = decoded['prediction']
                pred_captions[f'{batch_index}_{sample_index}'] = [decoded['prediction']]
                gt_captions[f'{batch_index}_{sample_index}'] = references
                rows.append(row)
            progress.update()
    decode_time_sec = time.perf_counter() - decode_started
    decode_stats = {'decode_time_sec': float(decode_time_sec), 'decode_samples_per_sec': float(len(rows) / decode_time_sec) if decode_time_sec > 0 else 0.0}
    return (results, pred_captions, gt_captions, rows, decode_stats)

def compute_prediction_metrics(pred_captions, gt_captions, rows, decode_stats, metrics_scope='full'):
    del metrics_scope
    caption_scores, _ = metrics.compute_scores(gt_captions, pred_captions)
    caption_metric_map, caption_diagnostics = compute_caption_quality_metrics(rows)
    metric_map = dict(caption_scores)
    metric_map.update(caption_metric_map)
    metric_map.update(decode_stats)
    diagnostics = dict(caption_diagnostics)
    diagnostics['metric_status'] = {'caption_metrics': {'enabled': True, 'metrics': sorted(caption_scores.keys())}, 'rich_metrics': {'enabled': False, 'reason': 'extended metrics were removed from the runtime path; re-add as a verified evaluator later'}}
    return (metric_map, diagnostics, rows, diagnostics['metric_status'])

def compute_reference_loss_details(model, dataset, image_index_by_id, image_id, vocab, device):
    dataset = unwrap_dataset(dataset)
    dataset_index = image_index_by_id[image_id]
    image_tensor, references, _ = dataset[dataset_index]
    reference_tensors = [build_reference_tensor(reference, vocab) for reference in references]
    pad_idx = vocab.stoi['<pad>']
    single_image_batch = nested_tensor_from_tensor_list([image_tensor]).to(device)
    with torch.no_grad():
        vis_inputs = model.detector(single_image_batch)
    grounding_meta = {'transformed_image_height': int(image_tensor.shape[-2]), 'transformed_image_width': int(image_tensor.shape[-1]), 'gri_token_count': int(vis_inputs['gri_feat'].shape[1]) if 'gri_feat' in vis_inputs else 0, 'reg_token_count': int(vis_inputs['reg_feat'].shape[1]) if 'reg_feat' in vis_inputs else 0}
    reference_rows = []
    for reference_tensor, reference_text in zip(reference_tensors, references):
        batched_images = nested_tensor_from_tensor_list([image_tensor]).to(device)
        batched_targets = pad_sequence([reference_tensor], batch_first=True, padding_value=pad_idx).to(device)
        with torch.no_grad():
            outputs = model(batched_images, batched_targets)
        captions_gt = batched_targets[:, 1:].contiguous()
        outputs = outputs[:, :-1].contiguous()
        token_losses = F.nll_loss(outputs.view(-1, outputs.shape[-1]), captions_gt.view(-1), ignore_index=pad_idx, reduction='none').view(1, -1)
        token_mask = (captions_gt != pad_idx).float()
        loss_sum = float((token_losses * token_mask).sum().item())
        token_count = int(token_mask.sum().item())
        mean_token_nll = float(loss_sum / token_count) if token_count > 0 else 0.0
        reference_rows.append({'reference': str(reference_text), 'teacher_forced_xe_loss': loss_sum, 'teacher_forced_token_count': token_count, 'teacher_forced_mean_token_nll': mean_token_nll})
    reference_rows.sort(key=lambda row: float(row['teacher_forced_xe_loss']))
    return {**grounding_meta, 'teacher_forced_reference_losses': reference_rows}

def attach_reference_losses(model, rows, dataset, device):
    vocab = resolve_vocab_from_dataset(dataset)
    image_index_by_id = build_dataset_index(dataset)
    for row in tqdm(rows, desc='official_test teacher-force', unit='img'):
        image_id = str(row.get('image_id'))
        if image_id not in image_index_by_id:
            continue
        details = compute_reference_loss_details(model=model, dataset=dataset, image_index_by_id=image_index_by_id, image_id=image_id, vocab=vocab, device=device)
        row.update({'transformed_image_height': details['transformed_image_height'], 'transformed_image_width': details['transformed_image_width'], 'gri_token_count': details['gri_token_count'], 'reg_token_count': details['reg_token_count']})
        reference_losses = details['teacher_forced_reference_losses']
        if not reference_losses:
            continue
        best = reference_losses[0]
        row['teacher_forced_best_reference'] = best['reference']
        row['teacher_forced_xe_loss_best_ref'] = float(best['teacher_forced_xe_loss'])
        row['teacher_forced_mean_token_nll_best_ref'] = float(best['teacher_forced_mean_token_nll'])
        row['teacher_forced_xe_loss_mean_refs'] = mean([entry['teacher_forced_xe_loss'] for entry in reference_losses])
        row['teacher_forced_mean_token_nll_mean_refs'] = mean([entry['teacher_forced_mean_token_nll'] for entry in reference_losses])
    return rows

def build_analysis_summary(rows, metric_map, diagnostics, context_payload):
    teacher_best = [row.get('teacher_forced_xe_loss_best_ref') for row in rows if row.get('teacher_forced_xe_loss_best_ref') is not None]
    teacher_nll = [row.get('teacher_forced_mean_token_nll_best_ref') for row in rows if row.get('teacher_forced_mean_token_nll_best_ref') is not None]
    pred_len = [row.get('prediction_token_len') for row in rows if row.get('prediction_token_len') is not None]
    beam_lp = [row.get('beam_sequence_logprob') for row in rows if row.get('beam_sequence_logprob') is not None]
    grounding_gri = [row.get('gri_token_count') for row in rows if row.get('gri_token_count') is not None]
    grounding_reg = [row.get('reg_token_count') for row in rows if row.get('reg_token_count') is not None]
    prediction_to_reference_ratio = [float(row.get('prediction_token_len', 0.0)) / float(row.get('reference_token_len_mean', 1.0)) for row in rows if row.get('reference_token_len_mean')]
    return {'context': context_payload, 'metrics': metric_map, 'caption_diagnostics': diagnostics, 'teacher_forced_xe_loss_best_ref': summarize_numeric(teacher_best, 'teacher_forced_xe_loss_best_ref'), 'teacher_forced_mean_token_nll_best_ref': summarize_numeric(teacher_nll, 'teacher_forced_mean_token_nll_best_ref'), 'prediction_token_len': summarize_numeric(pred_len, 'prediction_token_len'), 'prediction_to_reference_len_ratio': summarize_numeric(prediction_to_reference_ratio, 'prediction_to_reference_len_ratio'), 'beam_sequence_logprob': summarize_numeric(beam_lp, 'beam_sequence_logprob'), 'gri_token_count': summarize_numeric(grounding_gri, 'gri_token_count'), 'reg_token_count': summarize_numeric(grounding_reg, 'reg_token_count')}

def preferred_detail_columns():
    return ['image_id', 'image_filename', 'image_path', 'image_width', 'image_height', 'transformed_image_height', 'transformed_image_width', 'reference_count', 'prediction', 'references', 'teacher_forced_best_reference', 'teacher_forced_xe_loss_best_ref', 'teacher_forced_mean_token_nll_best_ref', 'teacher_forced_xe_loss_mean_refs', 'teacher_forced_mean_token_nll_mean_refs', 'beam_sequence_logprob', 'beam_mean_token_logprob', 'prediction_token_len', 'reference_token_len_mean', 'prediction_to_reference_len_ratio', 'oracle_unigram_precision', 'oracle_unigram_recall', 'oracle_unigram_f1', 'content_unigram_precision', 'content_unigram_recall', 'content_unigram_f1', 'exact_match_any_ref', 'ended_with_eos', 'contains_unk', 'duplicate_ngram_flag', 'gri_token_count', 'reg_token_count', 'length_ratio', 'template_prefix_3', 'prediction_exact_frequency_hint', 'top_exact_caption_frequency', 'top_prefix_template_frequency', 'CLIPScore', 'RefCLIPScore', 'ALOHa', 'QACE-Ref', 'QACE-Img']

def build_pairwise_rows(primary_rows, comparison_rows, comparison_label):
    comparison_by_image = {str(row.get('image_id')): row for row in comparison_rows}
    pairwise_rows = []
    for row in primary_rows:
        image_id = str(row.get('image_id'))
        other = comparison_by_image.get(image_id)
        if not other:
            continue
        pairwise_rows.append({'image_id': image_id, 'references': row.get('references', []), 'candidate_prediction': row.get('prediction', ''), f'{comparison_label}_prediction': other.get('prediction', ''), 'candidate_length_ratio': row.get('length_ratio'), f'{comparison_label}_length_ratio': other.get('length_ratio'), 'candidate_duplicate_ngram_flag': row.get('duplicate_ngram_flag'), f'{comparison_label}_duplicate_ngram_flag': other.get('duplicate_ngram_flag'), 'candidate_CLIPScore': row.get('CLIPScore'), f'{comparison_label}_CLIPScore': other.get('CLIPScore'), 'candidate_RefCLIPScore': row.get('RefCLIPScore'), f'{comparison_label}_RefCLIPScore': other.get('RefCLIPScore'), 'candidate_ALOHa': row.get('ALOHa'), f'{comparison_label}_ALOHa': other.get('ALOHa'), 'candidate_QACE-Ref': row.get('QACE-Ref'), f'{comparison_label}_QACE-Ref': other.get('QACE-Ref'), 'candidate_QACE-Img': row.get('QACE-Img'), f'{comparison_label}_QACE-Img': other.get('QACE-Img'), 'manual_preference': '', 'manual_notes': ''})
    return pairwise_rows

def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    source_train_dir = Path(args.source_train_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    source_manifest = load_json(source_train_dir / 'run_manifest.json')
    source_resolved_config = load_json(source_train_dir / 'trace' / 'resolved_config.json')
    source_contract = load_source_contract(source_train_dir)
    selected_kind, checkpoint_path = pick_checkpoint(source_train_dir, source_manifest, args.checkpoint_preference)
    resolved_dataset_env = configure_dataset_environment(source_contract, dataset_root)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '6688')
    config = build_eval_config(source_config=source_resolved_config, checkpoint_path=checkpoint_path, run_dir=run_dir, tracking_mode=args.tracking_mode, run_name_suffix=args.run_name_suffix)
    config.dataset.max_test_items = int(args.max_test_items)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    observer = RunObserver(config, rank=0)
    tracker = ExperimentTracker(config, rank=0)
    context_payload = {'mode': 'official_test', 'checked_at': iso_timestamp(), 'project_root': str(project_root), 'source_train_dir': str(source_train_dir), 'source_manifest_status': source_manifest.get('status'), 'checkpoint_preference': args.checkpoint_preference, 'selected_checkpoint_kind': selected_kind, 'selected_checkpoint': str(checkpoint_path), 'resolved_dataset_env': resolved_dataset_env, 'source_contract': source_contract, 'system': collect_system_info()}
    try:
        observer.update_status('running')
        write_json(Path('analysis') / 'official_test_context.json', context_payload)
        observer.add_artifact('official_test_context', 'analysis/official_test_context.json')
        if not torch.cuda.is_available():
            raise RuntimeError('Standalone official-test evaluation requires a CUDA-visible GPU environment.')
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        _, dataloaders = get_dataloaders(device=device, batch_size=int(config.optimizer.batch_size), num_workers=int(config.optimizer.num_workers), resize_name=str(config.dataset.transform_cfg.resize_name), size=tuple((int(item) for item in config.dataset.transform_cfg.size)), randaug=bool(config.dataset.transform_cfg.randaug), max_train_items=int(getattr(config.dataset, 'max_train_items', 0)), max_valid_items=int(getattr(config.dataset, 'max_valid_items', 0)), max_test_items=int(getattr(config.dataset, 'max_test_items', 0)))
        detector = build_detector(config).to(device)
        model = Transformer(detector=detector, config=config).to(device)
        model.cached_features = False
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
        observer.event('eval_checkpoint_loaded', checkpoint=str(checkpoint_path), missing=len(missing), unexpected=len(unexpected))
        if len(unexpected) > 0:
            raise RuntimeError(f'Eval checkpoint produced {len(unexpected)} unexpected key(s) at {checkpoint_path}; aborting to avoid evaluating a mismatched model. First few: {list(unexpected)[:8]}')
        if len(missing) > 0:
            print(f'WARNING: eval checkpoint missing {len(missing)} key(s) -> those params stay at init; verify the eval config matches the trained architecture. First few: {list(missing)[:8]}')
        tracker.write_tracking_info(Path('trace') / 'tracking_info.json')
        observer.add_artifact('tracking_info', 'trace/tracking_info.json')
        tracker.log_artifact('trace/tracking_info.json')
        tracker.log_artifact(Path('analysis') / 'official_test_context.json')
        tracker.log_params({'official_test': True, 'rich_metrics_enabled': False, 'source_train_dir': str(source_train_dir), 'selected_checkpoint_kind': selected_kind, 'selected_checkpoint': str(checkpoint_path), 'beam_size': int(config.model.beam_size), 'beam_len': int(config.model.beam_len), 'test_dataset_len': len(dataloaders['test_dict'].dataset), 'caption_field': resolved_dataset_env['caption_field'], 'tokenizer_backend': resolved_dataset_env['tokenizer_backend'], 'vocab_source': resolved_dataset_env['vocab_source'], 'train_json_for_vocab': resolved_dataset_env['train_json'], 'test_json': resolved_dataset_env['test_json']})
        tracker.set_tags({'run_type': 'official_test', 'eval_mode': 'eval_only', 'checkpoint_file': checkpoint_path.name, 'caption_field': resolved_dataset_env['caption_field'], 'tokenizer_backend': resolved_dataset_env['tokenizer_backend']})
        epoch = max(int(getattr(config.optimizer, 'finetune_xe_epochs', 1)), 1)
        predictions, pred_captions, gt_captions, rows, decode_stats = collect_predictions(model=model, dataloader=dataloaders['test_dict'], config=config)
        rows = attach_reference_losses(model=model, rows=rows, dataset=dataloaders['test_dict'].dataset, device=device)
        metric_map, diagnostics, rows, metric_status = compute_prediction_metrics(pred_captions, gt_captions, rows, decode_stats, metrics_scope=args.metrics_scope)
        summary = build_analysis_summary(rows, metric_map, diagnostics, context_payload)
        predictions_file = Path(f'predictions_test_epoch{epoch:02d}.json')
        details_file = Path(f'prediction_details_test_epoch{epoch:02d}.json')
        details_jsonl_file = Path(f'prediction_details_test_epoch{epoch:02d}.jsonl')
        details_csv_file = Path(f'prediction_details_test_epoch{epoch:02d}.csv')
        diagnostics_file = Path(f'caption_diagnostics_test_epoch{epoch:02d}.json')
        metric_status_file = Path(f'metric_status_test_epoch{epoch:02d}.json')
        scores_file = Path(f'test_scores_epoch{epoch:02d}.json')
        loss_summary_file = Path('test_prediction_loss_summary.json')
        pairwise_file = Path('analysis') / 'pairwise_xe_vs_scst_test.csv'
        write_json(predictions_file, predictions)
        write_json(details_file, rows)
        write_jsonl(details_jsonl_file, rows)
        write_csv(details_csv_file, rows, collect_row_fieldnames(rows, preferred=preferred_detail_columns()))
        write_json(diagnostics_file, {**diagnostics, **metric_map})
        write_json(metric_status_file, metric_status)
        write_json(scores_file, metric_map)
        write_json('test_scores_final.json', metric_map)
        write_json(loss_summary_file, summary)
        if args.comparison_details_json:
            comparison_rows = load_json(Path(args.comparison_details_json))
            pairwise_rows = build_pairwise_rows(rows, comparison_rows, args.comparison_label)
            write_csv(pairwise_file, pairwise_rows, collect_row_fieldnames(pairwise_rows))
        write_result_csv(Path('result.csv'), [{'split': 'test', 'epoch': epoch, **metric_map, 'teacher_forced_xe_loss_best_ref_mean': summary['teacher_forced_xe_loss_best_ref']['teacher_forced_xe_loss_best_ref_mean'], 'teacher_forced_mean_token_nll_best_ref_mean': summary['teacher_forced_mean_token_nll_best_ref']['teacher_forced_mean_token_nll_best_ref_mean']}])
        for artifact_name, artifact_path in (('test_predictions', predictions_file), ('test_details', details_file), ('test_details_jsonl', details_jsonl_file), ('test_details_csv', details_csv_file), ('test_caption_diagnostics', diagnostics_file), ('test_metric_status', metric_status_file), ('test_scores', scores_file), ('test_scores_final', Path('test_scores_final.json')), ('test_prediction_loss_summary', loss_summary_file), ('epoch_metrics_csv', Path('result.csv'))):
            observer.add_artifact(artifact_name, str(artifact_path))
            tracker.log_artifact(artifact_path)
        if args.comparison_details_json and pairwise_file.exists():
            observer.add_artifact('test_pairwise_xe_vs_scst', str(pairwise_file))
            tracker.log_artifact(pairwise_file)
        tracker.log_metrics({f'test/{key}': value for key, value in metric_map.items()}, step=epoch)
        tracker.log_table('test_prediction_details', rows, step=epoch)
        context_payload['test_scores'] = metric_map
        context_payload['prediction_rows'] = len(rows)
        write_json(Path('analysis') / 'official_test_context.json', context_payload)
        observer.update_status('completed', test_scores=metric_map, eval_skip_valid=True)
        observer.event('evaluation_completed', test_cider=metric_map['CIDEr'], prediction_rows=len(rows), eval_skip_valid=True)
    except Exception as exc:
        write_json(Path('alert_summary_runtime.json'), {'alerts': [{'code': 'official_test_failure', 'message': str(exc)}]})
        observer.capture_exception(exc)
        observer.update_status('failed', error=str(exc))
        raise
    finally:
        tracker.finish()
if __name__ == '__main__':
    main()
