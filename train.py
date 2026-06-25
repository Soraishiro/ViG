import os
import sys
CONTENT_STOP_TOKENS = frozenset({'<pad>', '<bos>', '<eos>', '<unk>', '<mask>'})
import csv
import json
import socket
import platform
import random
import traceback
import subprocess
import time
import unicodedata
import tempfile
import numpy as np
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import NLLLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm import tqdm
from models.caption import Transformer
from models.caption.detector import build_detector
from utils.cap_scheduler import CosineLRScheduler
from utils.ext_params import is_ext_param
from engine.caption_engine import build_optimizers, gather_result
from dataset import CAPTION_FIELD, RAW_CAPTION_FIELD, TOKENIZER_BACKEND, collect_dataset_contract, ensure_train_vocab_source, get_dataloaders, get_datasets, resolve_ktvic_paths

def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)

def tokenize_caption(text):
    return str(text).strip().split()

def select_best_reference_tokens(prediction_tokens, references, transform=None):
    pred_set = set(prediction_tokens)
    best_score = -1
    best_ref_tokens = references[0] if references else []
    for ref in references:
        ref_tokens = str(ref).split()
        ref_set = set(ref_tokens)
        common = pred_set & ref_set
        score = len(common)
        if score > best_score:
            best_score = score
            best_ref_tokens = ref_tokens
    common = pred_set & set(best_ref_tokens)
    precision = len(common) / len(pred_set) if pred_set else 0
    recall = len(common) / len(best_ref_tokens) if best_ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    result = {'tokens': best_ref_tokens, 'content': filter_content_tokens(best_ref_tokens), 'precision': precision, 'recall': recall, 'f1': f1}
    if transform is not None:
        result['content'] = transform(best_ref_tokens)
    return result

def filter_content_tokens(tokens):
    return [t for t in tokens if t not in CONTENT_STOP_TOKENS]

def as_non_empty_string(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def split_csv_items(value):
    text = as_non_empty_string(value)
    if not text:
        return []
    return [item.strip() for item in text.split(',') if item.strip()]

def slugify_identifier(value):
    text = unicodedata.normalize('NFKD', str(value))
    ascii_text = text.encode('ascii', 'ignore').decode('ascii')
    slug = ''.join((char.lower() if char.isalnum() else '-' for char in ascii_text))
    while '--' in slug:
        slug = slug.replace('--', '-')
    slug = slug.strip('-')
    return slug or 'run'

def resolve_tracking_run_name(base_name, run_dir, use_run_dir_name_suffix=True, explicit_suffix=None):
    parts = []
    if as_bool(use_run_dir_name_suffix):
        run_dir_path = Path(str(run_dir))
        for candidate in (run_dir_path.parent.name, run_dir_path.name):
            candidate = as_non_empty_string(candidate)
            if not candidate:
                continue
            candidate_slug = slugify_identifier(candidate)
            if candidate_slug not in parts:
                parts.append(candidate_slug)
    suffix = as_non_empty_string(explicit_suffix)
    if suffix:
        suffix_slug = slugify_identifier(suffix)
        if suffix_slug not in parts:
            parts.append(suffix_slug)
    if not parts:
        return str(base_name)
    return f"{base_name}-{'-'.join(parts)}"

def iso_timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat()

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)

def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + '\n')

def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in resolved_fieldnames:
                resolved_fieldnames.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, 'a', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def collect_row_fieldnames(rows, preferred=None):
    resolved = list(preferred or [])
    for row in rows:
        for key in row.keys():
            if key not in resolved:
                resolved.append(key)
    return resolved
RESULT_CSV_FIELDNAMES = ['epoch', 'train_loss', 'lr', 'epoch_duration_sec', 'throughput_samples_per_sec', 'throughput_steps_per_sec', 'gpu_memory_allocated_mb', 'gpu_memory_reserved_mb', 'gpu_max_memory_allocated_mb', 'gpu_max_memory_reserved_mb', 'gpu_total_memory_mb', 'gpu_max_memory_utilization']

def rewrite_result_csv(rows):
    write_csv('result.csv', rows, RESULT_CSV_FIELDNAMES)

def probe_git_context():
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True, stderr=subprocess.DEVNULL).strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(['git', 'diff', '--quiet'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        status = subprocess.check_output(['git', 'status', '--short'], text=True, stderr=subprocess.DEVNULL).splitlines()
        return {'repo_root': repo_root, 'commit': commit, 'branch': branch, 'dirty': dirty, 'status_preview': status[:20]}
    except Exception:
        return {'available': False}

def collect_system_info():
    return {'timestamp': iso_timestamp(), 'hostname': socket.gethostname(), 'python': sys.version, 'platform': platform.platform(), 'torch_version': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()), 'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(), 'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'git': probe_git_context()}

def load_trusted_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def get_rng_state_snapshot():
    snapshot = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        snapshot['torch_cuda'] = torch.cuda.get_rng_state_all()
    return snapshot

def restore_rng_state_snapshot(snapshot):
    if not snapshot:
        return
    if snapshot.get('python') is not None:
        random.setstate(snapshot['python'])
    if snapshot.get('numpy') is not None:
        np.random.set_state(snapshot['numpy'])
    if snapshot.get('torch') is not None:
        torch.set_rng_state(snapshot['torch'])
    if torch.cuda.is_available() and snapshot.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all(snapshot['torch_cuda'])

def resolve_resume_checkpoint_path(config):
    resume_checkpoint = as_non_empty_string(getattr(config.exp, 'resume_checkpoint', ''))
    if resume_checkpoint:
        return Path(resume_checkpoint)
    if as_bool(getattr(config.exp, 'auto_resume_latest', False)):
        latest_candidate = Path(str(config.exp.run_dir)) / 'checkpoints' / 'recovery_latest.pth'
        if latest_candidate.exists():
            return latest_candidate
    return None

def build_recovery_checkpoint_payload(model, optimizers, scheduler, epoch, train_loss_history, result_rows, phase_name=None, best_valid_epoch=None, best_valid_cider=None, best_valid_scores=None):
    model_without_ddp = getattr(model, 'module', model)
    return {'checkpoint_type': 'ktvic_baseline3_recovery', 'state_dict': model_without_ddp.state_dict(), 'optim_model': optimizers['model'].state_dict(), 'optim_backbone': optimizers['backbone'].state_dict(), 'optim_ext': optimizers['ext'].state_dict() if optimizers.get('ext') else None, 'scheduler': None if scheduler is None else scheduler.state_dict(), 'completed_epoch': int(epoch), 'phase_name': phase_name, 'train_loss_history': list(train_loss_history), 'result_rows': list(result_rows), 'best_valid_epoch': best_valid_epoch, 'best_valid_cider': best_valid_cider, 'best_valid_scores': best_valid_scores, 'rng_state': get_rng_state_snapshot(), 'saved_at': iso_timestamp()}

def write_recovery_checkpoint(model, optimizers, scheduler, epoch, train_loss_history, result_rows, phase_name=None, best_valid_epoch=None, best_valid_cider=None, best_valid_scores=None):
    checkpoints_dir = Path('checkpoints')
    epoch_path = checkpoints_dir / f'recovery_epoch_{int(epoch):02d}.pth'
    latest_path = checkpoints_dir / 'recovery_latest.pth'
    manifest_path = checkpoints_dir / 'recovery_latest.json'
    payload = build_recovery_checkpoint_payload(model=model, optimizers=optimizers, scheduler=scheduler, epoch=epoch, train_loss_history=train_loss_history, result_rows=result_rows, phase_name=phase_name, best_valid_epoch=best_valid_epoch, best_valid_cider=best_valid_cider, best_valid_scores=best_valid_scores)
    atomic_torch_save(payload, epoch_path)
    atomic_torch_save(payload, latest_path)
    write_json(manifest_path, {'completed_epoch': int(epoch), 'phase_name': phase_name, 'best_valid_epoch': best_valid_epoch, 'best_valid_cider': best_valid_cider, 'epoch_checkpoint': str(epoch_path), 'latest_checkpoint': str(latest_path), 'saved_at': payload['saved_at']})
    return (epoch_path, latest_path, manifest_path)

def write_model_checkpoint(model, path, extra_payload=None):
    model_without_ddp = getattr(model, 'module', model)
    payload = {'state_dict': model_without_ddp.state_dict()}
    if extra_payload:
        payload.update(extra_payload)
    atomic_torch_save(payload, path)
    return Path(path)

def load_model_checkpoint_into_model(model, checkpoint_path):
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location='cpu')
    model_without_ddp = getattr(model, 'module', model)
    missing, unexpected = model_without_ddp.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    return (checkpoint, missing, unexpected)

def set_trainability(model, freeze_detector=False, freeze_backbone=False, freeze_grit=False):
    model_without_ddp = getattr(model, 'module', model)
    for _, parameter in model_without_ddp.named_parameters():
        parameter.requires_grad = True
    if freeze_grit:
        GRIT_STACK_MARKERS = ('detector', 'grid_net', 'cap_generator')
        for name, parameter in model_without_ddp.named_parameters():
            if not is_ext_param(name) and any((marker in name for marker in GRIT_STACK_MARKERS)):
                parameter.requires_grad = False
    if freeze_detector:
        for name, parameter in model_without_ddp.named_parameters():
            if 'detector' in name:
                parameter.requires_grad = False
    elif freeze_backbone:
        for name, parameter in model_without_ddp.named_parameters():
            if 'detector.backbone' in name:
                parameter.requires_grad = False

def zero_optimizers(optimizers):
    for key in ('model', 'backbone', 'ext'):
        optimizer = optimizers.get(key)
        if optimizer is not None:
            optimizer.zero_grad()

def step_optimizers(optimizers):
    for key in ('model', 'backbone', 'ext'):
        optimizer = optimizers.get(key)
        if optimizer is not None:
            optimizer.step()

def clip_trainable_gradients(model, config):
    if config is None:
        return None
    grad_clip = float(getattr(config.optimizer, 'grad_clip', 0.0) or 0.0)
    if grad_clip <= 0:
        return None
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        return None
    return torch.nn.utils.clip_grad_norm_(params, grad_clip)

def load_training_state_from_resume(model, optimizers, scheduler, resume_checkpoint_path):
    checkpoint = load_trusted_checkpoint(resume_checkpoint_path, map_location='cpu')
    checkpoint_type = checkpoint.get('checkpoint_type')
    if checkpoint_type != 'ktvic_baseline3_recovery':
        raise RuntimeError(f'Resume requested but checkpoint is not a KTVIC recovery checkpoint: {resume_checkpoint_path}')
    model_without_ddp = getattr(model, 'module', model)
    missing, unexpected = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizers['model'].load_state_dict(checkpoint['optim_model'])
    optimizers['backbone'].load_state_dict(checkpoint['optim_backbone'])
    if optimizers.get('ext'):
        if checkpoint.get('optim_ext') is None:
            raise RuntimeError('Resume checkpoint is missing optim_ext for an active ext optimizer.')
        optimizers['ext'].load_state_dict(checkpoint['optim_ext'])
    if scheduler is not None and checkpoint.get('scheduler'):
        scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng_state_snapshot(checkpoint.get('rng_state'))
    completed_epoch = int(checkpoint.get('completed_epoch', 0))
    return {'completed_epoch': completed_epoch, 'phase_name': checkpoint.get('phase_name'), 'next_epoch': completed_epoch, 'missing': len(missing), 'unexpected': len(unexpected), 'train_loss_history': list(checkpoint.get('train_loss_history', [])), 'result_rows': list(checkpoint.get('result_rows', [])), 'best_valid_epoch': checkpoint.get('best_valid_epoch'), 'best_valid_cider': checkpoint.get('best_valid_cider'), 'best_valid_scores': checkpoint.get('best_valid_scores'), 'path': str(resume_checkpoint_path), 'checkpoint': checkpoint}

def restore_optimizer_state_from_resume(optimizers, scheduler, resume_state):
    checkpoint = resume_state.get('checkpoint') if resume_state else None
    if not checkpoint:
        return False
    optimizers['model'].load_state_dict(checkpoint['optim_model'])
    optimizers['backbone'].load_state_dict(checkpoint['optim_backbone'])
    if optimizers.get('ext'):
        if checkpoint.get('optim_ext') is None:
            raise RuntimeError('Resume checkpoint is missing optim_ext for an active ext optimizer.')
        optimizers['ext'].load_state_dict(checkpoint['optim_ext'])
    if scheduler is not None and checkpoint.get('scheduler'):
        scheduler.load_state_dict(checkpoint['scheduler'])
    return True

def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    raise RuntimeError('Checkpoint does not contain a supported model state dict.')

def checkpoint_probe(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    report = {'path': str(checkpoint_path), 'exists': checkpoint_path.exists()}
    if not checkpoint_path.exists():
        return report
    stat = checkpoint_path.stat()
    report.update({'size_bytes': stat.st_size, 'mtime': stat.st_mtime})
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint if isinstance(checkpoint, dict) else {})
    report['state_dict_keys'] = len(state_dict)
    report['top_level_keys'] = list(checkpoint.keys())[:20] if isinstance(checkpoint, dict) else []
    return report

def get_gpu_memory_snapshot(device):
    if not torch.cuda.is_available():
        return {'gpu_memory_allocated_mb': 0.0, 'gpu_memory_reserved_mb': 0.0, 'gpu_max_memory_allocated_mb': 0.0, 'gpu_max_memory_reserved_mb': 0.0, 'gpu_total_memory_mb': 0.0, 'gpu_max_memory_utilization': 0.0}
    total_memory_mb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return {'gpu_memory_allocated_mb': round(torch.cuda.memory_allocated(device) / 1024 ** 2, 3), 'gpu_memory_reserved_mb': round(torch.cuda.memory_reserved(device) / 1024 ** 2, 3), 'gpu_max_memory_allocated_mb': round(max_allocated_mb, 3), 'gpu_max_memory_reserved_mb': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 3), 'gpu_total_memory_mb': round(total_memory_mb, 3), 'gpu_max_memory_utilization': round(max_allocated_mb / total_memory_mb, 6) if total_memory_mb > 0 else 0.0}

def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(value)

def get_primary_visible_gpu_index():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible_devices:
        return '0'
    first_device = visible_devices.split(',')[0].strip()
    return first_device or '0'

def get_nvidia_smi_snapshot():
    query_fields = ['utilization.gpu', 'utilization.memory', 'memory.used', 'memory.total', 'temperature.gpu', 'power.draw']
    try:
        output = subprocess.check_output(['nvidia-smi', f"--query-gpu={','.join(query_fields)}", '--format=csv,noheader,nounits', '-i', get_primary_visible_gpu_index()], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception:
        return {}
    if not output:
        return {}
    values = [part.strip() for part in output.split(',')]
    if len(values) != len(query_fields):
        return {}
    used_mb = safe_float(values[2])
    total_mb = safe_float(values[3])
    used_ratio = used_mb / total_mb if used_mb is not None and total_mb not in (None, 0.0) else None
    snapshot = {'gpu_utilization_pct': safe_float(values[0]), 'gpu_memory_utilization_pct': safe_float(values[1]), 'gpu_memory_used_mb': used_mb, 'gpu_memory_total_mb_nvidia_smi': total_mb, 'gpu_temperature_c': safe_float(values[4]), 'gpu_power_draw_w': safe_float(values[5])}
    if used_ratio is not None:
        snapshot['gpu_memory_used_ratio'] = round(float(used_ratio), 6)
    return {key: value for key, value in snapshot.items() if value is not None}

def compute_grad_norm(parameters):
    squared_norm = 0.0
    tensor_count = 0
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        grad_data = grad.detach()
        if grad_data.is_sparse:
            grad_data = grad_data.coalesce().values()
        grad_norm = grad_data.float().norm(2).item()
        squared_norm += grad_norm ** 2
        tensor_count += 1
    return {'grad_norm_l2': round(float(squared_norm ** 0.5), 6), 'grad_tensor_count': int(tensor_count)}

def sanitize_metric_map(metric_map):
    clean = {}
    for key, value in metric_map.items():
        if isinstance(value, bool):
            clean[key] = int(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = safe_float(value)
            if numeric is not None:
                clean[key] = numeric
    return clean

def stringify_param_value(value, max_len=500):
    if isinstance(value, bool):
        text = 'true' if value else 'false'
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text[:max_len]

def has_duplicate_ngram(tokens, n=2):
    if len(tokens) < n:
        return False
    seen = set()
    for index in range(len(tokens) - n + 1):
        ngram = tuple(tokens[index:index + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False

def decode_prediction(token_ids, vocab):
    prediction_tokens = []
    contains_unk = False
    ended_with_eos = False
    for token_id in token_ids:
        token = vocab.itos[int(token_id)]
        if token == '<eos>':
            ended_with_eos = True
            break
        if token == '<unk>':
            contains_unk = True
        if token in ['<pad>', '<unk>', '<sos>', 'eos']:
            continue
        prediction_tokens.append(token)
    return {'prediction': ' '.join(prediction_tokens), 'prediction_tokens': prediction_tokens, 'prediction_token_len': len(prediction_tokens), 'ended_with_eos': ended_with_eos, 'contains_unk': contains_unk, 'duplicate_ngram_flag': has_duplicate_ngram(prediction_tokens)}

def select_sample_image_ids(dataset, limit):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_ids = [base_dataset.img_ids[index] for index in dataset.indices if index < len(base_dataset.img_ids)]
        sorted_ids = sorted(subset_ids)
    else:
        sorted_ids = sorted(dataset.img_ids)
    return sorted_ids[:limit]

def build_probe_loader(dataset, collate_fn, image_ids, batch_size):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        id_to_index = {base_dataset.img_ids[base_index]: subset_index for subset_index, base_index in enumerate(dataset.indices) if base_index < len(base_dataset.img_ids)}
    else:
        id_to_index = {image_id: index for index, image_id in enumerate(dataset.img_ids)}
    indices = [id_to_index[image_id] for image_id in image_ids if image_id in id_to_index]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=max(1, min(batch_size, len(indices))), shuffle=False, num_workers=0, collate_fn=collate_fn)

def build_sample_probe_loaders(dataloaders, config):
    sample_size = int(config.tracking.sample_table_size_per_split)
    batch_size = int(config.tracking.sample_table_batch_size)
    probe_loaders = {}
    for split_name, loader_key in (('train', 'train_dict'),):
        dataset = dataloaders[loader_key].dataset
        image_ids = select_sample_image_ids(dataset, sample_size)
        probe_loaders[split_name] = {'image_ids': image_ids, 'loader': build_probe_loader(dataset, dataloaders[loader_key].collate_fn, image_ids, batch_size)}
    return probe_loaders

def build_alert(code, severity, message, **payload):
    return {'code': code, 'severity': severity, 'message': message, **payload}

def mean(values):
    return float(sum(values) / len(values)) if values else 0.0

def stddev(values):
    if not values:
        return 0.0
    mu = mean(values)
    return float((sum(((value - mu) ** 2 for value in values)) / len(values)) ** 0.5)

def distinct_n(texts, n):
    total = 0
    unique = set()
    for text in texts:
        tokens = str(text).split()
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            ngram = tuple(tokens[index:index + n])
            unique.add(ngram)
            total += 1
    return float(len(unique) / total) if total > 0 else 0.0

def contains_vietnamese_diacritic(text):
    normalized = unicodedata.normalize('NFD', str(text))
    return any((unicodedata.category(char) == 'Mn' for char in normalized))

def compute_caption_quality_metrics(rows):
    if not rows:
        return ({}, {})
    prediction_lengths = [int(row.get('prediction_token_len', 0)) for row in rows]
    reference_lengths = [float(row.get('reference_token_len_mean', 0.0)) for row in rows]
    predictions = [str(row.get('prediction', '')) for row in rows]
    ended_with_eos_rate = sum((1 for row in rows if row.get('ended_with_eos'))) / len(rows)
    unk_rate = sum((1 for row in rows if row.get('contains_unk'))) / len(rows)
    duplicate_rate = sum((1 for row in rows if row.get('duplicate_ngram_flag'))) / len(rows)
    empty_rate = sum((1 for row in rows if int(row.get('prediction_token_len', 0)) <= 1)) / len(rows)
    diacritic_rate = sum((1 for row in rows if contains_vietnamese_diacritic(row.get('prediction', '')))) / len(rows)
    length_ratio_values = []
    for prediction_len, reference_len in zip(prediction_lengths, reference_lengths):
        if reference_len > 0:
            length_ratio_values.append(prediction_len / reference_len)
    return ({'prediction_token_len_mean': mean(prediction_lengths), 'prediction_token_len_std': stddev(prediction_lengths), 'reference_token_len_mean': mean(reference_lengths), 'prediction_to_reference_len_ratio_mean': mean(length_ratio_values), 'ended_with_eos_rate': float(ended_with_eos_rate), 'unk_rate': float(unk_rate), 'duplicate_ngram_rate': float(duplicate_rate), 'empty_or_near_empty_rate': float(empty_rate), 'distinct_1': distinct_n(predictions, 1), 'distinct_2': distinct_n(predictions, 2), 'vietnamese_diacritic_rate': float(diacritic_rate)}, {})

def resolve_vocab_from_dataset(dataset):
    current = dataset
    for _ in range(5):
        if hasattr(current, 'vocab'):
            return current.vocab
        if hasattr(current, 'dataset'):
            current = current.dataset
            continue
        break
    raise AttributeError(f'Dataset of type {type(dataset).__name__} does not expose vocab')

class RunObserver:

    def __init__(self, config: DictConfig, rank: int):
        self.rank = rank
        self.enabled = rank == 0 and as_bool(config.exp.write_trace)
        self.run_dir = Path.cwd()
        self.trace_dir = self.run_dir / 'trace'
        self.events_path = self.trace_dir / 'events.jsonl'
        self.manifest_path = self.run_dir / 'run_manifest.json'
        self.config_json_path = self.trace_dir / 'resolved_config.json'
        self.config_yaml_path = self.trace_dir / 'resolved_config.yaml'
        self.system_info_path = self.trace_dir / 'system_info.json'
        self.preflight_path = self.trace_dir / 'preflight_report.json'
        self.summary = {'run_name': str(config.exp.name), 'status': 'initializing', 'started_at': iso_timestamp(), 'run_dir': str(self.run_dir), 'caption_field': CAPTION_FIELD, 'tokenizer_backend': TOKENIZER_BACKEND, 'artifacts': {}}
        if not self.enabled:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        config_payload = OmegaConf.to_container(config, resolve=True)
        write_json(self.config_json_path, config_payload)
        with open(self.config_yaml_path, 'w', encoding='utf-8') as file_obj:
            file_obj.write(OmegaConf.to_yaml(config, resolve=True))
        write_json(self.system_info_path, collect_system_info())
        self.summary['artifacts']['resolved_config_json'] = str(self.config_json_path.relative_to(self.run_dir))
        self.summary['artifacts']['resolved_config_yaml'] = str(self.config_yaml_path.relative_to(self.run_dir))
        self.summary['artifacts']['system_info'] = str(self.system_info_path.relative_to(self.run_dir))
        self.flush()
        self.event('run_initialized', config_path=str(self.config_json_path))

    def flush(self):
        if self.enabled:
            write_json(self.manifest_path, self.summary)

    def event(self, event_type, **payload):
        if not self.enabled:
            return
        append_jsonl(self.events_path, {'ts': iso_timestamp(), 'event': event_type, **payload})

    def add_artifact(self, name, path):
        if self.enabled:
            self.summary['artifacts'][name] = str(path)
            self.flush()

    def update_status(self, status, **payload):
        if not self.enabled:
            return
        self.summary['status'] = status
        self.summary.update(payload)
        if status in {'completed', 'failed', 'preflight_ok', 'preflight_failed'}:
            self.summary['finished_at'] = iso_timestamp()
        self.flush()

    def write_preflight(self, report):
        if not self.enabled:
            return
        write_json(self.preflight_path, report)
        self.add_artifact('preflight_report', self.preflight_path.relative_to(self.run_dir))

    def capture_exception(self, exc):
        self.event('exception', error=str(exc), traceback=traceback.format_exc())

class LiveMetricRecorder:

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.jsonl_path = Path('trace') / 'step_metrics.jsonl'
        self.csv_path = Path('trace') / 'step_metrics.csv'
        self.latest_path = Path('trace') / 'live_status.json'
        self.fieldnames = None

    def record(self, payload):
        if not self.enabled:
            return
        append_jsonl(self.jsonl_path, payload)
        incoming_fieldnames = collect_row_fieldnames([payload], preferred=self.fieldnames)
        if self.fieldnames != incoming_fieldnames:
            self.fieldnames = incoming_fieldnames
            rows = []
            if self.jsonl_path.exists():
                with open(self.jsonl_path, 'r', encoding='utf-8') as file_obj:
                    rows = [json.loads(line) for line in file_obj if line.strip()]
            write_csv(self.csv_path, rows, self.fieldnames)
        else:
            append_csv_row(self.csv_path, payload, self.fieldnames)
        write_json(self.latest_path, payload)

class ExperimentTracker:

    def __init__(self, config: DictConfig, rank: int):
        self.rank = rank
        self.use_mlflow = False
        self.use_wandb = False
        self.mlflow = None
        self.wandb_run = None
        self.project_name = str(getattr(config.tracking, 'project_name', 'ktvic-baseline3'))
        self.run_name = resolve_tracking_run_name(base_name=str(getattr(config.tracking, 'run_name', 'run')), run_dir=str(config.exp.run_dir), use_run_dir_name_suffix=getattr(config.tracking, 'use_run_dir_name_suffix', True), explicit_suffix=getattr(config.tracking, 'run_name_suffix', ''))
        self._metrics_jsonl = Path('trace') / 'run_metrics.jsonl'
        self._params_path = Path('trace') / 'run_params.json'
        if self.rank == 0:
            print(f'[RunLogger] run_name={self.run_name} (file-only, no wandb/mlflow)')

    def log_metrics(self, metric_map, step=None):
        if self.rank != 0:
            return
        clean_metrics = sanitize_metric_map(metric_map)
        if not clean_metrics:
            return
        payload = {'step': step, 'ts': iso_timestamp(), **clean_metrics}
        append_jsonl(self._metrics_jsonl, payload)

    def log_params(self, param_map):
        if self.rank != 0:
            return
        clean_params = {key: stringify_param_value(value) for key, value in param_map.items() if value is not None}
        if not clean_params:
            return
        write_json(self._params_path, clean_params)

    def set_tags(self, tag_map):
        pass

    def log_artifact(self, path):
        path = Path(path)
        if self.rank != 0:
            return
        if not path.exists():
            print(f'[RunLogger] WARN: artifact not found: {path}')

    def log_table(self, name, rows, step=None):
        if self.rank != 0 or not rows:
            return
        columns = collect_row_fieldnames(rows)
        table_dir = Path('tables')
        table_dir.mkdir(parents=True, exist_ok=True)
        data = [[row.get(column) for column in columns] for row in rows]
        write_json(table_dir / f'{name}.json', {'columns': columns, 'rows': data, 'step': step})

    def write_tracking_info(self, path):
        if self.rank != 0:
            return
        write_json(path, {'project_name': self.project_name, 'run_name': self.run_name, 'backend': 'file-only'})

    def finish(self):
        pass

    def register_model_checkpoint(self, checkpoint_path, config_path=None, test_scores_path=None):
        if self.rank != 0:
            return None
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        metadata_path = Path('trace') / 'registered_model_metadata.json'
        write_json(metadata_path, {'registered_at': iso_timestamp(), 'run_name': self.run_name, 'checkpoint_path': str(checkpoint_path.resolve())})
        return {'metadata_path': str(metadata_path), 'checkpoint_path': str(checkpoint_path.resolve())}

def predict_captions(model, dataloader, config):
    results = {}
    pred_captions = {}
    gt_captions = {}
    detailed_rows = []
    vocab = resolve_vocab_from_dataset(dataloader.dataset)
    decode_started = time.perf_counter()
    for it, batch in enumerate(iter(dataloader)):
        with torch.no_grad():
            out, _ = model(batch['samples'], seq=None, use_beam_search=True, max_len=config.model.beam_len, eos_idx=config.model.eos_idx, beam_size=config.model.beam_size, out_size=1, return_probs=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out = out.cpu().numpy()
        for index, token_ids in enumerate(out):
            decoded = decode_prediction(token_ids, vocab)
            references = batch['captions'][index]
            image_id = str(batch.get('image_ids', [str(index)] * (index + 1))[index])
            reference_lengths = [len(str(reference).split()) for reference in references]
            pred_captions[f'{it}_{index}'] = [decoded['prediction']]
            gt_captions[f'{it}_{index}'] = references
            results[image_id] = decoded['prediction']
            detailed_rows.append({'image_id': image_id, 'prediction': decoded['prediction'], 'references': references, 'prediction_token_len': decoded['prediction_token_len'], 'ended_with_eos': decoded['ended_with_eos'], 'contains_unk': decoded['contains_unk'], 'duplicate_ngram_flag': decoded['duplicate_ngram_flag'], 'reference_token_len_mean': mean(reference_lengths), 'prediction_has_vietnamese_diacritic': contains_vietnamese_diacritic(decoded['prediction'])})
    decode_time_sec = time.perf_counter() - decode_started
    decode_stats = {'decode_time_sec': float(decode_time_sec), 'decode_samples_per_sec': float(len(detailed_rows) / decode_time_sec) if decode_time_sec > 0 else 0.0}
    return (results, pred_captions, gt_captions, detailed_rows, decode_stats)

def evaluate_metrics(model, dataloader, config, output_file, details_output_file=None, unified_output_file=None, run_meta=None):
    model.eval()
    results, pred_captions, gt_captions, detailed_rows, decode_stats = predict_captions(model, dataloader, config)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, sort_keys=True, ensure_ascii=False)
    if details_output_file is not None:
        write_json(details_output_file, detailed_rows)
    from datasets.caption import metrics as _metrics
    score_map = _metrics.compute_scores(gt_captions, pred_captions)[0]
    metric_map = {'BLEU-1': float(score_map['BLEU'][0]), 'BLEU-4': float(score_map['BLEU'][3]), 'METEOR': float(score_map['METEOR']), 'ROUGE': float(score_map['ROUGE']), 'CIDEr': float(score_map['CIDEr'])}
    _quality_metrics, _ = compute_caption_quality_metrics(detailed_rows)
    metric_map.update(_quality_metrics)
    metric_map.update(decode_stats)
    if unified_output_file is not None:
        gt_by_image_id = {row['image_id']: row['references'] for row in detailed_rows}
        write_json(unified_output_file, {'meta': run_meta or {}, 'predictions': results, 'groundtruth': gt_by_image_id})
    return (metric_map, detailed_rows)

def capture_sample_caption_tables(model, probe_loaders, config, epoch, observer=None, tracker=None):
    model.eval()
    all_rows = []
    for split_name, payload in probe_loaders.items():
        _, _, _, rows, _ = predict_captions(model, payload['loader'], config)
        for row in rows:
            row['split'] = split_name
        all_rows.extend(rows)
    json_path = Path('samples') / f'sample_captions_epoch_{epoch:02d}.json'
    csv_path = Path('samples') / f'sample_captions_epoch_{epoch:02d}.csv'
    fieldnames = collect_row_fieldnames(all_rows, preferred=['split', 'image_id', 'prediction', 'references', 'prediction_token_len', 'ended_with_eos', 'contains_unk', 'duplicate_ngram_flag'])
    write_json(json_path, all_rows)
    write_csv(csv_path, all_rows, fieldnames)
    if observer is not None:
        observer.add_artifact(f'sample_captions_epoch_{epoch:02d}_json', str(json_path))
        observer.add_artifact(f'sample_captions_epoch_{epoch:02d}_csv', str(csv_path))
    if tracker is not None:
        tracker.log_artifact(json_path)
        tracker.log_artifact(csv_path)
        tracker.log_table(f'sample_captions_epoch_{epoch:02d}', all_rows, step=epoch)
    return (all_rows, json_path, csv_path)

def build_sample_alerts(sample_rows, config):
    alerts = []
    if not sample_rows:
        return alerts
    total = len(sample_rows)
    truncation_rate = sum((1 for row in sample_rows if int(row['prediction_token_len']) < int(config.tracking.caption_truncation_min_tokens))) / total
    unk_rate = sum((1 for row in sample_rows if row['contains_unk'])) / total
    duplicate_rate = sum((1 for row in sample_rows if row['duplicate_ngram_flag'])) / total
    empty_rate = sum((1 for row in sample_rows if int(row['prediction_token_len']) <= 1)) / total
    if truncation_rate > float(config.tracking.caption_truncation_alert_rate):
        alerts.append(build_alert('caption_truncation_spike', 'warning', f'Sample caption truncation rate reached {truncation_rate:.2%}.', rate=truncation_rate))
    if unk_rate > float(config.tracking.unk_alert_rate):
        alerts.append(build_alert('unk_spike', 'warning', f'Sample caption <unk> rate reached {unk_rate:.2%}.', rate=unk_rate))
    if duplicate_rate > float(config.tracking.duplicate_ngram_alert_rate):
        alerts.append(build_alert('duplicate_ngram_spike', 'warning', f'Sample duplicate-ngram rate reached {duplicate_rate:.2%}.', rate=duplicate_rate))
    if empty_rate > float(config.tracking.empty_caption_alert_rate):
        alerts.append(build_alert('empty_output_spike', 'warning', f'Sample empty/near-empty caption rate reached {empty_rate:.2%}.', rate=empty_rate))
    diacritic_rate = sum((1 for row in sample_rows if row.get('prediction_has_vietnamese_diacritic'))) / total
    if diacritic_rate < float(config.tracking.vietnamese_diacritic_min_rate):
        alerts.append(build_alert('vietnamese_diacritic_drop', 'warning', f'Sample Vietnamese diacritic rate dropped to {diacritic_rate:.2%}.', rate=diacritic_rate))
    return alerts

def evaluate_named_split(model, dataloader, config, split_name, epoch, observer=None, tracker=None):
    predictions_file = f'predictions_{split_name}_epoch{epoch:02d}.json'
    details_file = f'prediction_details_{split_name}_epoch{epoch:02d}.json'
    unified_file = f'predictions_unified_{split_name}_epoch{epoch:02d}.json'
    run_meta = {'epoch': epoch, 'split': split_name, 'training_regime': str(getattr(config.exp, 'training_regime', '') or ''), 'run_dir': str(Path.cwd()), 'timestamp': iso_timestamp()}
    scores, rows = evaluate_metrics(model=model, dataloader=dataloader, config=config, output_file=predictions_file, details_output_file=details_file, unified_output_file=unified_file, run_meta=run_meta)
    scores_file = f'{split_name}_scores_epoch{epoch:02d}.json'
    write_json(scores_file, scores)
    if observer is not None:
        observer.add_artifact(f'{split_name}_predictions_epoch_{epoch:02d}', predictions_file)
        observer.add_artifact(f'{split_name}_details_epoch_{epoch:02d}', details_file)
        observer.add_artifact(f'{split_name}_scores_epoch_{epoch:02d}', scores_file)
        observer.add_artifact(f'{split_name}_predictions_unified_epoch_{epoch:02d}', unified_file)
    if tracker is not None:
        tracker.log_artifact(predictions_file)
        tracker.log_artifact(details_file)
        tracker.log_artifact(scores_file)
        tracker.log_artifact(unified_file)
        tracker.log_metrics({f'{split_name}/{key}': value for key, value in scores.items()}, step=epoch)
    return (scores, rows, Path(scores_file))

def compute_l_div(C_loc):
    C_norm = F.normalize(C_loc, dim=-1)
    sim = torch.bmm(C_norm, C_norm.transpose(1, 2))
    mask = ~torch.eye(C_loc.shape[1], dtype=torch.bool, device=C_loc.device)
    off_diag = sim[:, mask].view(C_loc.shape[0], C_loc.shape[1], C_loc.shape[1] - 1)
    return (off_diag ** 2).mean()

def compute_l_loc(C_proj, phrase_embeddings, positives_mask, tau=0.07):
    C_norm = F.normalize(C_proj, dim=-1)
    E_norm = F.normalize(phrase_embeddings, dim=-1)
    cos_sim = torch.einsum('bkd,pd->bkp', C_norm, E_norm)
    s_ip = torch.logsumexp(cos_sim / tau, dim=1)
    pos_scores = s_ip.masked_fill(~positives_mask, float('-inf'))
    num_logsumexp = torch.logsumexp(pos_scores, dim=1)
    denom_logsumexp = torch.logsumexp(s_ip, dim=1)
    loss = -(num_logsumexp - denom_logsumexp).mean()
    return loss

def compute_l_route(eta_loc, g_loc, T_loc_mask, eps=1e-08):
    if not T_loc_mask.any():
        return torch.tensor(0.0, device=eta_loc.device)
    eta = eta_loc.squeeze(-1)
    g_mean = g_loc.mean(dim=-1)
    route_score = eta * g_mean
    cultural_scores = route_score[T_loc_mask]
    loss = -torch.log(eps + cultural_scores).mean()
    return loss

def _offdiag_cos(x):
    xn = F.normalize(x, dim=-1)
    K = xn.size(1)
    cm = torch.bmm(xn, xn.transpose(1, 2))
    eye = torch.eye(K, dtype=torch.bool, device=x.device)
    return cm[:, ~eye].mean().item()

def instrument_grad_decomp(l_xe_raw, l_loc_raw, l_div_raw, C_loc, lam_loc, lam_div, l_gate_raw=None, l_route_raw=None):
    m = {}

    def grad_of(loss):
        if not torch.is_tensor(loss) or loss.grad_fn is None:
            return None
        return torch.autograd.grad(loss, C_loc, retain_graph=True, allow_unused=True)[0]
    g_xe, g_loc, g_div = (grad_of(l_xe_raw), grad_of(l_loc_raw), grad_of(l_div_raw))
    g_gate = grad_of(l_gate_raw) if l_gate_raw is not None else None
    g_route = grad_of(l_route_raw) if l_route_raw is not None else None

    def nrm(g):
        return g.norm().item() if g is not None else 0.0
    m['grad_xe_raw'], m['grad_loc_raw'], m['grad_div_raw'] = (nrm(g_xe), nrm(g_loc), nrm(g_div))
    m['grad_xe_eff'] = m['grad_xe_raw']
    m['grad_loc_eff'] = lam_loc * m['grad_loc_raw']
    m['grad_div_eff'] = lam_div * m['grad_div_raw']
    if g_gate is not None:
        m['grad_gate_raw'] = nrm(g_gate)
    if g_route is not None:
        m['grad_route_raw'] = nrm(g_route)

    def cos(a, b):
        if a is None or b is None:
            return 0.0
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    m['cos_xe_div'] = cos(g_xe, g_div)
    m['cos_xe_loc'] = cos(g_xe, g_loc)
    m['cos_loc_div'] = cos(g_loc, g_div)
    m['cos_xe_gate'] = cos(g_xe, g_gate)
    m['cos_xe_route'] = cos(g_xe, g_route)
    if g_xe is not None:
        m['grad_xe_slot_cos'] = _offdiag_cos(g_xe)
    if g_loc is not None:
        m['grad_loc_slot_cos'] = _offdiag_cos(g_loc)
    return m

def train_xe(model, dataloaders, optimizers, epoch, total_epochs=1, trace_every_n_steps=0, step_metrics_every_n_steps=0, observer=None, tracker=None, live_metric_recorder=None, scheduler=None, enable_nvidia_smi_metrics=False, log_gradient_norm=False, max_train_batches=0, config=None):
    vocab = resolve_vocab_from_dataset(dataloaders['train'].dataset)
    model.train()
    loss_fn = NLLLoss(ignore_index=vocab.stoi['<pad>'])
    running_loss = 0.0
    processed_samples = 0
    device = next(model.parameters()).device
    loss_ext_cfg = getattr(config, 'loss_ext', None) if config is not None else None
    if loss_ext_cfg is not None:
        lambda_loc = float(getattr(loss_ext_cfg, 'lambda_loc', 0.0))
        lambda_route = float(getattr(loss_ext_cfg, 'lambda_route', 0.1))
        lambda_div = float(getattr(loss_ext_cfg, 'lambda_div', 0.01))
        tau = float(getattr(loss_ext_cfg, 'tau', 0.07))
        warmup_epochs = int(getattr(loss_ext_cfg, 'warmup_epochs', 2))
        ramp_epochs = int(getattr(loss_ext_cfg, 'ramp_epochs', 3))
        loss_eps = float(getattr(loss_ext_cfg, 'eps', 1e-08))
    else:
        lambda_loc = 0.0
        lambda_route = 0.0
        lambda_div = 0.0
        tau = 0.07
        warmup_epochs = 2
        ramp_epochs = 3
        loss_eps = 1e-08
    model_ext_cfg = getattr(config, 'model_ext', None) if config is not None else None
    lcqm_cfg = getattr(model_ext_cfg, 'lcqm', None) if model_ext_cfg is not None else None
    lcqm_enabled = bool(getattr(lcqm_cfg, 'enabled', False)) if lcqm_cfg is not None else False
    lambda_gate = 0.0
    gate_target_openness = 0.1
    use_heuristic_bias = False
    heuristic_step_size = 0.01
    if lcqm_enabled and lcqm_cfg is not None:
        lambda_gate = float(getattr(lcqm_cfg, 'lambda_gate', 0.0))
        gate_target_openness = float(getattr(lcqm_cfg, 'gate_target_openness', 0.1))
        use_heuristic_bias = bool(getattr(lcqm_cfg, 'use_heuristic_bias', False))
        heuristic_step_size = float(getattr(lcqm_cfg, 'heuristic_step_size', 0.01))
    gate_bias_schedule_enabled = False
    gate_bias_start = -5.0
    gate_bias_end = -1.0
    gate_schedule_epochs = 5
    if lcqm_enabled and lcqm_cfg is not None:
        gate_bias_start = float(getattr(lcqm_cfg, 'gate_bias_start', -5.0))
        gate_bias_end = float(getattr(lcqm_cfg, 'gate_bias_end', -1.0))
        gate_schedule_epochs = int(getattr(lcqm_cfg, 'gate_schedule_epochs', 5))
        gate_bias_schedule_enabled = gate_schedule_epochs > 0 and gate_bias_start != gate_bias_end
    if gate_bias_schedule_enabled and epoch >= warmup_epochs:
        progress = min(1.0, (epoch - warmup_epochs) / max(gate_schedule_epochs, 1))
        new_bias = gate_bias_start + (gate_bias_end - gate_bias_start) * progress
        m = getattr(model, 'module', model)
        if hasattr(m, 'cap_generator') and hasattr(m.cap_generator, 'layers'):
            for layer in m.cap_generator.layers:
                if hasattr(layer, 'fc_alpha4'):
                    layer.fc_alpha4.bias.data.fill_(new_bias)
            print(f'[GATE] Epoch {epoch + 1}: fc_alpha4.bias = {new_bias:.2f} (σ≈{torch.sigmoid(torch.tensor(new_bias)).item():.3f})')
        else:
            print(f'[GATE] WARN: schedule enabled but cap_generator.layers not found')
    phrase_embeddings = None
    positives_mask = None
    token_masks = None
    phrase_list = None
    phrase_index = None
    _script_dir = Path(__file__).resolve().parent
    _emb = _script_dir / 'data/phrase_embeddings.npy'
    _pos = _script_dir / 'data/phrase_positives_train.json'
    _plist = _script_dir / 'data/phrase_list.json'
    _tkm = _script_dir / 'data/token_masks_train.json'
    if lcqm_enabled and lambda_loc > 0:
        missing = []
        if not _emb.exists():
            missing.append(str(_emb))
        if not _pos.exists():
            missing.append(str(_pos))
        if not _plist.exists():
            missing.append(str(_plist))
        if missing:
            raise FileNotFoundError(f'[LCQM] FATAL: lambda_loc={lambda_loc} > 0 nhưng thiếu phrase data files:\n  missing: {missing}\n  Cần chạy:\n    1. python tools/mine_local_phrases.py --train_annotations <path/to/train.json>\n    2. python tools/build_phrase_supervision.py --vocab_file data/local_phrase_vocab_v1.json --train_annotations <path/to/train.json>\n  Sau đó resubmit job.')
        import numpy as np
        phrase_embeddings = torch.tensor(np.load(str(_emb)), dtype=torch.float32, device=device)
        with open(_pos, 'r') as f:
            positives_mask = json.load(f)
        with open(_plist, 'r') as f:
            phrase_list = json.load(f)
        phrase_index = {p: i for i, p in enumerate(phrase_list)}
        print(f'[LCQM] Loaded phrase_embeddings {phrase_embeddings.shape}, {len(positives_mask)} images, {len(phrase_list)} phrases')
    if lcqm_enabled and lambda_route > 0:
        if not _tkm.exists():
            raise FileNotFoundError(f'[LCQM] FATAL: lambda_route={lambda_route} > 0 nhưng thiếu token_masks file:\n  missing: {_tkm}\n  Token masks cần được precompute từ phrase vocabulary + training captions.\n  Resubmit job sau khi tạo file này.')
        with open(_tkm, 'r') as f:
            token_masks = json.load(f)
        print(f'[LCQM] Loaded token_masks {len(token_masks)} images')
    elif _tkm.exists():
        with open(_tkm, 'r') as f:
            token_masks = json.load(f)
        print(f'[LCQM] Loaded token_masks {len(token_masks)} images (lambda_route=0 — không dùng trong loss)')
    _l_loc_batches = 0
    _l_loc_skipped = 0
    _l_route_batches = 0
    _l_route_skipped = 0
    _gate_openness_sum = 0.0
    _gate_batches = 0
    _l_route_skipped = 0
    start_time = time.perf_counter()
    window_started_at = start_time
    window_samples = 0
    window_steps = 0
    window_loss = 0.0
    configured_steps = len(dataloaders['train'])
    batch_limit = int(max_train_batches or 0)
    total_steps = min(configured_steps, batch_limit) if batch_limit > 0 else configured_steps
    total_run_steps = total_steps * max(int(total_epochs), 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with tqdm(desc=f'Epoch {epoch + 1} - train', unit='it', total=total_steps) as pbar:
        for it, batch in enumerate(dataloaders['train']):
            if it >= total_steps:
                break
            step_started_at = time.perf_counter()
            result = model(batch['samples'], batch['captions'], return_aux=True)
            out, aux = result if isinstance(result, tuple) else (result, {})
            zero_optimizers(optimizers)
            captions_gt = batch['captions'][:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            l_xe_raw = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))
            C_loc = aux.get('loc_feat')
            C_loc_proj = aux.get('loc_feat_proj')
            if C_loc is None and epoch >= warmup_epochs and lcqm_enabled and (lambda_loc > 0 or lambda_route > 0 or lambda_div > 0):
                raise RuntimeError(f"[LCQM] FATAL: LCQM enabled (lambda_loc={lambda_loc}, lambda_route={lambda_route}, lambda_div={lambda_div}) nhung aux['loc_feat'] is None o epoch={epoch + 1}, step={it + 1}.\n  Transformer._add_local_cultural_features() khong set loc_feat vao vis_inputs.\n  Kiem tra: self.lcqm co None khong? use_local_cultural_branch co True khong?")
            _device = C_loc.device if C_loc is not None else device
            l_div_raw = compute_l_div(C_loc) if C_loc is not None else torch.zeros((), device=_device)
            l_loc_raw = torch.zeros((), device=_device)
            _batch_has_pos = False
            if C_loc_proj is not None and phrase_embeddings is not None and (positives_mask is not None) and (phrase_list is not None):
                img_ids = batch.get('image_ids')
                if img_ids is not None:
                    _l_loc_batches += 1
                    P = len(phrase_list)
                    B = len(img_ids)
                    pos_mask = torch.zeros(B, P, dtype=torch.bool, device=_device)
                    for b, img_id in enumerate(img_ids):
                        for phrase in positives_mask.get(str(img_id), []):
                            if phrase in phrase_index:
                                pos_mask[b, phrase_index[phrase]] = True
                    if pos_mask.any():
                        l_loc_raw = compute_l_loc(C_loc_proj, phrase_embeddings, pos_mask, tau=tau)
                        l_loc_raw = torch.clamp(l_loc_raw, max=10.0)
                        _batch_has_pos = True
                    else:
                        _l_loc_skipped += 1
                        if _l_loc_skipped == 1:
                            print(f'[LCQM] WARN: L_loc skipped batch đầu tiên (epoch={epoch + 1}, step={it + 1}) — ảnh trong batch không có phrase trong positives_mask. Sẽ tiếp tục skip các batch tương tự.')
            l_gate_raw = None
            l_route_raw = None
            alpha4_real = aux.get('alpha4')
            if alpha4_real is not None:
                alpha4_mean = alpha4_real.mean()
                l_gate_raw = torch.relu(gate_target_openness - alpha4_mean)
                _gate_openness_sum += alpha4_mean.item()
                _gate_batches += 1
                if it == 0:
                    a4_val = alpha4_mean.item()
                    print(f'[GATE] Epoch {epoch + 1}: α4_mean={a4_val:.3f}, target={gate_target_openness})')
            eta_loc = aux.get('eta_loc')
            g_loc = aux.get('g_loc')
            if eta_loc is not None:
                eta_loc = eta_loc[:, 1:, :]
            if g_loc is not None:
                g_loc = g_loc[:, 1:, :]
            if eta_loc is not None and g_loc is not None and (token_masks is not None):
                img_ids = batch.get('image_ids')
                if img_ids is not None:
                    _l_route_batches += 1
                    T = captions_gt.shape[1]
                    T_loc_mask = torch.zeros(len(img_ids), T, dtype=torch.bool, device=_device)
                    for b, img_id in enumerate(img_ids):
                        positions = token_masks.get(str(img_id), [])
                        for pos in positions:
                            if pos < T:
                                T_loc_mask[b, pos] = True
                    if T_loc_mask.any():
                        l_route_raw = compute_l_route(eta_loc, g_loc, T_loc_mask, eps=loss_eps)
                        l_route_raw = torch.clamp(l_route_raw, max=10.0)
                    else:
                        _l_route_skipped += 1
                        if _l_route_skipped == 1:
                            print(f'[LCQM] WARN: L_route skipped batch đầu tiên (epoch={epoch + 1}, step={it + 1}) — không có cultural token nào trong batch. Sẽ tiếp tục skip các batch tương tự.')
            if C_loc is not None and epoch >= warmup_epochs:
                ramp = min(1.0, (epoch - warmup_epochs + 1) / max(ramp_epochs, 1))
                lam_loc = lambda_loc * ramp
                lam_route = lambda_route * ramp
            else:
                lam_loc = 0.0
                lam_route = 0.0
            if it % 50 == 0 and C_loc is not None:
                if it == 0 and _batch_has_pos:
                    try:
                        _g_test = torch.autograd.grad(l_loc_raw, C_loc, retain_graph=True, allow_unused=False)
                    except RuntimeError as _e:
                        _g_test = None
                        print(f'[STREAM2] FATAL: G3 FAILED — {_e}', file=sys.stderr)
                    if _g_test is None or _g_test[0] is None:
                        print(f'[STREAM2] FATAL: C_loc not in graph of l_loc_raw (G3 FAILED). C_loc.requires_grad={C_loc.requires_grad}, l_loc_raw.grad_fn={l_loc_raw.grad_fn}', file=sys.stderr)
                    else:
                        _gn = _g_test[0].norm().item()
                        print(f'[STREAM2] G3 PASS: l_loc_raw→C_loc grad_norm={_gn:.6f} (L_loc IS connected to C_loc)')
                _inst = instrument_grad_decomp(l_xe_raw, l_loc_raw, l_div_raw, C_loc, lam_loc, lambda_div, l_gate_raw=l_gate_raw, l_route_raw=l_route_raw)
                _inst['batch_has_pos'] = int(_batch_has_pos)
                _inst_dir = Path(__file__).resolve().parent
                _inst_path = _inst_dir / 'instrument_metrics.jsonl'
                with open(_inst_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_inst, ensure_ascii=False) + '\n')
            loss = l_xe_raw + lam_loc * l_loc_raw + lambda_div * l_div_raw
            if l_gate_raw is not None and lambda_gate > 0 and (epoch >= warmup_epochs):
                loss = loss + lambda_gate * l_gate_raw
            if l_route_raw is not None and lam_route > 0 and (epoch >= warmup_epochs):
                loss = loss + lam_route * l_route_raw
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite train loss detected at epoch={epoch + 1}, step={it + 1}: {loss.item()}')
            loss.backward()
            clip_trainable_gradients(model, config)
            step_optimizers(optimizers)
            if scheduler is not None:
                scheduler.step()
            loss = gather_result(loss)
            current_loss = float(loss.item())
            batch_size = int(batch['captions'].shape[0])
            current_step = it + 1
            global_step = epoch * total_steps + current_step
            running_loss += current_loss
            processed_samples += batch_size
            window_loss += current_loss
            window_samples += batch_size
            window_steps += 1
            should_trace = trace_every_n_steps > 0 and (current_step == 1 or current_step % trace_every_n_steps == 0 or current_step == total_steps)
            should_log_step = step_metrics_every_n_steps > 0 and (current_step == 1 or current_step % step_metrics_every_n_steps == 0 or current_step == total_steps)
            if should_trace or should_log_step:
                now = time.perf_counter()
                elapsed_epoch_sec = now - start_time
                elapsed_step_sec = now - step_started_at
                avg_step_time_sec = elapsed_epoch_sec / current_step if current_step > 0 else 0.0
                window_duration_sec = max(now - window_started_at, 1e-09)
                step_payload = {'epoch': epoch + 1, 'step': current_step, 'global_step': global_step, 'total_steps': total_steps, 'total_run_steps': total_run_steps, 'processed_samples': processed_samples, 'epoch_progress': round(current_step / total_steps, 6), 'run_progress': round(global_step / total_run_steps, 6), 'batch_loss': round(current_loss, 6), 'running_loss': round(running_loss / current_step, 6), 'window_loss': round(window_loss / window_steps, 6), 'lr': float(optimizers['model'].param_groups[0]['lr']), 'elapsed_epoch_sec': round(elapsed_epoch_sec, 6), 'elapsed_step_sec': round(elapsed_step_sec, 6), 'avg_step_time_sec': round(avg_step_time_sec, 6), 'window_step_time_sec': round(window_duration_sec / window_steps, 6), 'window_steps_per_sec': round(window_steps / window_duration_sec, 6), 'window_samples_per_sec': round(window_samples / window_duration_sec, 6), 'eta_epoch_sec': round(max(total_steps - current_step, 0) * avg_step_time_sec, 3), 'eta_run_sec': round(max(total_run_steps - global_step, 0) * avg_step_time_sec, 3), **get_gpu_memory_snapshot(device)}
                if enable_nvidia_smi_metrics:
                    step_payload.update(get_nvidia_smi_snapshot())
                if log_gradient_norm:
                    step_payload.update(compute_grad_norm(model.parameters()))
                alpha_rel = aux.get('alpha_rel')
                if alpha_rel is not None:
                    step_payload['ext_alpha_rel'] = round(alpha_rel, 6)
                if observer is not None and should_trace:
                    observer.event('train_step', **step_payload)
                if should_log_step:
                    if tracker is not None:
                        tracker.log_metrics({f'step/{key}': value for key, value in step_payload.items()}, step=global_step)
                    if live_metric_recorder is not None:
                        live_metric_recorder.record({'ts': iso_timestamp(), **step_payload})
                    window_started_at = now
                    window_samples = 0
                    window_steps = 0
                    window_loss = 0.0
            pbar.set_postfix(loss=running_loss / current_step)
            pbar.update()
    epoch_duration_sec = time.perf_counter() - start_time
    train_loss = running_loss / max(total_steps, 1)
    throughput_samples_per_sec = processed_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0
    throughput_steps_per_sec = total_steps / epoch_duration_sec if epoch_duration_sec > 0 else 0.0
    if _l_loc_batches > 0:
        loc_hit = _l_loc_batches - _l_loc_skipped
        loc_pct = 100.0 * loc_hit / _l_loc_batches
        print(f'[LCQM] Epoch {epoch + 1} L_loc: {loc_hit}/{_l_loc_batches} batches ({loc_pct:.1f}%) có phrase positives')
        if loc_hit == 0:
            print(f'[LCQM] ERROR: L_loc được bật (lambda_loc={lambda_loc}) nhưng 0/{_l_loc_batches} batch có positive phrase. positives_mask có {len(positives_mask)} images — kiểm tra align giữa image_ids trong dataloader và key trong positives_mask.')
    if _l_route_batches > 0:
        route_hit = _l_route_batches - _l_route_skipped
        route_pct = 100.0 * route_hit / _l_route_batches
        print(f'[LCQM] Epoch {epoch + 1} L_route: {route_hit}/{_l_route_batches} batches ({route_pct:.1f}%) có cultural tokens')
        if route_hit == 0:
            print(f'[LCQM] ERROR: L_route được bật (lambda_route={lambda_route}) nhưng 0/{_l_route_batches} batch có cultural token. token_masks có {len(token_masks)} images — kiểm tra align.')
    if _gate_batches > 0:
        gate_mean = _gate_openness_sum / _gate_batches
        m = getattr(model, 'module', model)
        offset_val = 0.0
        bias_val = -5.0
        if hasattr(m, 'cap_generator') and hasattr(m.cap_generator, 'layers'):
            layer0 = m.cap_generator.layers[0]
            if hasattr(layer0, 'gate_bias_offset'):
                offset_val = layer0.gate_bias_offset.item()
            if hasattr(layer0, 'fc_alpha4'):
                bias_val = layer0.fc_alpha4.bias.data.mean().item()
        print(f'[GATE] Epoch {epoch + 1}: α4_mean={gate_mean:.4f} | fc_alpha4.bias={bias_val:.2f} | offset={offset_val:.3f} | effective_σ≈{torch.sigmoid(torch.tensor(bias_val + offset_val)).item():.3f}')
        if epoch >= 3 and gate_mean < 0.05:
            print(f'[GATE] WARN: α4 vẫn < 0.05 sau epoch {epoch + 1} — gate chưa mở. Cân nhắc: tăng ext_lr, giảm gate_bias_init, hoặc bật heuristic bias.')
    if use_heuristic_bias and epoch >= warmup_epochs and (_gate_batches > 0):
        gate_mean_epoch = _gate_openness_sum / _gate_batches
        m = getattr(model, 'module', model)
        if hasattr(m, 'cap_generator') and hasattr(m.cap_generator, 'layers'):
            for layer in m.cap_generator.layers:
                if hasattr(layer, 'gate_bias_offset'):
                    if gate_mean_epoch < gate_target_openness:
                        layer.gate_bias_offset.data += heuristic_step_size
                    else:
                        layer.gate_bias_offset.data -= heuristic_step_size * 0.1
                    layer.gate_bias_offset.data.clamp_(-4.0, 4.0)
            new_offset = m.cap_generator.layers[0].gate_bias_offset.item()
            new_bias_total = m.cap_generator.layers[0].fc_alpha4.bias.data.item() + new_offset
            print(f'[GATE] Heuristic update: offset={new_offset:.3f}, effective_bias={new_bias_total:.2f} (σ≈{torch.sigmoid(torch.tensor(new_bias_total)).item():.3f})')
    return {'train_loss': train_loss, 'epoch_duration_sec': epoch_duration_sec, 'throughput_samples_per_sec': throughput_samples_per_sec, 'throughput_steps_per_sec': throughput_steps_per_sec, **get_gpu_memory_snapshot(device)}

def train_sc(model, dataloaders, optimizers, epoch, observer=None, tracker=None, config=None):
    model.train()
    device = next(model.parameters()).device
    seq_len = int(config.model.beam_len)
    beam_size = int(config.model.beam_size)
    cider = metrics.Cider()
    running_loss = 0.0
    running_reward = 0.0
    running_reward_baseline = 0.0
    processed_samples = 0
    start_time = time.perf_counter()
    with tqdm(desc=f'Epoch {epoch + 1} - scst', unit='it', total=len(dataloaders['train_dict'])) as pbar:
        for it, batch in enumerate(dataloaders['train_dict']):
            if 'samples' in batch and (not isinstance(batch['samples'], dict)):
                batch_size = batch['samples'].tensors.shape[0]
            else:
                batch_size = batch['samples']['reg_feat'].shape[0]
            zero_optimizers(optimizers)
            outs, log_probs = model(batch['samples'], seq=None, use_beam_search=True, max_len=config.model.beam_len, eos_idx=config.model.eos_idx, beam_size=config.model.beam_size, out_size=beam_size, return_probs=False)
            vocab = resolve_vocab_from_dataset(dataloaders['train'].dataset)
            flat_predictions = outs.view(-1, seq_len)
            decoded_predictions = []
            for row in flat_predictions:
                decoded_predictions.append(detokenize_prediction(row, vocab, int(config.model.eos_idx))['prediction'])
            repeated_references = []
            for references in batch['captions']:
                repeated_references.extend([references] * beam_size)
            tokenized_predictions = metrics.PTBTokenizer.tokenize({str(idx): [caption] for idx, caption in enumerate(decoded_predictions)})
            tokenized_references = metrics.PTBTokenizer.tokenize({str(idx): refs for idx, refs in enumerate(repeated_references)})
            reward_array = cider.compute_score(tokenized_references, tokenized_predictions)[1].astype(np.float32)
            reward = torch.from_numpy(reward_array).to(device).view(batch_size, beam_size)
            reward_baseline = torch.mean(reward, dim=-1, keepdim=True)
            loss = -torch.mean(log_probs, dim=-1) * (reward - reward_baseline)
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite SCST loss detected at epoch={epoch + 1}, step={it + 1}: {loss.item()}')
            loss.backward()
            clip_trainable_gradients(model, config)
            step_optimizers(optimizers)
            loss = gather_result(loss)
            reward_mean = gather_result(reward.mean())
            reward_baseline_mean = gather_result(reward_baseline.mean())
            running_loss += float(loss.item())
            running_reward += float(reward_mean.item())
            running_reward_baseline += float(reward_baseline_mean.item())
            processed_samples += int(batch_size)
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1), reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
    epoch_duration_sec = time.perf_counter() - start_time
    total_steps = max(len(dataloaders['train_dict']), 1)
    return {'train_loss': running_loss / total_steps, 'train_reward': running_reward / total_steps, 'reward_baseline': running_reward_baseline / total_steps, 'epoch_duration_sec': epoch_duration_sec, 'throughput_samples_per_sec': processed_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0, 'throughput_steps_per_sec': total_steps / epoch_duration_sec if epoch_duration_sec > 0 else 0.0, **get_gpu_memory_snapshot(device)}

def _lcqm_smoke_forward(config):
    lcqm_enabled = hasattr(config, 'model_ext') and hasattr(config.model_ext, 'lcqm') and getattr(config.model_ext.lcqm, 'enabled', False)
    if not lcqm_enabled:
        return {'enabled': False}
    try:
        import torch
        from models.caption.cultural_memory import CulturalQueryMemory
        from types import SimpleNamespace
        lcqm_cfg = SimpleNamespace(d_model=getattr(config.model, 'd_model', 512), **{k: v for k, v in config.model_ext.lcqm.__dict__.items() if not k.startswith('_')})
        lcqm = CulturalQueryMemory(lcqm_cfg)
        lcqm.eval()
        B, N, M = (2, 150, 60)
        d = lcqm_cfg.d_model
        reg = torch.randn(B, N, d)
        gri = torch.randn(B, M, d)
        mask = torch.zeros(B, 1, 1, M, dtype=torch.bool)
        with torch.no_grad():
            C_loc, C_proj = lcqm(reg, gri, mask)
        return {'enabled': True, 'passed': not torch.isnan(C_loc).any(), 'c_loc_shape': list(C_loc.shape), 'c_loc_mean_norm': round(float(C_loc.norm(dim=-1).mean()), 4), 'k_loc': getattr(config.model_ext.lcqm, 'k_loc', 16)}
    except Exception as exc:
        return {'enabled': True, 'passed': False, 'error': str(exc), 'error_type': type(exc).__name__}

def build_preflight_report(config):
    ensure_train_vocab_source()
    datasets = get_datasets()
    train_sample = datasets['train'][0]
    dataset_contract = collect_dataset_contract()
    for split_name in ('train', 'valid', 'test'):
        summary = dataset_contract.get(f'{split_name}_summary', {})
        missing = summary.get('missing_image_count', 0)
        if missing > 0:
            preview = summary.get('missing_images_preview', [])
            raise RuntimeError(f"Preflight FAILED: {missing} image(s) missing from {split_name} split. image_root={summary.get('image_root')}. First missing: {preview[:3]}. Check KTVIC_{split_name.upper()}_IMAGES env var or dataset path.")
    paths = resolve_ktvic_paths()
    train_data = json.load(open(paths['train_json'], 'r', encoding='utf-8'))
    missing_segment_caption = sum((1 for ann in train_data['annotations'] if not ann.get(CAPTION_FIELD)))
    return {'checked_at': iso_timestamp(), 'config_name': str(config.exp.name), 'resolved_paths': dataset_contract['resolved_paths'], 'dataset_contract': dataset_contract, 'checkpoint': checkpoint_probe(config.exp.checkpoint), 'system': collect_system_info(), 'train_sample': {'image_shape': list(train_sample[0].shape), 'token_count': int(train_sample[1].shape[0]), 'caption_preview': train_sample[2]}, 'vocab_source_exists': Path(paths['vocab_source']).exists(), 'vocab_size': len(datasets['train'].vocab), 'train_dataset_len': len(datasets['train']), 'valid_dataset_len': len(datasets['valid']), 'test_dataset_len': len(datasets['test']), 'caption_field': CAPTION_FIELD, 'raw_caption_field': RAW_CAPTION_FIELD, 'tokenizer_backend': TOKENIZER_BACKEND, 'missing_segment_caption_in_train': int(missing_segment_caption), 'lcqm_smoke': _lcqm_smoke_forward(config)}

def run_preflight(config: DictConfig) -> None:
    observer = RunObserver(config, rank=0)
    observer.event('preflight_started')
    try:
        report = build_preflight_report(config)
        observer.write_preflight(report)
        observer.update_status('preflight_ok', preflight=report)
        observer.event('preflight_completed', train_dataset_len=report['train_dataset_len'], test_dataset_len=report['test_dataset_len'])
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as exc:
        observer.capture_exception(exc)
        observer.update_status('preflight_failed', error=str(exc))
        raise

def main(gpu, config):
    torch.backends.cudnn.enabled = False
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    use_ddp = int(config.exp.world_size) > 1
    if use_ddp:
        dist.init_process_group('nccl', 'env://', rank=rank, world_size=config.exp.world_size)
    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available - check GPU/MPS allocation')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(gpu)
    observer = RunObserver(config, rank)
    tracker = None
    live_metric_recorder = None
    runtime_alerts = []
    train_loss_history = []
    result_rows = []
    start_epoch = 0
    resume_state = None
    try:
        observer.event('ddp_initialized', rank=rank, world_size=config.exp.world_size, gpu=gpu, device=str(device))
        print('Building dataloaders...')
        transform_cfg = config.dataset.transform_cfg
        samplers, dataloaders = get_dataloaders(device=device, batch_size=config.optimizer.batch_size, num_workers=config.optimizer.num_workers, resize_name=str(transform_cfg.resize_name), size=tuple((int(item) for item in transform_cfg.size)), randaug=bool(transform_cfg.randaug), max_train_items=int(getattr(config.dataset, 'max_train_items', 0)), max_valid_items=int(getattr(config.dataset, 'max_valid_items', 0)), max_test_items=int(getattr(config.dataset, 'max_test_items', 0)), split_mode=str(getattr(config.dataset, 'split_mode', 'full')), dev_split=str(getattr(config.dataset, 'dev_split', 'seed42_valid10')))
        if isinstance(dataloaders['valid_dict'].sampler, DistributedSampler):
            raise RuntimeError('Validation evaluation must not use DistributedSampler; full-split eval is required.')
        if isinstance(dataloaders['test_dict'].sampler, DistributedSampler):
            raise RuntimeError('Official-test evaluation must not use DistributedSampler; full-split eval is required.')
        probe_loaders = build_sample_probe_loaders(dataloaders, config) if rank == 0 else {}
        observer.event('dataloaders_ready', dataset_sizes={key: len(loader.dataset) for key, loader in dataloaders.items()}, batch_size=int(config.optimizer.batch_size), num_workers=int(config.optimizer.num_workers))
        print('Building model...')
        detector = build_detector(config).to(device)
        model = Transformer(detector=detector, config=config)
        model.cached_features = False
        model = model.to(device)
        if config.optimizer.freeze_detector:
            for param in model.detector.parameters():
                param.requires_grad = False
        if use_ddp:
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
        optimizers = build_optimizers(model, config, mode='xe')
        scheduler = CosineLRScheduler(optimizers['model'], num_epochs=config.optimizer.finetune_xe_epochs, num_its_per_epoch=len(dataloaders['train']), init_lr=config.optimizer.xe_lr, min_lr=config.optimizer.min_lr, warmup_init_lr=config.optimizer.warmup_init_lr)
        if as_bool(config.exp.resume):
            resume_checkpoint_path = resolve_resume_checkpoint_path(config)
            if resume_checkpoint_path is None or not resume_checkpoint_path.exists():
                raise RuntimeError('Resume requested but no recovery checkpoint was found. Set exp.resume_checkpoint or enable exp.auto_resume_latest in the same run directory.')
            resume_state = load_training_state_from_resume(model, optimizers, scheduler, resume_checkpoint_path)
            start_epoch = int(resume_state['next_epoch'])
            train_loss_history = resume_state['train_loss_history']
            result_rows = resume_state['result_rows']
            resume_best_valid_cider = resume_state.get('best_valid_cider')
            resume_best_valid_epoch = resume_state.get('best_valid_epoch')
            resume_best_valid_scores = resume_state.get('best_valid_scores')
            observer.event('resume_loaded', checkpoint=resume_state['path'], completed_epoch=resume_state['completed_epoch'], next_epoch=start_epoch + 1 if start_epoch < int(config.optimizer.finetune_xe_epochs) else start_epoch, best_valid_epoch=resume_best_valid_epoch, best_valid_cider=resume_best_valid_cider, missing=resume_state['missing'], unexpected=resume_state['unexpected'])
        elif os.path.exists(config.exp.checkpoint):
            checkpoint = load_trusted_checkpoint(config.exp.checkpoint, map_location='cpu')
            model_without_ddp = getattr(model, 'module', model)
            missing, unexpected = model_without_ddp.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
            print(f'caption checkpoint load -> missing: {len(missing)}, unexpected: {len(unexpected)}')
            observer.event('checkpoint_loaded', checkpoint=str(config.exp.checkpoint), missing=len(missing), unexpected=len(unexpected))
            if len(missing) > int(config.tracking.checkpoint_missing_key_alert_threshold):
                runtime_alerts.append(build_alert('checkpoint_missing', 'warning', f'Checkpoint load missing keys: {len(missing)}.', count=len(missing)))
            if len(unexpected) > int(config.tracking.checkpoint_unexpected_key_alert_threshold):
                raise RuntimeError(f'Checkpoint load produced {len(unexpected)} unexpected key(s) (> threshold {config.tracking.checkpoint_unexpected_key_alert_threshold}); aborting to avoid a silent partial load. First few: {list(unexpected)[:8]}')
        else:
            runtime_alerts.append(build_alert('checkpoint_missing', 'critical', f'Checkpoint path does not exist: {config.exp.checkpoint}.'))
        if as_bool(getattr(config.exp, 'eval', False)):
            eval_checkpoint_path = as_non_empty_string(getattr(config.exp, 'eval_checkpoint', '')) or (str(resolve_resume_checkpoint_path(config)) if as_bool(config.exp.resume) and resolve_resume_checkpoint_path(config) else str(config.exp.checkpoint))
            if not eval_checkpoint_path or not Path(eval_checkpoint_path).exists():
                raise RuntimeError(f'Eval-only requested but checkpoint is unavailable: {eval_checkpoint_path}')
            checkpoint = load_trusted_checkpoint(eval_checkpoint_path, map_location='cpu')
            model_without_ddp = getattr(model, 'module', model)
            missing, unexpected = model_without_ddp.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
            observer.event('eval_checkpoint_loaded', checkpoint=str(eval_checkpoint_path), missing=len(missing), unexpected=len(unexpected))
            if len(unexpected) > 0:
                raise RuntimeError(f'Eval-only checkpoint produced {len(unexpected)} unexpected key(s) at {eval_checkpoint_path}; aborting to avoid evaluating a mismatched model. First few: {list(unexpected)[:8]}')
            if len(missing) > 0:
                print(f'WARNING: eval-only checkpoint missing {len(missing)} key(s) -> those params stay at init. First few: {list(missing)[:8]}')
            start_epoch = int(config.optimizer.finetune_xe_epochs)
        tracker = ExperimentTracker(config, rank)
        live_metric_recorder = LiveMetricRecorder(enabled=rank == 0 and int(getattr(config.tracking, 'step_metrics_every_n_steps', 0)) > 0)
        if rank == 0:
            tracker.write_tracking_info(Path('trace') / 'tracking_info.json')
            observer.add_artifact('tracking_info', 'trace/tracking_info.json')
            observer.add_artifact('step_metrics_jsonl', 'trace/step_metrics.jsonl')
            observer.add_artifact('step_metrics_csv', 'trace/step_metrics.csv')
            observer.add_artifact('live_status', 'trace/live_status.json')
            tracker.log_artifact('trace/resolved_config.json')
            tracker.log_artifact('trace/resolved_config.yaml')
            tracker.log_artifact('trace/system_info.json')
            tracker.log_artifact('trace/tracking_info.json')
            preflight_report_path = Path('..') / 'preflight' / 'trace' / 'preflight_report.json'
            if preflight_report_path.exists():
                tracker.log_artifact(preflight_report_path)
            tracker.log_params({'seed': int(config.exp.seed), 'world_size': int(config.exp.world_size), 'batch_size': int(config.optimizer.batch_size), 'num_workers': int(config.optimizer.num_workers), 'finetune_xe_epochs': int(config.optimizer.finetune_xe_epochs), 'finetune_sc_epochs': int(config.optimizer.finetune_sc_epochs), 'xe_lr': float(config.optimizer.xe_lr), 'min_lr': float(config.optimizer.min_lr), 'warmup_init_lr': float(config.optimizer.warmup_init_lr), 'beam_size': int(config.model.beam_size), 'beam_len': int(config.model.beam_len), 'freeze_detector': bool(config.optimizer.freeze_detector), 'freeze_backbone': bool(config.optimizer.freeze_backbone), 'checkpoint_path': str(config.exp.checkpoint), 'resume_enabled': bool(as_bool(config.exp.resume)), 'resume_checkpoint': str(resolve_resume_checkpoint_path(config)) if resolve_resume_checkpoint_path(config) else '', 'eval_only': bool(as_bool(getattr(config.exp, 'eval', False))), 'dataset_root': os.environ.get('DATA_ROOT'), 'caption_field': CAPTION_FIELD, 'tokenizer_backend': TOKENIZER_BACKEND, 'train_dataset_len': len(dataloaders['train'].dataset), 'valid_dataset_len': len(dataloaders['valid'].dataset), 'max_train_items': int(getattr(config.dataset, 'max_train_items', 0)), 'max_valid_items': int(getattr(config.dataset, 'max_valid_items', 0)), 'max_test_items': int(getattr(config.dataset, 'max_test_items', 0)), 'train_dict_len': len(dataloaders['train_dict'].dataset), 'valid_dict_len': len(dataloaders['valid_dict'].dataset), 'test_dict_len': len(dataloaders['test_dict'].dataset), 'phase_plan': str(getattr(config.exp, 'phase_plan', 'xe_only')), 'selection_split': str(getattr(config.exp, 'selection_split', 'valid')), 'step_metrics_every_n_steps': int(getattr(config.tracking, 'step_metrics_every_n_steps', 0)), 'system_metrics_sampling_interval_sec': int(getattr(config.tracking, 'system_metrics_sampling_interval_sec', 15)), 'tracking_run_name': tracker.run_name})
            tracker.set_tags({'caption_field': CAPTION_FIELD, 'tokenizer_backend': TOKENIZER_BACKEND, 'run_type': 'ktvic_baseline3', 'checkpoint_file': Path(str(config.exp.checkpoint)).name, 'hostname': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'hydra_run_dir': str(config.exp.run_dir), 'resume_mode': 'enabled' if as_bool(config.exp.resume) else 'disabled', 'eval_mode': 'eval_only' if as_bool(getattr(config.exp, 'eval', False)) else 'train_and_eval', 'tracking_backend_mode': 'file-only'})
        if rank == 0:
            rewrite_result_csv(result_rows)
            observer.add_artifact('epoch_metrics_csv', 'result.csv')
            observer.update_status('running')
        if as_bool(getattr(config.exp, 'eval', False)):
            if rank == 0:
                valid_scores, _, _ = evaluate_named_split(model=model, dataloader=dataloaders['valid_dict'], config=config, split_name='valid_main', epoch=max(int(config.optimizer.finetune_xe_epochs), 1), observer=observer, tracker=tracker)
                test_scores, test_rows, _ = evaluate_named_split(model=model, dataloader=dataloaders['test_dict'], config=config, split_name='test', epoch=max(int(config.optimizer.finetune_xe_epochs), 1), observer=observer, tracker=tracker)
                write_json('valid_main_scores_best.json', valid_scores)
                write_json('test_scores_final.json', test_scores)
                observer.add_artifact('valid_main_scores_best', 'valid_main_scores_best.json')
                observer.add_artifact('test_scores', 'test_scores_final.json')
                observer.update_status('completed', valid_scores=valid_scores, test_scores=test_scores)
                observer.event('evaluation_completed', valid_cider=valid_scores['CIDEr'], test_cider=test_scores['CIDEr'], prediction_rows=len(test_rows))
            return
        phase_plan = str(getattr(config.exp, 'phase_plan', 'xe_only'))
        phase_entries = []
        if phase_plan == 'xe_only':
            phase_entries.append({'name': 'xe', 'epochs': int(config.optimizer.finetune_xe_epochs), 'freeze_detector': False, 'freeze_backbone': False})
        elif phase_plan == 'freeze_then_xe':
            if int(config.optimizer.freezing_xe_epochs) > 0:
                phase_entries.append({'name': 'freeze_xe', 'epochs': int(config.optimizer.freezing_xe_epochs), 'freeze_detector': bool(config.optimizer.freeze_detector), 'freeze_backbone': bool(config.optimizer.freeze_backbone)})
            phase_entries.append({'name': 'xe', 'epochs': int(config.optimizer.finetune_xe_epochs), 'freeze_detector': False, 'freeze_backbone': False})
        elif phase_plan == 'scst_only':
            phase_entries.append({'name': 'scst', 'epochs': int(config.optimizer.finetune_sc_epochs), 'freeze_detector': False, 'freeze_backbone': False})
        else:
            raise RuntimeError(f'Unsupported phase_plan for train.py
        best_valid_cider = -1.0
        best_valid_epoch = None
        best_valid_scores = None
        if resume_state:
            if resume_state.get('best_valid_cider') is not None:
                best_valid_cider = float(resume_state['best_valid_cider'])
            if resume_state.get('best_valid_epoch') is not None:
                best_valid_epoch = int(resume_state['best_valid_epoch'])
            if resume_state.get('best_valid_scores') is not None:
                best_valid_scores = dict(resume_state['best_valid_scores'])
        best_valid_checkpoint_name = 'checkpoint_best_valid_scst.pth' if phase_plan == 'scst_only' else 'checkpoint_best_valid_xe.pth'
        xe_seed_checkpoint = as_non_empty_string(getattr(config.exp, 'scst_seed_checkpoint', ''))
        for phase_entry in phase_entries:
            phase_name = phase_entry['name']
            phase_epochs = int(phase_entry['epochs'])
            if phase_epochs <= 0:
                continue
            if phase_name == 'scst':
                seed_checkpoint_path = Path(xe_seed_checkpoint) if xe_seed_checkpoint else None
                if seed_checkpoint_path is None or not seed_checkpoint_path.exists():
                    raise RuntimeError('SCST phase requires exp.scst_seed_checkpoint pointing to the best XE validation checkpoint.')
                checkpoint, missing, unexpected = load_model_checkpoint_into_model(model, seed_checkpoint_path)
                observer.event('scst_seed_loaded', checkpoint=str(seed_checkpoint_path), missing=len(missing), unexpected=len(unexpected))
                set_trainability(model, freeze_detector=False, freeze_backbone=False)
                optimizers = build_optimizers(model, config, mode='sc')
                scheduler = None
            else:
                set_trainability(model, freeze_detector=phase_entry['freeze_detector'], freeze_backbone=phase_entry['freeze_backbone'], freeze_grit=bool(getattr(config.optimizer, 'freeze_grit', False)))
                optimizers = build_optimizers(model, config, mode='xe')
                scheduler = CosineLRScheduler(optimizers['model'], num_epochs=phase_epochs, num_its_per_epoch=len(dataloaders['train']), init_lr=config.optimizer.xe_lr, min_lr=config.optimizer.min_lr, warmup_init_lr=config.optimizer.warmup_init_lr)
            phase_completed_epochs = sum((1 for row in result_rows if row.get('phase') == phase_name))
            if phase_completed_epochs >= phase_epochs:
                observer.event('phase_resume_skip_completed', phase=phase_name, completed_epochs=phase_completed_epochs, configured_epochs=phase_epochs)
                continue
            if resume_state and resume_state.get('phase_name') == phase_name:
                restored_optimizer = restore_optimizer_state_from_resume(optimizers, scheduler, resume_state)
                observer.event('resume_optimizer_state_loaded', phase=phase_name, restored=restored_optimizer, checkpoint=resume_state['path'])
            for local_epoch in range(phase_completed_epochs, phase_epochs):
                epoch_index = len(result_rows)
                observer.event('epoch_started', epoch=epoch_index + 1, phase=phase_name)
                if phase_name == 'scst':
                    epoch_stats = train_sc(model=model, dataloaders=dataloaders, optimizers=optimizers, epoch=epoch_index, observer=observer, tracker=tracker, config=config)
                else:
                    epoch_stats = train_xe(model=model, dataloaders=dataloaders, optimizers=optimizers, epoch=epoch_index, total_epochs=phase_epochs, trace_every_n_steps=int(config.exp.trace_every_n_steps), step_metrics_every_n_steps=int(getattr(config.tracking, 'step_metrics_every_n_steps', 0)), observer=observer, tracker=tracker, live_metric_recorder=live_metric_recorder, scheduler=scheduler, enable_nvidia_smi_metrics=as_bool(getattr(config.tracking, 'enable_nvidia_smi_metrics', True)), log_gradient_norm=as_bool(getattr(config.tracking, 'log_gradient_norm', True)), max_train_batches=int(getattr(config.exp, 'max_train_batches', 0)), config=config)
                    if samplers['train'] is not None:
                        samplers['train'].set_epoch(epoch_index)
                if rank == 0:
                    lr_now = float(optimizers['model'].param_groups[0]['lr'])
                    row_payload = {'epoch': epoch_index + 1, 'phase': phase_name, 'train_loss': round(epoch_stats['train_loss'], 6), 'lr': round(lr_now, 8), 'epoch_duration_sec': round(epoch_stats['epoch_duration_sec'], 6), 'throughput_samples_per_sec': round(epoch_stats['throughput_samples_per_sec'], 6), 'throughput_steps_per_sec': round(epoch_stats['throughput_steps_per_sec'], 6), 'gpu_memory_allocated_mb': round(epoch_stats['gpu_memory_allocated_mb'], 3), 'gpu_memory_reserved_mb': round(epoch_stats['gpu_memory_reserved_mb'], 3), 'gpu_max_memory_allocated_mb': round(epoch_stats['gpu_max_memory_allocated_mb'], 3), 'gpu_max_memory_reserved_mb': round(epoch_stats['gpu_max_memory_reserved_mb'], 3), 'gpu_total_memory_mb': round(epoch_stats['gpu_total_memory_mb'], 3), 'gpu_max_memory_utilization': round(epoch_stats['gpu_max_memory_utilization'], 6)}
                    if 'train_reward' in epoch_stats:
                        row_payload['train_reward'] = round(epoch_stats['train_reward'], 6)
                        row_payload['reward_baseline'] = round(epoch_stats['reward_baseline'], 6)
                    result_rows.append(row_payload)
                    train_loss_history.append(epoch_stats['train_loss'])
                    rewrite_result_csv(result_rows)
                    recovery_epoch_path, recovery_latest_path, recovery_manifest_path = write_recovery_checkpoint(model=model, optimizers=optimizers, scheduler=scheduler, epoch=epoch_index + 1, train_loss_history=train_loss_history, result_rows=result_rows, phase_name=phase_name, best_valid_epoch=best_valid_epoch, best_valid_cider=best_valid_cider, best_valid_scores=best_valid_scores)
                    observer.add_artifact(f'recovery_epoch_{epoch_index + 1:02d}', str(recovery_epoch_path))
                    observer.add_artifact('recovery_latest', str(recovery_latest_path))
                    observer.add_artifact('recovery_manifest', str(recovery_manifest_path))
                    tracker.log_artifact(recovery_manifest_path)
                    if as_bool(getattr(config.exp, 'stop_after_train_epoch', False)):
                        observer.event('stop_after_train_epoch', epoch=epoch_index + 1, phase=phase_name)
                        observer.update_status('completed', stopped_after_train_epoch=epoch_index + 1, phase=phase_name)
                        return
                    valid_scores, _, _ = evaluate_named_split(model=model, dataloader=dataloaders['valid_dict'], config=config, split_name='valid_main', epoch=epoch_index + 1, observer=observer, tracker=tracker)
                    write_json('valid_main_scores_latest.json', valid_scores)
                    observer.add_artifact('valid_main_scores_latest', 'valid_main_scores_latest.json')
                    tracker.log_artifact('valid_main_scores_latest.json')
                    if float(valid_scores['CIDEr']) >= float(best_valid_cider):
                        best_valid_cider = float(valid_scores['CIDEr'])
                        best_valid_epoch = epoch_index + 1
                        best_valid_scores = dict(valid_scores)
                        write_json('valid_main_scores_best.json', best_valid_scores)
                        observer.add_artifact('valid_main_scores_best', 'valid_main_scores_best.json')
                        tracker.log_artifact('valid_main_scores_best.json')
                        best_checkpoint_path = write_model_checkpoint(model, Path('checkpoints') / best_valid_checkpoint_name, {'epoch': best_valid_epoch, 'phase': phase_name, 'selection_split': 'valid_main', 'selection_metric': 'CIDEr', 'selection_score': best_valid_cider})
                        observer.add_artifact('best_valid_checkpoint', str(best_checkpoint_path))
                        tracker.log_artifact(best_checkpoint_path)
                    sample_rows, _, _ = capture_sample_caption_tables(model=model, probe_loaders=probe_loaders, config=config, epoch=epoch_index + 1, observer=observer, tracker=tracker)
                    runtime_alerts.extend(build_sample_alerts(sample_rows, config))
                    observer.event('epoch_completed', epoch=epoch_index + 1, phase=phase_name, valid_cider=valid_scores['CIDEr'])
                    write_recovery_checkpoint(model=model, optimizers=optimizers, scheduler=scheduler, epoch=epoch_index + 1, train_loss_history=train_loss_history, result_rows=result_rows, phase_name=phase_name, best_valid_epoch=best_valid_epoch, best_valid_cider=best_valid_cider, best_valid_scores=best_valid_scores)
                if dist.is_initialized():
                    torch.distributed.barrier()
        if rank == 0:
            final_checkpoint_name = 'model_stage2_scst_final.pth' if phase_plan == 'scst_only' else 'model_stage1_xe_final.pth'
            write_model_checkpoint(model, final_checkpoint_name)
            observer.add_artifact('final_checkpoint', final_checkpoint_name)
            if best_valid_scores is None:
                raise RuntimeError('No best validation checkpoint was recorded.')
            test_scores, test_rows, _ = evaluate_named_split(model=model, dataloader=dataloaders['test_dict'], config=config, split_name='test', epoch=max(len(result_rows), 1), observer=observer, tracker=tracker)
            write_json('test_scores_final.json', test_scores)
            observer.add_artifact('test_scores', 'test_scores_final.json')
            tracker.log_artifact('test_scores_final.json')
            tracker.log_artifact('result.csv')
            tracker.log_artifact('trace/step_metrics.jsonl')
            tracker.log_artifact('trace/step_metrics.csv')
            tracker.log_artifact('trace/live_status.json')
            runtime_alert_summary = {'alert_count': len(runtime_alerts), 'alerts': runtime_alerts, 'best_valid_epoch': best_valid_epoch, 'best_valid_cider': best_valid_cider, 'phase_plan': phase_plan}
            write_json('alert_summary_runtime.json', runtime_alert_summary)
            observer.add_artifact('runtime_alert_summary', 'alert_summary_runtime.json')
            tracker.log_artifact('alert_summary_runtime.json')
            observer.update_status('completed', best_valid_epoch=best_valid_epoch, best_valid_scores=best_valid_scores, test_scores=test_scores)
            observer.event('evaluation_completed', best_valid_epoch=best_valid_epoch, best_valid_cider=best_valid_cider, test_cider=test_scores['CIDEr'], prediction_rows=len(test_rows), alert_count=len(runtime_alerts))
    except Exception as exc:
        runtime_alerts.append(build_alert('train_runtime_failure', 'critical', str(exc)))
        if rank == 0:
            write_json('alert_summary_runtime.json', {'alert_count': len(runtime_alerts), 'alerts': runtime_alerts})
        observer.capture_exception(exc)
        observer.update_status('failed', error=str(exc))
        raise
    finally:
        if tracker is not None:
            tracker.finish()
        if dist.is_initialized():
            torch.distributed.barrier()

def run_main(config: DictConfig) -> None:
    if as_bool(config.exp.preflight_only):
        run_preflight(config)
        return
    if int(config.exp.ngpus_per_node) == 1:
        main(0, config)
        return
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))
if __name__ == '__main__':
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '6688')
    import sys as _sys, traceback as _tb
    try:
        from omegaconf import OmegaConf
        print('OmegaConf imported', flush=True)
        _cfg = OmegaConf.load('configs/caption/default_config.yaml')
        print('config loaded', flush=True)
        _dotlist = []
        for _arg in _sys.argv[1:]:
            if '=' in _arg:
                _k, _v = _arg.split('=', 1)
                if _k == 'hydra.run.dir':
                    _cfg.exp.run_dir = _v
                elif _v == '':
                    OmegaConf.update(_cfg, _k, '')
                else:
                    _dotlist.append(_arg)
        if _dotlist:
            _overrides = OmegaConf.from_dotlist(_dotlist)
            _cfg = OmegaConf.merge(_cfg, _overrides)
        print('overrides applied', flush=True)
        import os as _os
        _stage1_root = _os.environ.get('STAGE1_RUN_ROOT', '')
        if _stage1_root:
            _ckpt_candidates = [_os.path.join(_stage1_root, 'train', 'checkpoints', 'checkpoint_best_valid_xe.pth'), _os.path.join(_stage1_root, 'checkpoints', 'checkpoint_best_valid_xe.pth'), _os.path.join(_stage1_root, 'train', 'checkpoints', 'recovery_latest.pth'), _os.path.join(_stage1_root, 'checkpoints', 'recovery_latest.pth')]
            for _ckpt in _ckpt_candidates:
                if _os.path.isfile(_ckpt):
                    _cfg.exp.checkpoint = _ckpt
                    print(f'stage_b checkpoint: {_ckpt} (from STAGE1_RUN_ROOT={_stage1_root})', flush=True)
                    break
            else:
                print(f'WARN: no checkpoint found under {_stage1_root}', flush=True)
        os.makedirs(str(_cfg.exp.run_dir), exist_ok=True)
        os.chdir(str(_cfg.exp.run_dir))
        print('calling run_main', flush=True)
        run_main(_cfg)
    except Exception as e:
        print(f'FATAL: {e}', flush=True)
        _tb.print_exc()
