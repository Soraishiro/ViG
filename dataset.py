import json
import os
from hashlib import sha256
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import DistributedSampler
from engine.utils import nested_tensor_from_tensor_list
from datasets.caption.transforms import RESIZE, MEAN, STD, get_transform, normalize, denormalize
from datasets.caption.transforms import Compose, ToTensor, Normalize, Resize
from datasets.caption.transforms import RandAugment, MinMaxResize, MaxWHResize
DEFAULT_KTVIC_ROOT = Path('/mnt/s/cs505_kltn/dataset/ktvic/ktvic_dataset')
CAPTION_FIELD = os.environ.get('KTVIC_CAPTION_FIELD', 'segment_caption')
RAW_CAPTION_FIELD = os.environ.get('KTVIC_RAW_CAPTION_FIELD', 'caption')
TOKENIZER_BACKEND = os.environ.get('KTVIC_TOKENIZER_BACKEND', 'rdrsegmenter_segment_caption')

def resolve_ktvic_paths(split_mode='full', dev_split='seed42_valid10'):
    root = Path(os.environ.get('KTVIC_ROOT', '')).expanduser()
    train_image_path = Path(os.environ.get('KTVIC_TRAIN_IMAGES', str(root / 'train-images' / 'train-images'))).expanduser()
    test_image_path = Path(os.environ.get('KTVIC_TEST_IMAGES', str(root / 'public-test-images' / 'public-test-images'))).expanduser()
    _valid_default = str(train_image_path) if split_mode == 'dev_select' else str(test_image_path)
    valid_image_path = Path(os.environ.get('KTVIC_VALID_IMAGES', _valid_default)).expanduser()
    if split_mode == 'dev_select':
        dev_dir = root / dev_split
        train_caption_path = Path(os.environ.get('KTVIC_TRAIN_JSON', str(dev_dir / 'train_main.json'))).expanduser()
        valid_caption_path = Path(os.environ.get('KTVIC_VALID_JSON', str(dev_dir / 'valid_main.json'))).expanduser()
        vocab_source_path = Path(os.environ.get('KTVIC_VOCAB_SOURCE_JSON', str(dev_dir / 'vi_captions_train_only.json'))).expanduser()
    else:
        train_caption_path = Path(os.environ.get('KTVIC_TRAIN_JSON', str(root / 'train_data.json'))).expanduser()
        valid_caption_path = Path(os.environ.get('KTVIC_VALID_JSON', os.environ.get('KTVIC_TEST_JSON', str(root / 'test_data.json')))).expanduser()
        vocab_source_path = Path(os.environ.get('KTVIC_VOCAB_SOURCE_JSON', str(root / 'vi_captions_train_only.json'))).expanduser()
    test_caption_path = Path(os.environ.get('KTVIC_TEST_JSON', str(root / 'test_data.json'))).expanduser()
    return {'root': root, 'train_images': train_image_path, 'valid_images': valid_image_path, 'test_images': test_image_path, 'train_json': train_caption_path, 'valid_json': valid_caption_path, 'test_json': test_caption_path, 'vocab_source': vocab_source_path, 'split_mode': split_mode}

def get_transform(resize_name='maxwh', size=[384, 640], randaug=False):
    resize = RESIZE[resize_name](size)
    if randaug:
        return {'train': Compose([resize, RandAugment(), ToTensor(), normalize()]), 'valid': Compose([resize, ToTensor(), normalize()])}
    else:
        return {'train': Compose([resize, ToTensor(), normalize()]), 'valid': Compose([resize, ToTensor(), normalize()])}

def normalize_segment_caption(text):
    return ' '.join(str(text).strip().split())

def _maybe_generate_segment_caption_from_raw(raw_caption: str) -> str:
    raw_caption = normalize_segment_caption(raw_caption)
    if not raw_caption:
        return raw_caption
    return raw_caption

def resolve_caption_text(annotation: dict) -> str:
    seg = annotation.get(CAPTION_FIELD)
    if seg:
        return normalize_segment_caption(seg)
    raw = annotation.get(RAW_CAPTION_FIELD, '')
    return _maybe_generate_segment_caption_from_raw(raw)

def ensure_train_vocab_source():
    paths = resolve_ktvic_paths()
    vocab_source_path = paths['vocab_source']
    if vocab_source_path.exists():
        return
    vocab_source_path.parent.mkdir(parents=True, exist_ok=True)
    train_data = json.load(open(paths['train_json'], 'r', encoding='utf-8'))
    train_captions = [resolve_caption_text(ann) for ann in train_data['annotations']]
    with open(vocab_source_path, 'w', encoding='utf-8') as f:
        json.dump(train_captions, f, indent=4, ensure_ascii=False)

def _fingerprint_file(path: Path):
    path = Path(path)
    if not path.exists():
        return {'exists': False, 'path': str(path)}
    digest = sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    stat = path.stat()
    return {'exists': True, 'path': str(path), 'size_bytes': stat.st_size, 'sha256': digest.hexdigest(), 'mtime': stat.st_mtime}

def _summarize_split(data, image_root: Path):
    captions_per_image = defaultdict(int)
    sample_captions = []
    for ann in data['annotations']:
        captions_per_image[ann['image_id']] += 1
        if len(sample_captions) < 3:
            sample_captions.append(resolve_caption_text(ann))
    missing_files = []
    for image in data['images']:
        image_path = image_root / image['filename']
        if not image_path.exists():
            missing_files.append(str(image_path))
    counts = list(captions_per_image.values())
    return {'image_root': str(image_root), 'num_images': len(data['images']), 'num_annotations': len(data['annotations']), 'unique_annotated_images': len(captions_per_image), 'captions_per_image_min': min(counts) if counts else 0, 'captions_per_image_max': max(counts) if counts else 0, 'missing_image_count': len(missing_files), 'missing_images_preview': missing_files[:5], 'sample_segment_captions': sample_captions}

def collect_dataset_contract():
    paths = resolve_ktvic_paths()
    train_data = json.load(open(paths['train_json'], 'r', encoding='utf-8'))
    valid_data = json.load(open(paths['valid_json'], 'r', encoding='utf-8'))
    test_data = json.load(open(paths['test_json'], 'r', encoding='utf-8'))
    return {'resolved_paths': {key: str(value) for key, value in paths.items()}, 'fingerprints': {'train_json': _fingerprint_file(paths['train_json']), 'test_json': _fingerprint_file(paths['test_json']), 'vocab_source': _fingerprint_file(paths['vocab_source'])}, 'text_pipeline': {'caption_field': CAPTION_FIELD, 'raw_caption_field': RAW_CAPTION_FIELD, 'tokenizer_backend': TOKENIZER_BACKEND, 'normalization': 'preserve_tokenized_caption_spacing', 'lowercase': False, 'token_rule': 'split_whitespace_on_materialized_caption_field', 'fallback_when_missing': 'use_caption_or_generate_segment_caption'}, 'splits': {'train': _summarize_split(train_data, paths['train_images']), 'valid': _summarize_split(valid_data, paths['valid_images']), 'test': _summarize_split(test_data, paths['test_images'])}}

class Vocabulary:

    def __init__(self, freq_threshold):
        self.itos = {0: '<unk>', 1: '<pad>', 2: '<bos>', 3: '<eos>'}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        normalized = normalize_segment_caption(text)
        return normalized.split() if normalized else []

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi['<unk>'] for token in tokenized_text]

class CustomDataset(Dataset):

    def __init__(self, root_dir, captions_file, vocab_file=None, transform=None, freq_threshold=1):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.data = json.load(open(captions_file, 'r'))
        self.transform = transform
        self.imgid2imgname = {entry['id']: entry['filename'] for entry in self.data['images']}
        self.captions = [resolve_caption_text(ann) for ann in self.data['annotations']]
        if vocab_file is None:
            vocab_file = resolve_ktvic_paths()['vocab_source']
        all_captions = json.load(open(vocab_file, 'r'))
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(all_captions)

    def __len__(self):
        return len(self.data['annotations'])

    def __getitem__(self, idx):
        annotation = self.data['annotations'][idx]
        caption = resolve_caption_text(annotation)
        image_id = annotation['image_id']
        image_name = self.imgid2imgname[image_id]
        full_path = os.path.join(self.root_dir, image_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f'Image not found: {full_path}\n  root_dir={self.root_dir}\n  image_name={image_name}\n  Hint: check KTVIC_*_IMAGES env var or dataset.split_mode in config.\n  For dev_select mode → valid images from train-images/.\n  For full mode → valid images from public-test-images/.')
        image = Image.open(full_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(image)
        caption_vec = []
        caption_vec += [self.vocab.stoi['<bos>']]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi['<eos>']]
        return (img, torch.tensor(caption_vec), image_id)

class DictDataset(CustomDataset):

    def __init__(self, root_dir, captions_file, vocab_file=None, transform=None, freq_threshold=1):
        super().__init__(root_dir, captions_file, vocab_file, transform, freq_threshold)
        self.img_id_2_captions = self.img_id_2_captions()
        self.img_ids = list(self.img_id_2_captions.keys())

    def img_id_2_captions(self):
        img_id_2_captions = defaultdict(list)
        for ann in self.data['annotations']:
            img_id_2_captions[ann['image_id']].append(resolve_caption_text(ann))
        return img_id_2_captions

    def __len__(self):
        return len(self.img_id_2_captions)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image_name = self.imgid2imgname[img_id]
        img = Image.open(os.path.join(self.root_dir, image_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        captions = self.img_id_2_captions[img_id]
        return (img, captions, img_id)

class EvalCollate:

    def __init__(self, pad_idx, batch_first=True, device='cuda'):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        imgs = nested_tensor_from_tensor_list(imgs).to(self.device)
        captions = [item[1] for item in batch]
        image_ids = [item[2] for item in batch]
        return {'samples': imgs, 'captions': captions, 'image_ids': image_ids}

class CapsCollate:

    def __init__(self, pad_idx, batch_first=True, device='cuda'):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        imgs = nested_tensor_from_tensor_list(imgs).to(self.device)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        targets = targets.to(self.device)
        image_ids = [item[2] for item in batch]
        result = {'samples': imgs, 'captions': targets, 'image_ids': image_ids}
        assert 'image_ids' in result, 'CapsCollate MUST include image_ids key'
        return result

def maybe_limit_dataset(dataset, max_items):
    max_items = int(max_items or 0)
    if max_items <= 0 or max_items >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_items)))

def dataset_vocab(dataset):
    current = dataset
    while hasattr(current, 'dataset'):
        current = current.dataset
    return current.vocab

def get_datasets(resize_name='maxwh', size=(384, 640), randaug=False, max_train_items=0, max_valid_items=0, max_test_items=0, split_mode='full', dev_split='seed42_valid10'):
    ensure_train_vocab_source()
    paths = resolve_ktvic_paths(split_mode=split_mode, dev_split=dev_split)
    transforms = get_transform(resize_name=resize_name, size=list(size), randaug=randaug)
    train_dataset = CustomDataset(root_dir=paths['train_images'], captions_file=paths['train_json'], vocab_file=paths['vocab_source'], transform=transforms['train'])
    valid_dataset = CustomDataset(root_dir=paths['valid_images'], captions_file=paths['valid_json'], vocab_file=paths['vocab_source'], transform=transforms['valid'])
    test_dataset = CustomDataset(root_dir=paths['test_images'], captions_file=paths['test_json'], vocab_file=paths['vocab_source'], transform=transforms['valid'])
    train_dict_dataset = DictDataset(root_dir=paths['train_images'], captions_file=paths['train_json'], vocab_file=paths['vocab_source'], transform=transforms['train'])
    valid_dict_dataset = DictDataset(root_dir=paths['valid_images'], captions_file=paths['valid_json'], vocab_file=paths['vocab_source'], transform=transforms['valid'])
    test_dict_dataset = DictDataset(root_dir=paths['test_images'], captions_file=paths['test_json'], vocab_file=paths['vocab_source'], transform=transforms['valid'])
    return {'train': maybe_limit_dataset(train_dataset, max_train_items), 'valid': maybe_limit_dataset(valid_dataset, max_valid_items), 'test': maybe_limit_dataset(test_dataset, max_test_items), 'train_dict': maybe_limit_dataset(train_dict_dataset, max_train_items), 'valid_dict': maybe_limit_dataset(valid_dict_dataset, max_valid_items), 'test_dict': maybe_limit_dataset(test_dict_dataset, max_test_items)}

def get_dataloaders(device='cuda', batch_size=8, num_workers=4, resize_name='maxwh', size=(384, 640), randaug=False, max_train_items=0, max_valid_items=0, max_test_items=0, split_mode='full', dev_split='seed42_valid10'):
    datasets = get_datasets(resize_name=resize_name, size=size, randaug=randaug, max_train_items=max_train_items, max_valid_items=max_valid_items, max_test_items=max_test_items, split_mode=split_mode, dev_split=dev_split)
    collators = {'train': CapsCollate(dataset_vocab(datasets['train']).stoi['<pad>'], device=device), 'valid': CapsCollate(dataset_vocab(datasets['valid']).stoi['<pad>'], device=device), 'test': CapsCollate(dataset_vocab(datasets['test']).stoi['<pad>'], device=device), 'train_dict': EvalCollate(dataset_vocab(datasets['train_dict']).stoi['<pad>'], device=device), 'valid_dict': EvalCollate(dataset_vocab(datasets['valid_dict']).stoi['<pad>'], device=device), 'test_dict': EvalCollate(dataset_vocab(datasets['test_dict']).stoi['<pad>'], device=device)}
    train_sampler = DistributedSampler(datasets['train'], shuffle=True) if dist.is_available() and dist.is_initialized() else None
    samplers = {'train': train_sampler, 'valid': None, 'test': None, 'train_dict': None, 'valid_dict': None, 'test_dict': None}
    dataloaders = {key: DataLoader(datasets[key], batch_size=batch_size, num_workers=num_workers, collate_fn=collators[key], sampler=samplers[key], shuffle=key == 'train' and samplers[key] is None) for key in datasets}
    return (samplers, dataloaders)
if __name__ == '__main__':
    ensure_train_vocab_source()
    datasets = get_datasets()
    train_dataset, test_dataset = (datasets['train'], datasets['valid'])
    img, target, caption = train_dataset[0]
    print(img.shape)
    print(target.shape)
    print(' '.join([train_dataset.vocab.itos[token] for token in target.numpy()]))
    print(caption)
    img, captions, img_id = test_dataset[0]
    print(img.shape)
    print(captions)
    print(img_id)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=CapsCollate(train_dataset.vocab.stoi['<pad>']))
    valid_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=EvalCollate(train_dataset.vocab.stoi['<pad>']))
    dataloaders = {'train': train_loader, 'valid': valid_loader}
    vocab = train_dataset.vocab
    for batch in valid_loader:
        imgs, captions = (batch['samples'], batch['captions'])
        print(imgs.tensors.shape)
        print(captions)
        break
    for batch in train_loader:
        imgs, captions = (batch['samples'], batch['captions'])
        print(imgs.tensors.shape)
        print(captions.shape)
        break
