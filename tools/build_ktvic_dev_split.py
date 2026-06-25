#!/usr/bin/env python3
"""Build a deterministic image-level KTVIC train/dev split + a shared vocab source.

Why this exists
---------------
The historic KTVIC-finetuned checkpoints were lost (deleted with the ``cs505_kltn_b3``
worktree; ``runs/`` is gitignored). Regenerating the ViG A0-base requires a clean two-stage
refit that fixes the §1D data-leak (which selected its checkpoint on the *test* split):

    Stage A  -> train on a 90% train split, select best checkpoint by held-out 10% dev CIDEr
    Stage B  -> warm-start from Stage-A best, refit on the FULL train set, then official test

This script produces the inputs both stages consume, so the split and vocabulary are fixed
once and reused by every later ablation row (A0..C2):

    <out-dir>/train_main.json            # 90% images (+ their captions)  -> Stage A train
    <out-dir>/valid_main.json            # 10% images (+ their captions)  -> dev select
    <out-dir>/vi_captions_train_only.json# vocab source built from the FULL train (no UNK)
    <out-dir>/split_manifest.json        # seed, fractions, counts, SHA256 fingerprints

Determinism
-----------
Image ids are sorted ascending then shuffled with ``random.Random(seed)``; the first
``round(n * valid_frac)`` go to dev, the rest to train. Splitting is image-level: all of an
image's captions move together, so no caption leaks across the boundary.

The vocab source replicates ``vicap_dataset.resolve_caption_text`` exactly (segment_caption
preferred, else raw caption, normalized via whitespace collapse) so the pinned file is
byte-equivalent to what ``ensure_train_vocab_source`` would otherwise auto-build from the full
train -- guaranteeing identical token->id maps across Stage A and Stage B.

Usage (run once, login node, no GPU)
------------------------------------
    python vig/tools/build_ktvic_dev_split.py \
        --train-json $DATASET_ROOT/train_data.json \
        --out-dir    $DATASET_ROOT/splits/seed42_valid10 \
        --seed 42 --valid-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import random
from hashlib import sha256
from pathlib import Path

# Mirror datasets/caption/vicap_dataset.py so the pinned vocab matches the auto-build exactly.
CAPTION_FIELD = "segment_caption"
RAW_CAPTION_FIELD = "caption"


def normalize_segment_caption(text: object) -> str:
    """Collapse whitespace, matching vicap_dataset.normalize_segment_caption."""
    return " ".join(str(text).strip().split())


def resolve_caption_text(annotation: dict) -> str:
    """Resolve caption text exactly as vicap_dataset.resolve_caption_text does."""
    seg = annotation.get(CAPTION_FIELD)
    if seg:
        return normalize_segment_caption(seg)
    return normalize_segment_caption(annotation.get(RAW_CAPTION_FIELD, ""))


def _sha256_of(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4, ensure_ascii=False)


def _subset(data: dict, image_ids: set[int]) -> dict:
    """Return a {images, annotations} subset whose members belong to image_ids."""
    images = [im for im in data["images"] if im["id"] in image_ids]
    annotations = [an for an in data["annotations"] if an["image_id"] in image_ids]
    return {"images": images, "annotations": annotations}


def build_split(train_json: Path, out_dir: Path, seed: int, valid_frac: float) -> dict:
    data = json.load(open(train_json, encoding="utf-8"))
    all_ids = sorted(im["id"] for im in data["images"])
    n_total = len(all_ids)
    if len(set(all_ids)) != n_total:
        raise ValueError("Duplicate image ids in train_data.json — split would be ambiguous.")

    shuffled = list(all_ids)
    random.Random(seed).shuffle(shuffled)
    n_valid = round(n_total * valid_frac)
    valid_ids = set(shuffled[:n_valid])
    train_ids = set(shuffled[n_valid:])

    # Fail loud: the split must be a clean, exhaustive, disjoint partition of all images.
    assert valid_ids.issubset(set(all_ids)), "dev ids escaped the train id set"
    assert train_ids.isdisjoint(valid_ids), "train and dev overlap"
    assert train_ids | valid_ids == set(all_ids), "split does not cover every image"
    assert len(valid_ids) == n_valid, "dev count mismatch"
    assert len(train_ids) == n_total - n_valid, "train count mismatch"

    out_dir.mkdir(parents=True, exist_ok=True)
    train_main = _subset(data, train_ids)
    valid_main = _subset(data, valid_ids)
    vocab_captions = [resolve_caption_text(an) for an in data["annotations"]]

    train_path = out_dir / "train_main.json"
    valid_path = out_dir / "valid_main.json"
    vocab_path = out_dir / "vi_captions_train_only.json"
    _write_json(train_path, train_main)
    _write_json(valid_path, valid_main)
    _write_json(vocab_path, vocab_captions)

    manifest = {
        "seed": seed,
        "valid_frac": valid_frac,
        "source_train_json": str(train_json),
        "counts": {
            "images_total": n_total,
            "images_train": len(train_ids),
            "images_valid": len(valid_ids),
            "annotations_train": len(train_main["annotations"]),
            "annotations_valid": len(valid_main["annotations"]),
            "vocab_captions": len(vocab_captions),
        },
        "sha256": {
            "source_train_json": _sha256_of(train_json),
            "train_main.json": _sha256_of(train_path),
            "valid_main.json": _sha256_of(valid_path),
            "vi_captions_train_only.json": _sha256_of(vocab_path),
        },
    }
    _write_json(out_dir / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic KTVIC dev split + vocab source.")
    parser.add_argument("--train-json", required=True, type=Path, help="Path to full train_data.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output split directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    args = parser.parse_args()

    manifest = build_split(args.train_json, args.out_dir, args.seed, args.valid_frac)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
