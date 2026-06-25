"""
ViG S2 — Adapter: V_cultural (facet+variants format) → build_phrase_supervision.py format

Input:  {phrase: {facet, variants}, ...}   (annotation_task output)
Output: {phrases: [...], facet_map: {...}, audit_status: "reviewed"}

Usage:
  python vig/tools/adapt_vocab_format.py \\
      --input annotation_task/output/task0/V_cultural_final.json \\
      --output vig/data/local_phrase_vocab_v1.json
"""

import json
from pathlib import Path
from typing import Dict, List


def extract_phrases(vocab: Dict) -> List[str]:
    """Trích xuất danh sách phrases không trùng lặp từ format facet+variants.

    Bao gồm: keys gốc (đã underscore) + tất cả variants.
    Chuẩn hóa spaces → underscores để khớp RDRSegmenter format.
    """
    phrases = set()
    for term, info in vocab.items():
        if not isinstance(info, dict):
            continue
        # Key gốc (đã ở dạng underscore)
        phrases.add(term)
        # Variants: chuẩn hóa spaces → underscores
        for variant in info.get("variants", []):
            normalized = variant.replace(" ", "_")
            phrases.add(normalized)
    return sorted(phrases)


def adapt(input_path: str, output_path: str) -> None:
    """Đọc V_cultural facet format, ghi ra format build_phrase_supervision.py."""
    with open(input_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    phrases = extract_phrases(vocab)

    # Build facet_map để preserve metadata
    facet_map = {}
    for term, info in vocab.items():
        if isinstance(info, dict) and "facet" in info:
            facet_map[term] = info["facet"]

    output = {
        "phrases": phrases,
        "facet_map": facet_map,
        "audit_status": "reviewed",
        "total_phrases": len(phrases),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adapt V_cultural facet format → phrase list format")
    parser.add_argument("--input", required=True, help="V_cultural JSON (facet+variants format)")
    parser.add_argument("--output", required=True, help="Output JSON for build_phrase_supervision.py")
    args = parser.parse_args()
    result = adapt(args.input, args.output)
    print(f"[adapt_vocab] {result['total_phrases']} phrases written to {args.output}")
