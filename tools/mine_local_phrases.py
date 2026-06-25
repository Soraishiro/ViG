"""
ViG S2 — Vietnamese Local Phrase Mining (C7)

Mines culturally-relevant phrases from KTVIC training captions.
Reads pre-segmented `segment_caption` field (RDRSegmenter output, PhoBERT-compatible).
Follows pattern: annotation_task/scripts/task0/mine_candidates_train_only.py

Output: data/local_phrase_vocab_v1.json
Manual audit: annotation_task/docs/cultural_branch_evaluation_plan_annotator.md
"""

import json, math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

FUNCTION_WORDS: Set[str] = {
    'là', 'và', 'ở', 'từ', 'trong', 'trên', 'dưới', 'ngoài', 'với',
    'có', 'không', 'được', 'cái', 'chiếc', 'những', 'một', 'hai', 'ba',
    'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'này', 'kia', 'đó',
    'đây', 'nào', 'nên', 'nếu', 'hay', 'hoặc', 'nhưng', 'tuy', 'mà', 'đến',
    'qua', 'theo', 'bằng', 'do', 'để', 'lại', 'còn', 'sẽ', 'đã', 'đang',
    'vừa', 'tại', 'trị', 'sao', 'gì', 'ai', 'nơi', 'thời', 'như', 'rồi',
    'mới', 'lên', 'xuống', 'ra', 'vào', 'cũng', 'thì', 'mặc', 'khoảng',
    'lúc', 'nay', 'chừng', 'thứ', 'chiều', 'phía', 'bên', 'góc', 'cạnh',
    'bao_gồm', 'gồm', 'tối', 'sáng', 'chính', 'riêng', 'khác', 'giống',
    'tương_tự', 'cùng', 'là_một', 'là_cái',
}

GENERIC_PHRASES: Set[str] = {
    'hình_ảnh', 'khung_cảnh', 'cảnh_tượng', 'quang_cảnh', 'cảnh', 'khung',
    'hình', 'cách', 'điều', 'chuyện', 'vấn_đề', 'khoảng', 'chừng', 'lúc',
    'thời_điểm', 'lúc_này', 'lúc_đó', 'người', 'đàn_ông', 'phụ_nữ',
    'cô_gái', 'bé_gái', 'bé_trai', 'nền', 'mặt_đất', 'bầu_trời',
    'phía_sau', 'phía_trước',
}


def load_captions(ann_file: str) -> List[str]:
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    captions = [ann.get('segment_caption') or ann.get('caption', '')
                for ann in data.get('annotations', [])]
    return [c for c in captions if c]


def extract_phrases(captions: List[str]) -> List[str]:
    """Extract 1-3 consecutive non-function words. Splits by SPACE."""
    all_phrases: List[str] = []
    for sent in captions:
        if not sent: continue
        words = sent.split()
        phrase: List[str] = []
        for word in words:
            wl = word.lower()
            if wl not in FUNCTION_WORDS:
                phrase.append(word)
                if len(phrase) == 3:
                    all_phrases.append(' '.join(phrase))
                    phrase = phrase[1:]
            else:
                if 1 <= len(phrase) <= 3:
                    all_phrases.append(' '.join(phrase))
                phrase = []
        if 1 <= len(phrase) <= 3:
            all_phrases.append(' '.join(phrase))
    return all_phrases


def should_filter(phrase: str) -> bool:
    pl = phrase.lower()
    if pl in GENERIC_PHRASES: return True
    for word in phrase.split():
        wl = word.lower()
        if wl in FUNCTION_WORDS or wl in GENERIC_PHRASES: return True
        if len(word.replace('_', '')) <= 1: return True
    return False


def mine_phrases(captions: List[str], min_freq: int = 5,
                 min_idf: float = 2.0) -> Dict:
    raw = extract_phrases(captions)
    freq: Counter = Counter()
    for p in raw:
        if not should_filter(p): freq[p] += 1
    n_docs = len(captions)
    doc_freq: Counter = Counter()
    for sent in captions:
        doc_freq.update(set(sent.lower().split()))
    candidates = {}
    for phrase, count in freq.most_common():
        if count < min_freq: continue
        tokens = phrase.lower().split()
        avg_idf = sum(math.log(n_docs/(doc_freq.get(t,0)+1))+1 for t in tokens)/len(tokens)
        if avg_idf < min_idf: continue
        candidates[phrase] = {'freq': count, 'mean_idf': round(avg_idf, 3),
                              'n_tokens': len(tokens)}
    return {'phrases': list(candidates.keys()), 'candidates': candidates,
            'total_captions': len(captions),
            'params': {'min_freq': min_freq, 'min_idf': min_idf}}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Mine Vietnamese local phrases")
    p.add_argument('--train_annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/local_phrase_vocab_v1.json'))
    p.add_argument('--min_freq', type=int, default=5)
    p.add_argument('--min_idf', type=float, default=2.0)
    args = p.parse_args()
    captions = load_captions(str(args.train_annotations))
    result = mine_phrases(captions, args.min_freq, args.min_idf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'phrases': result['phrases'], 'candidates': result['candidates'],
            'total_captions': result['total_captions'], 'params': result['params'],
            'facet_map': {}, 'audit_status': 'unreviewed',
        }, f, ensure_ascii=False, indent=2)
    print(f"[mine_phrases] {len(result['phrases'])} phrases → {args.output}")
    print("[mine_phrases] NEXT: native speaker audit (annotation_task/docs/)")


if __name__ == '__main__':
    main()
