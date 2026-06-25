import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub('[^\\w\\s_]', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def _extract_phrases(text: str, vocab: Set[str]) -> Set[str]:
    text_norm = _normalize(text)
    return {p for p in vocab if p in text_norm}

class LCR:

    def __init__(self, cultural_vocab: Set[str]) -> None:
        self.vocab = cultural_vocab

    def compute_score(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Tuple[float, Dict[int, float]]:
        total_gt, total_hit = (0, 0)
        per_image: Dict[int, float] = {}
        for img_id in gts:
            g_phrases: Set[str] = set()
            for ref in gts[img_id]:
                g_phrases |= _extract_phrases(ref, self.vocab)
            gen = res.get(img_id, [''])[0]
            p_phrases = _extract_phrases(gen, self.vocab)
            hit = len(p_phrases & g_phrases)
            total_gt += len(g_phrases)
            total_hit += hit
            per_image[img_id] = hit / max(len(g_phrases), 1)
        return (total_hit / max(total_gt, 1), per_image)

    def __str__(self) -> str:
        return 'LCR'

class LOR:

    def __init__(self, cultural_vocab: Set[str]) -> None:
        self.vocab = cultural_vocab

    def compute_score(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Tuple[float, Dict[int, float]]:
        total_gen, total_over = (0, 0)
        per_image: Dict[int, float] = {}
        for img_id in gts:
            g_phrases: Set[str] = set()
            for ref in gts[img_id]:
                g_phrases |= _extract_phrases(ref, self.vocab)
            gen = res.get(img_id, [''])[0]
            p_phrases = _extract_phrases(gen, self.vocab)
            over = len(p_phrases - g_phrases)
            total_gen += len(p_phrases)
            total_over += over
            per_image[img_id] = over / max(len(p_phrases), 1)
        return (total_over / max(total_gen, 1), per_image)

    def __str__(self) -> str:
        return 'LOR'

class CulturalMetrics:

    def __init__(self, cultural_vocab: Set[str], facet_map: Optional[Dict[str, str]]=None) -> None:
        self.vocab = cultural_vocab
        self.facet_map = facet_map or {}
        self.lcr = LCR(cultural_vocab)
        self.lor = LOR(cultural_vocab)

    def compute_score(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Dict:
        lcr_val, lcr_pi = self.lcr.compute_score(gts, res)
        lor_val, lor_pi = self.lor.compute_score(gts, res)
        per_facet = {}
        if self.facet_map:
            for facet in set(self.facet_map.values()):
                f_vocab = {p for p, f in self.facet_map.items() if f == facet}
                fl, _ = LCR(f_vocab).compute_score(gts, res)
                fo, _ = LOR(f_vocab).compute_score(gts, res)
                per_facet[facet] = {'lcr': fl, 'lor': fo}
        return {'lcr': lcr_val, 'lor': lor_val, 'lcr_per_image': lcr_pi, 'lor_per_image': lor_pi, 'per_facet': per_facet}

    @classmethod
    def from_vocab_file(cls, vocab_path: str, facet_path: Optional[str]=None) -> 'CulturalMetrics':
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = set(data.get('phrases', []))
        facet_map = data.get('facets', None)
        if facet_path:
            with open(facet_path, 'r', encoding='utf-8') as f:
                facet_map = json.load(f)
        return cls(vocab, facet_map)

    def __str__(self) -> str:
        return 'CulturalMetrics'
