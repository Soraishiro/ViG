from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer
from .lcr_lor import LCR, LOR, CulturalMetrics
from .ags import AttentionGroundingScore
from .ssi import SlotSpecializationIndex
from .gdi import GateDiscriminationIndex

def compute_scores(gts, gen):
    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return (all_score, all_scores)
