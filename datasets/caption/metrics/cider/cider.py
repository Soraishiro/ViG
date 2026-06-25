from .cider_scorer import CiderScorer

class Cider:

    def __init__(self, gts=None, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self.doc_frequency = None
        self.ref_len = None
        if gts is not None:
            tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma)
            self.doc_frequency = tmp_cider.doc_frequency
            self.ref_len = tmp_cider.ref_len

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        cider_scorer = CiderScorer(gts, test=res, n=self._n, sigma=self._sigma, doc_frequency=self.doc_frequency, ref_len=self.ref_len)
        return cider_scorer.compute_score()

    def __str__(self):
        return 'CIDEr'
