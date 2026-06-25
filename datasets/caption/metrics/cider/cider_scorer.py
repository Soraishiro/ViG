import copy
from collections import defaultdict
import numpy as np
import math

def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    return precook(test, n)

class CiderScorer(object):

    def __init__(self, refs, test=None, n=4, sigma=6.0, doc_frequency=None, ref_len=None):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.doc_frequency = defaultdict(float)
        self.ref_len = None
        for k in refs.keys():
            self.crefs.append(cook_refs(refs[k]))
            if test is not None:
                self.ctest.append(cook_test(test[k][0]))
            else:
                self.ctest.append(None)
        if doc_frequency is None and ref_len is None:
            self.compute_doc_freq()
            self.ref_len = np.log(float(len(self.crefs)))
        else:
            self.doc_frequency = doc_frequency
            self.ref_len = ref_len

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for ngram, count in ref.items()]):
                self.doc_frequency[ngram] += 1

    def compute_cider(self):

        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for ngram, term_freq in cnts.items():
                df = np.log(max(1.0, self.doc_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return (vec, norm, length)

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for ngram, count in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= norm_hyp[n] * norm_ref[n]
                assert not math.isnan(val[n])
                val[n] *= np.e ** (-delta ** 2 / (2 * self.sigma ** 2))
            return val
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        score = self.compute_cider()
        return (np.mean(np.array(score)), np.array(score))
