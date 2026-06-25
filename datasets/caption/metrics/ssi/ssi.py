from typing import Dict, List
import numpy as np

def _softmax_t(logits: np.ndarray, tau: float=0.07) -> np.ndarray:
    scaled = logits / max(tau, 1e-08)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)

def _entropy(probs: np.ndarray, axis: int=-1) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=axis)

class SlotSpecializationIndex:

    def __init__(self, k_loc: int=16, tau: float=0.07) -> None:
        self.k_loc = k_loc
        self.tau = tau

    def compute_similarity(self, C_loc: np.ndarray, W_c: np.ndarray, phrase_embeddings: np.ndarray) -> np.ndarray:
        proj = C_loc @ W_c.T
        pn = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        en = phrase_embeddings / (np.linalg.norm(phrase_embeddings, axis=1, keepdims=True) + 1e-12)
        return pn @ en.T

    def compute_ssi(self, C_loc: np.ndarray, W_c: np.ndarray, phrase_embeddings: np.ndarray) -> Dict:
        sim = self.compute_similarity(C_loc, W_c, phrase_embeddings)
        K, P = sim.shape
        probs = _softmax_t(sim, self.tau)
        ssi_per_slot = []
        assignments = {}
        for k in range(K):
            h = _entropy(probs[k])
            ssi_per_slot.append(float(1.0 - h / np.log(max(P, 2))))
            assignments[k] = [int(i) for i in np.argsort(sim[k])[-3:][::-1]]
        return {'ssi_per_slot': ssi_per_slot, 'mean_ssi': float(np.mean(ssi_per_slot)), 'assignment': assignments, 'similarity_matrix': sim.tolist()}

    def compute_for_model(self, images: List[np.ndarray]) -> Dict:
        raise NotImplementedError('SSI: LCQM not yet implemented.')

    def __str__(self) -> str:
        return 'SSI'
