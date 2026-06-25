from typing import Dict
import numpy as np

def _auc_trap(tpr: np.ndarray, fpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))

def _delong_se(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos, n_neg = (len(idx_pos), len(idx_neg))
    if n_pos < 2 or n_neg < 2:
        return (0.0, 0.0)
    theta = np.array([np.mean(1.0 * (y_score[idx_neg] < y_score[i])) + 0.5 * np.mean(1.0 * (y_score[idx_neg] == y_score[i])) for i in range(len(y_true))])
    v_pos = np.var(theta[idx_pos], ddof=1) / n_pos
    v_neg = np.var(theta[idx_neg], ddof=1) / n_neg
    desc = np.argsort(y_score)[::-1]
    ys = y_true[desc]
    tpr = np.cumsum(ys == 1) / n_pos
    fpr = np.cumsum(ys == 0) / n_neg
    auc = _auc_trap(tpr, fpr)
    return (auc, float(np.sqrt(v_pos + v_neg)))

class GateDiscriminationIndex:

    def __init__(self) -> None:
        pass

    def compute_auc(self, eta_loc_values: np.ndarray, token_labels: np.ndarray, alpha: float=0.05) -> Dict:
        mask = ~np.isnan(eta_loc_values)
        ys, yt = (eta_loc_values[mask], token_labels[mask])
        n_pos, n_neg = (int(np.sum(yt == 1)), int(np.sum(yt == 0)))
        if n_pos == 0 or n_neg == 0:
            return {'auc': np.nan, 'error': 'Only one class present'}
        auc, se = _delong_se(yt, ys)
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        return {'auc': float(np.clip(auc, 0, 1)), 'ci_lower': float(max(auc - z * se, 0)), 'ci_upper': float(min(auc + z * se, 1)), 'n_cultural': n_pos, 'n_function': n_neg}

    def compute_per_dataset(self, ktvic_logs: Dict, uit_viic_logs: Dict) -> Dict:
        raise NotImplementedError('GDI: Inference diagnostic logging not yet implemented. Wire to eval_vicap.py --save_diagnostics with η_loc recording.')

    def __str__(self) -> str:
        return 'GDI'
