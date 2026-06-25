from typing import Dict, List, Tuple
import numpy as np

def _top_k_mask(heatmap: np.ndarray, k_frac: float=0.2) -> np.ndarray:
    threshold = np.quantile(heatmap, 1.0 - k_frac)
    return heatmap >= threshold

def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / max(union, 1))

def _grid_to_spatial(attn_weights: np.ndarray, H: int, W: int, downsample: int=64) -> np.ndarray:
    from scipy.ndimage import zoom
    h_grid, w_grid = (H // downsample, W // downsample)
    spatial = attn_weights[:h_grid * w_grid].reshape(h_grid, w_grid)
    heatmap = zoom(spatial, (downsample, downsample), order=1)
    return heatmap[:H, :W]

class AttentionGroundingScore:

    def __init__(self, k_frac: float=0.2) -> None:
        self.k_frac = k_frac

    def _extract_slot_attention(self, image: np.ndarray, slot_idx: int) -> np.ndarray:
        raise NotImplementedError('AGS: LCQM not yet implemented. Wire to cultural_memory.py cross-attention with return_attn=True when ready.')

    def compute_per_image(self, image: np.ndarray, gt_boxes: List[Tuple[float, float, float, float]], slot_indices: List[int], H: int=384, W: int=640) -> Dict[int, List[float]]:
        results: Dict[int, List[float]] = {}
        for slot_idx in slot_indices:
            slot_ags: List[float] = []
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                slot_ags.append(0.0)
            results[slot_idx] = slot_ags
        return results

    def compute_groups(self, group_images: Dict[str, List[np.ndarray]], group_boxes: Dict[str, List[List[Tuple[float, float, float, float]]]], slot_indices: List[int]) -> Dict[int, Dict[str, List[float]]]:
        result = {}
        for slot_idx in slot_indices:
            slot_res = {}
            for grp in ['A', 'B', 'C']:
                all_ags: List[float] = []
                for img, boxes in zip(group_images[grp], group_boxes[grp]):
                    ags_map = self.compute_per_image(img, boxes, [slot_idx])
                    all_ags.extend(ags_map.get(slot_idx, []))
                slot_res[grp] = all_ags
            result[slot_idx] = slot_res
        return result

    def __str__(self) -> str:
        return 'AGS'
