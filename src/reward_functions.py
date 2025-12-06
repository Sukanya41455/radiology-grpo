from typing import List
import numpy as np


CLINICAL_TERMS = [
    "pneumothorax",
    "effusion",
    "consolidation",
    "atelectasis",
    "cardiomegaly",
    "edema",
    "opacity",
    "fibrosis",
]


def _unigram_f1(pred_tokens, ref_tokens):
    ref_set = set(ref_tokens)
    pred_set = set(pred_tokens)

    tp = len(ref_set & pred_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def _clinical_f1(pred: str, ref: str) -> float:
    pred_lower = pred.lower()
    ref_lower = ref.lower()

    tp = fp = fn = 0
    for term in CLINICAL_TERMS:
        in_pred = term in pred_lower
        in_ref = term in ref_lower
        if in_pred and in_ref:
            tp += 1
        elif in_pred and not in_ref:
            fp += 1
        elif not in_pred and in_ref:
            fn += 1
    if tp + fp + fn == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def composite_reward(
    predictions: List[str],
    references: List[str],
) -> np.ndarray:
    """
    Reward = 0.6 * unigram F1 + 0.4 * clinical keyword F1 - small length penalty.
    Cheap but better than raw overlap.
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()

        uni_f1 = _unigram_f1(pred_tokens, ref_tokens)
        clinical = _clinical_f1(pred, ref)

        # light length penalty for very long rambly outputs
        length_penalty = max(0.0, (len(pred_tokens) - 80) / 80.0)

        reward = 0.6 * uni_f1 + 0.4 * clinical - 0.1 * length_penalty
        rewards.append(float(reward))

    return np.array(rewards, dtype=np.float32)
