from typing import List
import numpy as np


def composite_reward(
    predictions: List[str],
    references: List[str],
) -> np.ndarray:
    """
    Very simple reward: word overlap / ref length - penalty for very long outputs.
    Replace this with BLEU/ROUGE/CIDEr + CheXbert + RadGraph when ready.
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()

        overlap = len(set(ref_tokens) & set(pred_tokens))
        recall_like = overlap / (len(ref_tokens) + 1e-8)

        # Length penalty if prediction too long
        length_penalty = max(0.0, (len(pred_tokens) - 60) / 60.0)

        reward = recall_like - 0.1 * length_penalty
        rewards.append(float(reward))

    return np.array(rewards, dtype=np.float32)
