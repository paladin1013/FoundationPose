import cupy as cp
from line_profiler import profile
from gpu_utils import get_rotation_dist

calibrated_R_ratio = 1
calibrated_t_ratio = 1

def nonconformity_func(
    center_Rs: cp.ndarray,  # (K, 3, 3)
    center_ts: cp.ndarray,  # (K, 3)
    pred_Rs: cp.ndarray,  # (M, 3, 3)
    pred_ts: cp.ndarray,  # (M, 3)
    pred_scores: cp.ndarray,  # (M, )
    aggregate_method: str, # "max" or "mean"
    normalize: bool,
    R_ratio: float,
    t_ratio: float,
) -> cp.ndarray:
    assert center_Rs.shape[1] == 3 & center_Rs.shape[2] == 3
    assert center_ts.shape[1] == 3
    assert pred_Rs.shape[1] == 3 & pred_Rs.shape[2] == 3
    assert pred_ts.shape[1] == 3
    assert pred_Rs.shape[0] == pred_scores.shape[0]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )

    if R_ratio > 0:
        R_diffs = get_rotation_dist(
            cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
            cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
        )  # (K, M)
        if normalize:
            pred_R_diffs = get_rotation_dist(
                cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
                cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
            )  # (M, M)
            mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
            R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)

        if aggregate_method == "max":
            R_nonconformity = cp.amax(R_diffs * pred_prob[None, :], axis=1)
        elif aggregate_method == "mean":
            R_nonconformity = cp.sum(R_diffs * pred_prob[None, :], axis=1)  # (K, )
        else:
            raise ValueError(f"aggregate_method should be either 'max' or 'mean', got {aggregate_method}")
    else:
        R_nonconformity = cp.zeros(center_Rs.shape[0])

    if t_ratio > 0:
        t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)
        if normalize:
            pred_t_diffs = cp.linalg.norm(pred_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (M, M)
            mean_t_diffs = cp.mean(pred_t_diffs, axis=1)  # (M, )
            t_diffs = t_diffs / mean_t_diffs[None, :]  # (K, M)

        if aggregate_method == "max":
            t_nonconformity = cp.amax(t_diffs * pred_prob[None, :], axis=1)
        elif aggregate_method == "mean":
            t_nonconformity = cp.sum(t_diffs * pred_prob[None, :], axis=1)  # (K, )
        else:
            raise ValueError(f"aggregate_method should be either 'max' or 'mean', got {aggregate_method}")
    else:
        t_nonconformity = cp.zeros(center_Rs.shape[0])

    nonconformity = cp.maximum(R_nonconformity * R_ratio, t_nonconformity * t_ratio)
    return nonconformity
