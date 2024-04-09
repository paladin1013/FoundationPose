import cupy as cp
from gpu_utils import get_rotation_diff


def max_R(
    center_Rs: cp.ndarray,  # (K, 3, 3)
    center_ts: cp.ndarray,  # (K, 3)
    pred_Rs: cp.ndarray,  # (M, 3, 3)
    pred_ts: cp.ndarray,  # (M, 3)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_Rs.shape[1] == 3 & center_Rs.shape[2] == 3
    assert center_ts.shape[1] == 3
    assert pred_Rs.shape[1] == 3 & pred_Rs.shape[2] == 3
    assert pred_ts.shape[1] == 3
    assert pred_Rs.shape[0] == pred_scores.shape[0]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )

    R_diffs = get_rotation_diff(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    nonconformity = cp.amax(R_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity


def mean_R(
    center_Rs: cp.ndarray,  # (K, 3, 3)
    center_ts: cp.ndarray,  # (K, 3)
    pred_Rs: cp.ndarray,  # (M, 3, 3)
    pred_ts: cp.ndarray,  # (M, 3)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_Rs.shape[1] == 3 & center_Rs.shape[2] == 3
    assert center_ts.shape[1] == 3
    assert pred_Rs.shape[1] == 3 & pred_Rs.shape[2] == 3
    assert pred_ts.shape[1] == 3
    assert pred_Rs.shape[0] == pred_scores.shape[0]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )

    R_diffs = get_rotation_diff(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    nonconformity = cp.sum(R_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity


def normalized_max_R(
    center_Rs: cp.ndarray,  # (K, 3, 3)
    center_ts: cp.ndarray,  # (K, 3)
    pred_Rs: cp.ndarray,  # (M, 3, 3)
    pred_ts: cp.ndarray,  # (M, 3)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_Rs.shape[1] == 3 & center_Rs.shape[2] == 3
    assert center_ts.shape[1] == 3
    assert pred_Rs.shape[1] == 3 & pred_Rs.shape[2] == 3
    assert pred_ts.shape[1] == 3
    assert pred_Rs.shape[0] == pred_scores.shape[0]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )

    R_diffs = cp.linalg.norm(
        center_Rs[:, None, :, :] - pred_Rs[None, :, :, :], axis=(2, 3)
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_diffs = get_rotation_diff(
        cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
    )  # (M, M)
    mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
    normalized_R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)

    nonconformity = cp.amax(normalized_R_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity


def normalized_mean_R(
    center_Rs: cp.ndarray,  # (K, 3, 3)
    center_ts: cp.ndarray,  # (K, 3)
    pred_Rs: cp.ndarray,  # (M, 3, 3)
    pred_ts: cp.ndarray,  # (M, 3)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:
    assert center_Rs.shape[1] == 3 & center_Rs.shape[2] == 3
    assert center_ts.shape[1] == 3
    assert pred_Rs.shape[1] == 3 & pred_Rs.shape[2] == 3
    assert pred_ts.shape[1] == 3
    assert pred_Rs.shape[0] == pred_scores.shape[0]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )

    R_diffs = cp.linalg.norm(
        center_Rs[:, None, :, :] - pred_Rs[None, :, :, :], axis=(2, 3)
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_diffs = get_rotation_diff(
        cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
    )  # (M, M)
    mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
    normalized_R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)

    nonconformity = cp.sum(normalized_R_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity
