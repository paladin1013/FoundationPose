import cupy as cp
from line_profiler import profile
from gpu_utils import get_rotation_dist


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

    R_diffs = get_rotation_dist(
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

    R_diffs = get_rotation_dist(
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

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_diffs = get_rotation_dist(
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

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_diffs = get_rotation_dist(
        cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
    )  # (M, M)
    mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
    normalized_R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)

    nonconformity = cp.sum(normalized_R_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity


def max_t(
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
    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)
    nonconformity = cp.amax(t_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity


def mean_t(
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
    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)
    nonconformity = cp.sum(t_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity

def normalized_max_t(
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

    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)

    # Calculate the pairwise difference:
    pred_t_diffs = cp.linalg.norm(pred_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (M, M)

    mean_t_diffs = cp.mean(pred_t_diffs, axis=1)  # (M, )
    normalized_t_diffs = t_diffs / mean_t_diffs[None, :]  # (K, M)

    nonconformity = cp.amax(normalized_t_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity



def normalized_mean_t(
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

    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)

    # Calculate the pairwise difference:
    pred_t_diffs = cp.linalg.norm(pred_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (M, M)

    mean_t_diffs = cp.mean(pred_t_diffs, axis=1)  # (M, )
    normalized_t_diffs = t_diffs / mean_t_diffs[None, :]  # (K, M)

    nonconformity = cp.sum(normalized_t_diffs * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity

R_T_RATIO = 8
NORMALIZED_R_T_RATIO = 0.21

def max_Rt(
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

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)

    R_nonconformity = cp.amax(R_diffs * pred_prob[None, :], axis=1)  # (K, )
    t_nonconformity = cp.amax(t_diffs * pred_prob[None, :], axis=1)  # (K, )

    nonconformity = cp.maximum(R_nonconformity, t_nonconformity * R_T_RATIO)

    return nonconformity

def mean_Rt(
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

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)

    R_nonconformity = cp.sum(R_diffs * pred_prob[None, :], axis=1)  # (K, )
    t_nonconformity = cp.sum(t_diffs * pred_prob[None, :], axis=1)  # (K, )

    nonconformity = cp.maximum(R_nonconformity, t_nonconformity * R_T_RATIO)

    return nonconformity


@profile
def normalized_max_Rt(
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

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)


    # Calculate the pairwise difference:
    pred_R_diffs = get_rotation_dist(
        cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
    )  # (M, M)
    mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
    normalized_R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)
    R_nonconformity = cp.amax(normalized_R_diffs * pred_prob[None, :], axis=1)  # (K, )

    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)
    pred_t_diffs = cp.linalg.norm(pred_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (M, M)
    mean_t_diffs = cp.mean(pred_t_diffs, axis=1)  # (M, )
    normalized_t_diffs = t_diffs / mean_t_diffs[None, :]  # (K, M)
    t_nonconformity = cp.amax(normalized_t_diffs * pred_prob[None, :], axis=1)  # (K, )

    nonconformity = cp.maximum(R_nonconformity, t_nonconformity * NORMALIZED_R_T_RATIO)

    return nonconformity


@profile
def normalized_mean_Rt(
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

    t_diffs = cp.linalg.norm(center_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (K, M)
    pred_t_diffs = cp.linalg.norm(pred_ts[:, None, :] - pred_ts[None, :, :], axis=2) # (M, M)
    mean_t_diffs = cp.mean(pred_t_diffs, axis=1)  # (M, )
    normalized_t_diffs = t_diffs / mean_t_diffs[None, :]  # (K, M)
    t_nonconformity = cp.sum(normalized_t_diffs * pred_prob[None, :], axis=1)  # (K, )

    R_diffs = get_rotation_dist(
        cp.repeat(center_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], center_Rs.shape[0], axis=0),
    )  # (K, M)
    pred_R_diffs = get_rotation_dist(
        cp.repeat(pred_Rs[:, None, :, :], pred_Rs.shape[0], axis=1),
        cp.repeat(pred_Rs[None, :, :, :], pred_Rs.shape[0], axis=0),
    )  # (M, M)
    mean_R_diffs = cp.mean(pred_R_diffs, axis=1)  # (M, )
    normalized_R_diffs = R_diffs / mean_R_diffs[None, :]  # (K, M)
    R_nonconformity = cp.sum(normalized_R_diffs * pred_prob[None, :], axis=1)  # (K, )

    nonconformity = cp.maximum(R_nonconformity, t_nonconformity * NORMALIZED_R_T_RATIO)

    return nonconformity