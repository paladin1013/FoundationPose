import cupy as cp
from gpu_utils import get_rot_diff

def max_R(
    center_poses: cp.ndarray,  # (K, 4, 4)
    pred_poses: cp.ndarray,  # (M, 4, 4)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_poses.shape[1] == 4 & center_poses.shape[2] == 4
    assert pred_poses.shape[1] == 4 & pred_poses.shape[2] == 4
    assert pred_poses.shape[0] == pred_scores.shape[0]

    gt_R = center_poses[:, :3, :3]  # (K, 3, 3)
    # gt_t = center_poses[:,:3, 3]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )
    pred_R = pred_poses[:, :3, :3]  # (M, 3, 3)
    # pred_t = pred_poses[:,:3,3]

    R_diff = get_rot_diff(
        cp.repeat(gt_R[:, None, :, :], pred_R.shape[0], axis=1),
        cp.repeat(pred_R[None, :, :, :], gt_R.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    nonconformity = cp.amax(R_diff * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity

def mean_R(
    center_poses: cp.ndarray,  # (K, 4, 4)
    pred_poses: cp.ndarray,  # (M, 4, 4)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_poses.shape[1] == 4 & center_poses.shape[2] == 4
    assert pred_poses.shape[1] == 4 & pred_poses.shape[2] == 4
    assert pred_poses.shape[0] == pred_scores.shape[0]

    gt_R = center_poses[:, :3, :3]  # (K, 3, 3)
    # gt_t = center_poses[:,:3, 3]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )
    pred_R = pred_poses[:, :3, :3]  # (M, 3, 3)
    # pred_t = pred_poses[:,:3,3]

    R_diff = get_rot_diff(
        cp.repeat(gt_R[:, None, :, :], pred_R.shape[0], axis=1),
        cp.repeat(pred_R[None, :, :, :], gt_R.shape[0], axis=0),
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    nonconformity = cp.sum(R_diff * pred_prob[None, :], axis=1)  # (K, )

    return nonconformity

def normalized_max_R(
    center_poses: cp.ndarray,  # (K, 4, 4)
    pred_poses: cp.ndarray,  # (M, 4, 4)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:

    assert center_poses.shape[1] == 4 & center_poses.shape[2] == 4
    assert pred_poses.shape[1] == 4 & pred_poses.shape[2] == 4
    assert pred_poses.shape[0] == pred_scores.shape[0]

    gt_R = center_poses[:, :3, :3]  # (K, 3, 3)
    # gt_t = center_poses[:,:3, 3]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )
    pred_R = pred_poses[:, :3, :3]  # (M, 3, 3)
    # pred_t = pred_poses[:,:3,3]

    R_diff = cp.linalg.norm(
        gt_R[:, None, :, :] - pred_R[None, :, :, :], axis=(2, 3)
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_differences = pred_R[:, None, :, :] - pred_R[None, :, :, :]
    all_R_diffs = cp.linalg.norm(pred_R_differences, axis=(2, 3))
    mean_R_diffs = cp.mean(all_R_diffs)

    nonconformity = (
        cp.amax(R_diff * pred_prob[None, :], axis=1) / mean_R_diffs
    )  # (K, )

    return nonconformity

def normalized_mean_R(
    center_poses: cp.ndarray,  # (K, 4, 4)
    pred_poses: cp.ndarray,  # (M, 4, 4)
    pred_scores: cp.ndarray,  # (M, )
) -> cp.ndarray:
    assert center_poses.shape[1] == 4 & center_poses.shape[2] == 4
    assert pred_poses.shape[1] == 4 & pred_poses.shape[2] == 4
    assert pred_poses.shape[0] == pred_scores.shape[0]

    gt_R = center_poses[:, :3, :3]  # (K, 3, 3)
    # gt_t = center_poses[:,:3, 3]

    pred_prob = pred_scores / cp.sum(pred_scores)  # (M, )
    pred_R = pred_poses[:, :3, :3]  # (M, 3, 3)
    # pred_t = pred_poses[:,:3,3]

    R_diff = cp.linalg.norm(
        gt_R[:, None, :, :] - pred_R[None, :, :, :], axis=(2, 3)
    )  # (K, M)
    # t_diff = cp.linalg.norm(gt_t - pred_t, axis=1)

    # Calculate the pairwise difference:
    pred_R_differences = pred_R[:, None, :, :] - pred_R[None, :, :, :]
    all_R_diffs = cp.linalg.norm(pred_R_differences, axis=(2, 3))
    mean_R_diffs = cp.mean(all_R_diffs)

    nonconformity = (
        cp.sum(R_diff * pred_prob[None, :], axis=1) / mean_R_diffs
    )  # (K, )

    return nonconformity