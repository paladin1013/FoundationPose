from typing import Optional, Union
import cupy as cp
import numpy as np
from line_profiler import profile

def get_axis_angle(Rs: cp.ndarray):
    """Get the axis-angle representation of a rotation matrix"""
    assert Rs.shape[1:] == (3, 3)  # Rs: (N, 3, 3)
    cos_theta = (cp.trace(Rs, axis1=1, axis2=2) - 1) / 2  # (N, )
    theta = cp.arccos(cos_theta)  # (N, )
    close_to_0 = cp.isclose(theta, 0)

    axes = cp.zeros((Rs.shape[0], 3))
    axes[close_to_0, :] = cp.zeros((3,))

    valid_Rs = Rs[~close_to_0, :, :]  # (N0, 3, 3)
    valid_theta = theta[~close_to_0]  # (N0, )
    valid_axes = cp.array(  # (N0, 3)
        [
            (valid_Rs[:, 2, 1] - valid_Rs[:, 1, 2]) / (2 * cp.sin(valid_theta)),
            (valid_Rs[:, 0, 2] - valid_Rs[:, 2, 0]) / (2 * cp.sin(valid_theta)),
            (valid_Rs[:, 1, 0] - valid_Rs[:, 0, 1]) / (2 * cp.sin(valid_theta)),
        ]
    ).T
    axes[~close_to_0, :] = valid_axes
    return theta[:, None] * axes

@profile
def get_rotation_dist(Rs1: Union[cp.ndarray, np.ndarray], Rs2: Union[cp.ndarray, np.ndarray]):
    """Get the rotation difference between two rotation matrices. Two inputs should have the same shape (..., 3, 3)"""
    assert Rs1.shape == Rs2.shape  # Rs1, Rs2: (..., 3, 3)
    assert Rs1.shape[-2:] == (3, 3)
    input_numpy = False
    if isinstance(Rs1, np.ndarray):
        Rs1 = cp.array(Rs1)
        input_numpy = True
    if isinstance(Rs2, np.ndarray):
        Rs2 = cp.array(Rs2)
        input_numpy = True

    mul_R_diff = cp.multiply(Rs2, Rs1)  # (..., )
    trace_0 = cp.sum(mul_R_diff[..., :, 0], axis=-1)
    trace_1 = cp.sum(mul_R_diff[..., :, 1], axis=-1)
    trace_2 = cp.sum(mul_R_diff[..., :, 2], axis=-1)
    trace_R_diff = trace_0+trace_1+trace_2

    # The above method is much faster than due to memory access
    # trace_R_diff = cp.sum(mul_R_diff, axis=(-1, -2))

    cos_theta = (trace_R_diff - 1) / 2  # (..., )
    theta = cp.arccos(cp.clip(cos_theta, -1, 1))  # (..., )
    if input_numpy and isinstance(theta, cp.ndarray):
        theta = cp.asnumpy(theta)
    return theta

def project_to_SO3(Rs: cp.ndarray) -> cp.ndarray:
    """Project the rotation matrices to SO(3), Rs: (N, 3, 3) or (3, 3)"""
    U, _, Vt = cp.linalg.svd(Rs)
    # U: (N, 3, 3), Vt: (N, 3, 3)
    dets = cp.linalg.det(cp.matmul(U, Vt))
    if len(Rs.shape) == 2:
        Vt[2, :] *= dets
    elif len(Rs.shape) == 3:
        Vt[:, 2, :] *= dets[:, None]
    else:
        raise ValueError("Rs should be either (N, 3, 3) or (3, 3)")
    return cp.matmul(U, Vt)  # (N, 3, 3) or (3, 3)


def sample_convex_combination(
    Rs: cp.ndarray,  # (N, 3, 3)
    ts: cp.ndarray,  # (N, 3)
    sample_num: int,
):
    """Sample a convex combination of the input rotations and translations"""
    assert Rs.shape[1:] == (3, 3)  # Rs: (N, 3, 3)
    assert ts.shape[1:] == (3,)  # ts: (N, 3)

    N = Rs.shape[0]
    alphas = cp.random.rand(sample_num, N)  # (sample_num, N)
    alphas /= cp.sum(alphas, axis=1, keepdims=True)  # (sample_num, N)

    Rs = Rs[None, :, :, :]  # (1, N, 3, 3)
    ts = ts[None, :, :]  # (1, N, 3)

    sampled_Rs = cp.sum(Rs * alphas[:, :, None, None], axis=1)  # (sample_num, 3, 3)
    sampled_Rs = project_to_SO3(sampled_Rs)
    sampled_ts = cp.sum(ts * alphas[:, :, None], axis=1)  # (sample_num, 3)

    return sampled_Rs, sampled_ts  # (sample_num, 3, 3), (sample_num, 3)


def rotation_average(Rs: cp.ndarray):
    """
    Get the average rotation matrix from the input rotation matrices.
    Rs: (N, 3, 3)
    """
    assert Rs.shape[1:] == (3, 3), "Rotation matrix shape should be (N, 3, 3)"
    assert Rs.shape[0] >= 2, "Rotation matrix number should be larger than 2"

    # Get the average rotation matrix
    R_avg = cp.sum(Rs, axis=0)  # (3, 3)
    R_avg /= Rs.shape[0]
    # Project the average rotation matrix to SO(3)
    R_avg = project_to_SO3(R_avg)
    return R_avg


def matrix_exponential(A: cp.ndarray, terms: int = 5):
    # A: (N, 3, 3)
    expA = cp.repeat(cp.eye(3)[None, :, :], A.shape[0], axis=0)  # (N, 3, 3)
    powA = A.copy()
    for n in range(1, terms):
        expA += powA
        powA = cp.matmul(powA, A) / (n + 1)  # type: ignore

    return expA


def get_rotation(
    ang_vels: cp.ndarray,  # (N, 3)
):
    v0: cp.ndarray = ang_vels[:, 0]  # (N, )
    v1: cp.ndarray = ang_vels[:, 1]  # (N, )
    v2: cp.ndarray = ang_vels[:, 2]  # (N, )
    skew_symmetric_matrix = cp.zeros((ang_vels.shape[0], 3, 3))  # (N, 3, 3)
    skew_symmetric_matrix[:, 0, 1] = -v2
    skew_symmetric_matrix[:, 0, 2] = v1
    skew_symmetric_matrix[:, 1, 0] = v2
    skew_symmetric_matrix[:, 1, 2] = -v0
    skew_symmetric_matrix[:, 2, 0] = -v1
    skew_symmetric_matrix[:, 2, 1] = v0

    return matrix_exponential(skew_symmetric_matrix)  # (N, 3, 3)


def get_top_k_perturbation_indices(
    inside: cp.ndarray,  # (N*batch_size, perturbation_num)
    residual: cp.ndarray,  # (N*batch_size, perturbation_num)
    top_k: int,
):
    assert inside.shape == residual.shape
    assert len(inside.shape) == 2

    residual[~inside] = -cp.inf
    perturbation_num = inside.shape[1]

    indices = cp.argpartition(residual, perturbation_num - top_k, axis=1)[:, -top_k:]
    return indices


def get_successful_idx(
    inside: cp.ndarray,  # boolean array with shape (N, scale_num, perturbation_num)
):
    assert len(inside.shape) == 3
    assert cp.all(cp.any(inside, axis=(1, 2)))
    N, scale_num, perturbation_num = inside.shape

    valid_scales = cp.any(inside, axis=2)  # (N, scale_num)
    best_valid_scales = cp.argmax(valid_scales, axis=1)

    # best_valid_scales = cp.min(, axis=1)  # (N, )
    N_idx = cp.arange(N)
    valid_perturbations = inside[
        N_idx, best_valid_scales, :
    ]  # boolean, (N, perturbation_num)

    assert cp.all(cp.any(valid_perturbations, axis=1))
    best_valid_perturbations = cp.argmax(
        valid_perturbations, axis=1
    )  # (num_valid_perturbations)

    return best_valid_scales, best_valid_perturbations  # both (N, )
