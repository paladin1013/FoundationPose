from re import U
import cupy as cp


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


def get_rot_diff(Rs1: cp.ndarray, Rs2: cp.ndarray):
    """Get the rotation difference between two rotation matrices"""
    assert Rs1.shape == Rs2.shape  # Rs1, Rs2: (..., 3, 3)
    assert Rs1.shape[-2:] == (3, 3)
    axes = list(range(Rs1.ndim))
    axes = axes[:-2] + [axes[-1], axes[-2]]
    Rs1_t = Rs1.transpose(axes)  # (..., 3, 3)
    R_diff = cp.matmul(Rs2, Rs1_t)  # (..., 3, 3)
    cos_theta = (cp.trace(R_diff, axis1=-2, axis2=-1) - 1) / 2  # (..., )
    theta = cp.arccos(cp.clip(cos_theta, -1, 1))  # (..., )
    return theta


def project_to_SO3(Rs: cp.ndarray) -> cp.ndarray:
    """Project the rotation matrices to SO(3), Rs: (N, 3, 3)"""
    U, _, Vt = cp.linalg.svd(Rs)
    dets = cp.linalg.det(cp.matmul(U, Vt))
    Vt[:, 2] *= dets[:, None]
    return cp.matmul(U, Vt)


def sample_convex_combination(
    Rs: cp.ndarray,  # (N, 3, 3)
    ts: cp.ndarray,  # (N, 3)
    num: int,
):
    """Sample a convex combination of the input rotations and translations"""
    assert Rs.shape[1:] == (3, 3)  # Rs: (N, 3, 3)
    assert ts.shape[1:] == (3,)  # ts: (N, 3)

    N = Rs.shape[0]
    alphas = cp.random.rand(num, N)  # (num, N)
    alphas /= cp.sum(alphas, axis=0, keepdims=True)  # (num, N)

    Rs = Rs[:, None, :, :]  # (N, 1, 3, 3)
    ts = ts[:, None, :]  # (N, 1, 3)

    sampled_Rs = cp.sum(Rs * alphas[:, :, None, None], axis=1)  # (num, 3, 3)
    sampled_Rs = project_to_SO3(sampled_Rs)
    sampled_ts = cp.sum(ts * alphas[:, :, None], axis=1)  # (num, 3)

    return sampled_Rs, sampled_ts  # (num, 3, 3), (num, 3)
