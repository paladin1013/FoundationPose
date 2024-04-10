import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from closure.miniball_module import Miniball
import numpy as np
import numpy.typing as npt
import cupy as cp
from numba import njit


def miniball(points: npt.NDArray[np.float64]):
    """Compute the minimum bounding sphere of a set of points.

    Parameters
    ----------
    points : array_like, shape (npoints, ndim)
        The coordinates of the points.

    Returns
    -------
    center : ndarray, shape (ndim,)
        The center of the sphere.
    radius : float
        The radius of the sphere.
    validity : bool
    """
    if points.ndim != 2:
        raise ValueError("points must be 2-D")
    if points.shape[0] < points.shape[1]:
        raise ValueError("Need at least as many points as dimensions")
    dim = points.shape[1]
    mb = Miniball(dim, points)
    center = np.array(mb.center(), dtype=np.float64)
    radius = float(np.sqrt(mb.squared_radius()))
    # validity = bool(mb.is_valid())

    return center, radius


@njit(cache=True, fastmath=True)
def rotation_matrix_to_quaternion(Rs: npt.NDArray[np.float64]):
    """
    Convert a batch of rotation matrices to quaternions.

    :param Rs: An array of 3x3 rotation matrices, shape (n, 3, 3).
    :return: An array of quaternions [w, x, y, z], shape (n, 4).
    """
    original_dim = len(Rs.shape)
    assert Rs.shape[1:] == (3, 3)
    n = Rs.shape[0]
    Ks = np.zeros((n, 4, 4))
    R = Rs[:, :, :]

    # Ks[:, 0, 0] = 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]
    # Ks[:, 1, 1] = 1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]
    # Ks[:, 2, 2] = 1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]
    # Ks[:, 3, 3] = 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # Ks[:, 0, 1] = Ks[:, 1, 0] = R[:, 0, 1] + R[:, 1, 0]
    # Ks[:, 0, 2] = Ks[:, 2, 0] = R[:, 0, 2] + R[:, 2, 0]
    # Ks[:, 0, 3] = Ks[:, 3, 0] = R[:, 1, 2] - R[:, 2, 1]
    # Ks[:, 1, 2] = Ks[:, 2, 1] = R[:, 1, 2] + R[:, 2, 1]
    # Ks[:, 1, 3] = Ks[:, 3, 1] = R[:, 2, 0] - R[:, 0, 2]
    # Ks[:, 2, 3] = Ks[:, 3, 2] = R[:, 0, 1] - R[:, 1, 0]

    Ks[:, 0, 0] = R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]
    Ks[:, 1, 1] = R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]
    Ks[:, 2, 2] = R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]
    Ks[:, 3, 3] = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    Ks[:, 0, 1] = Ks[:, 1, 0] = R[:, 0, 1] + R[:, 1, 0]
    Ks[:, 0, 2] = Ks[:, 2, 0] = R[:, 0, 2] + R[:, 2, 0]
    Ks[:, 0, 3] = Ks[:, 3, 0] = R[:, 2, 1] - R[:, 1, 2]
    Ks[:, 1, 2] = Ks[:, 2, 1] = R[:, 1, 2] + R[:, 2, 1]
    Ks[:, 1, 3] = Ks[:, 3, 1] = R[:, 0, 2] - R[:, 2, 0]
    Ks[:, 2, 3] = Ks[:, 3, 2] = R[:, 1, 0] - R[:, 0, 1]
    Ks /= 3

    quaternions = np.zeros((n, 4))
    for i in range(n):
        eigenvalues, eigenvectors = np.linalg.eigh(Ks[i])
        quaternions[i] = [eigenvectors[3, np.argmax(eigenvalues)],eigenvectors[0, np.argmax(eigenvalues)],eigenvectors[1, np.argmax(eigenvalues)],eigenvectors[2, np.argmax(eigenvalues)]]
        # print(eigenvectors[2, np.argmax(eigenvalues)])

    return quaternions


@njit(cache=True, fastmath=True)
def quaternion_to_rotation_matrix(quaternions: npt.NDArray[np.float64]):
    """
    Convert a batch of quaternions to rotation matrices.
    """

    n = quaternions.shape[0]
    Rs = np.zeros((n, 3, 3))
    q = quaternions[:, :]
    Rs[:, 0, 0] = 1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2
    Rs[:, 1, 1] = 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2
    Rs[:, 2, 2] = 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2
    Rs[:, 0, 1] = 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3]
    Rs[:, 0, 2] = 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]
    Rs[:, 1, 2] = 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]
    Rs[:, 1, 0] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3]
    Rs[:, 2, 0] = 2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2]
    Rs[:, 2, 1] = 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1]

    # Rs[:, 0, 0] = 2 * q[:, 0] ** 2 + 2 * q[:, 1] ** 2 - 1
    # Rs[:, 1, 1] = 2 * q[:, 0] ** 2 + 2 * q[:, 0] ** 2 - 1
    # Rs[:, 2, 2] = 2 * q[:, 0] ** 2 - 2 * q[:, 3] ** 2 - 1
    # Rs[:, 0, 1] = 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3]
    # Rs[:, 0, 2] = 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]
    # Rs[:, 1, 2] = 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]
    # Rs[:, 1, 0] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3]
    # Rs[:, 2, 0] = 2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2]
    # Rs[:, 2, 1] = 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1]

    return Rs


@njit(cache=True)
def project_to_SO3(R: npt.NDArray[np.float64]):
    """Project the rotation matrix to SO(3)"""
    U, _, V = np.linalg.svd(R)
    if np.linalg.det(U @ V) < 0:
        V[2, :] = -V[2, :]
    U: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]
    return U @ V


@njit(cache=True)
def rotation_average(Rs: npt.NDArray[np.float64]):
    """Average a set of rotation matrices using the method in [1]
    [1] Horn, Berthold KP. "Closed-form solution of absolute orientation using unit quaternions." Journal of the optical society of America A 4.4 (1987): 629-642.
    """
    assert Rs.shape[1:] == (3, 3), "Rotation matrix shape should be (N, 3, 3)"
    assert Rs.shape[0] >= 2, "Rotation matrix number should be larger than 2"

    # Get the average rotation matrix
    R_avg = np.zeros((3, 3), dtype=np.float64)
    for R in Rs:
        R_avg += R
    R_avg /= Rs.shape[0]
    # Project the average rotation matrix to SO(3)
    R_avg = project_to_SO3(R_avg)
    return R_avg


def rotation_miniball(R_set: npt.NDArray[np.float64]):
    """
    Compute the minimum bounding sphere of a set of rotation matrices.

    Parameters
    ----------
    R_set : array_like, shape (npoints, 3, 3)
        The rotation matrices.

    Returns
    -------
    center : ndarray, shape (3, 3)
        The center of the sphere.
    radius : float
    validity : bool
    """
    assert R_set.shape[1:] == (3, 3)
    R_avg = rotation_average(R_set)

    q_set = rotation_matrix_to_quaternion(R_set)  # (n, 4)
    q_avg = rotation_matrix_to_quaternion(R_avg.reshape(1, 3, 3)).reshape(
        4,
    )  # (4, )

    signs = q_set @ q_avg  # (n,)
    q_set[signs < 0] *= -1

    [quat_center, quat_radius] = miniball(q_set)

    quat_center  = quat_center / np.linalg.norm(quat_center)

    # print(f"Quat Center Norm: {np.linalg.norm(quat_center)}")

    angs = q_set @ quat_center

    # print(f'Quat Center: {quat_center}')
    R_center = quaternion_to_rotation_matrix(quat_center.reshape(1, 4)).reshape(3, 3)

    # print(R_center)

    # print(f'Quat Radius: {quat_radius}')

    R_radius = np.arcsin(quat_radius) * 2

    data = {}
    data["R_set"] = R_set
    data["R_avg"] = R_avg
    data["q_set"] = q_set
    data["q_avg"] = q_avg
    data["quat_center"] = quat_center
    data["quat_radius"] = quat_radius
    data["R_center"] = R_center
    data["R_radius"] = R_radius
    data["signs"] = signs
    data["angs"] = angs

    # print(f'Max angs error: {min(abs(angs))}')
    # np.save("data/closure_test/rotation_miniball.npy", data, allow_pickle=True)

    return R_center, R_radius


if __name__ == "__main__":
    import time

    np.random.seed(1)

    num = 1
    R_set = np.random.randn(num, 3, 3)
    for i in range(num):
        R_set[i] = project_to_SO3(R_set[i])
    # start = time.time()
    # R_center, R_radius = rotation_miniball(R_set)
    # end = time.time()
    # print(f"Time: {end-start}")
    # print(R_center)
    # print(R_radius)


    # R_test = project_to_SO3(np.random.randn(3, 3))

    q_test = rotation_matrix_to_quaternion(R_set)
    # q_test[0,3] = -q_test[0,3]
    R_test_recover = quaternion_to_rotation_matrix(q_test.reshape(1, 4)).reshape(3, 3)

    print(R_set)
    print(q_test)
    print(R_test_recover)