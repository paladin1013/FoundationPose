from abc import abstractmethod
from typing import Tuple, Union
import cupy as cp
import time
import numpy as np
import numpy.typing as npt

from closure.gpu_utils import get_axis_angle, get_rotation, rotation_average


class ClosureBase:
    def __init__(
        self,
        n_iterations: int,
        n_walks: int,
        init_Rs: cp.ndarray,  # (N, 3, 3)
        init_ts: cp.ndarray,  # (N, 3, )
        base_ang_vel: float,
        base_lin_vel: float,
        decay_factor: float,
        n_time_steps: int,
        R_perturbation_scale: float,
        t_perturbation_scale: float,
        n_perturbations: int,
        n_optimal_perturbations: int,
        device_id: int,
    ):
        assert init_Rs.shape[1:] == (3, 3), f"{init_Rs.shape=}"
        assert init_ts.shape[1:] == (3,), f"{init_ts.shape=}"

        self.n_iterations = n_iterations
        self.n_walks = n_walks
        cp.cuda.Device(device_id).use()

        self.init_Rs = init_Rs  # (N, 3, 3)
        self.init_ts = init_ts  # (N, 3)

        self.base_ang_vel = base_ang_vel
        self.base_lin_vel = base_lin_vel
        self.decay_factor = decay_factor
        self.n_time_steps = n_time_steps
        self.R_perturbation_scale = R_perturbation_scale
        self.t_perturbation_scale = t_perturbation_scale
        self.n_perturbations = n_perturbations
        self.n_optimal_perturbations = n_optimal_perturbations

        assert cp.all(
            self.check_final_poses(self.init_Rs, self.init_ts)
        ), "Initial poses does not satisfy PURSE constraints"

    @abstractmethod
    def check_final_poses(
        self,
        Rs: cp.ndarray,  # (N, 3, 3)
        ts: cp.ndarray,  # (N, 3)
    ) -> cp.ndarray:
        """
        Return:
            inside: cp.ndarray[bool], (N, )
        """
        raise NotImplementedError

    @abstractmethod
    def check_additional_poses(
        self,
        additional_Rs: cp.ndarray,  # (N, n_walks, 3, 3)
        additional_ts: cp.ndarray,  # (N, n_walks, 3)
    ) -> cp.ndarray:
        """
        Return:
            inside: cp.ndarray[bool], (N, n_walks)
        """
        raise NotImplementedError

    def get_rotation_movements(self) -> cp.ndarray:
        """
        Return:
            R_movements: cp.ndarray, (N, n_walks, n_time_steps, 3, 3)
        """
        N = self.init_Rs.shape[0]
        R_center = cp.array(rotation_average(self.init_Rs))  # (3, 3)
        R_diff = cp.matmul(self.init_Rs, R_center.T[None, :, :])  # (N, 3, 3)
        ax_ang = get_axis_angle(R_diff)  # (N, 3)
        base_ang_vel = (
            ax_ang
            / cp.linalg.norm(ax_ang, axis=1, keepdims=True)
            * self.base_ang_vel
        )  # (N, 3)
        rand_ang_vel = (
            cp.random.rand(N, self.n_walks, 3) * 2 - 1
        )  # (N, n_walks, 3)
        rand_ang_vel = rand_ang_vel / cp.linalg.norm(
            rand_ang_vel, axis=2, keepdims=True
        )  # (N, n_walks, 3)
        rand_ang_vel = rand_ang_vel * self.base_ang_vel / 2  # (N, n_walks, 3)

        ang_vel = base_ang_vel[:, None, :] + rand_ang_vel  # (N, n_walks, 3)
        scales = cp.array(
            [self.decay_factor**i for i in range(self.n_time_steps)]
        )  # (n_time_steps, )
        scales[-1] = 0

        scaled_ang_vel = (
            ang_vel[:, :, None, :] * scales[None, None, :, None]
        )  # (N, n_walks, n_time_steps, 3)

        R_movements = get_rotation(scaled_ang_vel.reshape(-1, 3)).reshape(
            (N, self.n_walks, self.n_time_steps, 3, 3)
        )  # (N, n_walks, n_time_steps, 3, 3)

        return R_movements


    def get_translation_movements(self) -> cp.ndarray:
        """
        Return:
            t_movements: cp.ndarray, (N, n_walks, n_time_steps, 3)
        """
        # t_movements = cp.zeros((init_ts.shape[0], n_walks, n_time_steps, 3))
        N = self.init_ts.shape[0]
        t_center = cp.mean(self.init_ts, axis=0)  # (3, )
        t_diff = self.init_ts - t_center  # (N, 3)

        base_lin_vel = (
            t_diff
            / cp.linalg.norm(t_diff, axis=1, keepdims=True)
            * self.base_lin_vel
        )  # (N, 3)
        rand_lin_vel = (
            cp.random.rand(N, self.n_walks, 3) * 2 - 1
        )  # (N, n_walks, 3)
        rand_lin_vel = rand_lin_vel / cp.linalg.norm(
            rand_lin_vel, axis=2, keepdims=True
        )  # (N, n_walks, 3)
        rand_lin_vel = rand_lin_vel * self.base_lin_vel / 2  # (N, n_walks, 3)

        lin_vel = base_lin_vel[:, None, :] + rand_lin_vel  # (N, n_walks, 3)

        scales = cp.array(
            [self.decay_factor**i for i in range(self.n_time_steps)]
        )  # (n_time_steps, )
        scales[-1] = 0
        scaled_lin_vel = (
            lin_vel[:, :, None, :] * scales[None, None, :, None]
        )  # (N, n_walks, n_time_steps, 3)

        return scaled_lin_vel

    def get_rotation_perturbation(self):
        ax_ang = (
            cp.random.rand(self.n_iterations, self.n_perturbations, 3) * 2 - 1
        )  # (n_iterations, n_perturbations, 3)
        ax_ang = ax_ang / cp.linalg.norm(
            ax_ang, axis=2, keepdims=True
        )  # (n_iterations, n_perturbations, 3)
        ax_ang = (
            ax_ang
            * self.R_perturbation_scale
            * cp.random.rand(self.n_iterations, self.n_perturbations, 1)
        )  # (n_iterations, n_perturbations, 3)
        temp_R_perturbation = get_rotation(ax_ang.reshape(-1, 3)).reshape(
            (self.n_iterations, self.n_perturbations, 3, 3)
        )  # (n_iterations, n_perturbations, 3, 3)

        # Set an identity perturbation for each step
        temp_R_perturbation[:, 0, :, :] = cp.eye(3)[None, None, :, :]

        return temp_R_perturbation

    def get_translation_perturbation(self):
        temp_t_perturbation = (
            cp.random.rand(self.n_iterations, self.n_perturbations, 3) * 2 - 1
        )
        temp_t_perturbation = temp_t_perturbation / cp.linalg.norm(
            temp_t_perturbation, axis=2, keepdims=True
        )  # (n_iterations, n_perturbations, 3)
        temp_t_perturbation = (
            temp_t_perturbation
            * self.t_perturbation_scale
            * cp.random.rand(self.n_iterations, self.n_perturbations, 1)
        )
        # Set a 0 perturbation for each step
        temp_t_perturbation[:, 0, :] = cp.zeros((self.n_iterations, 3))

        return temp_t_perturbation  # (n_iterations, n_perturbations, 3)


    @abstractmethod
    def get_final_poses(
        self,
        additional_Rs: cp.ndarray,
        additional_ts: cp.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            Rs: cp.ndarray, (N, 3, 3)
            ts: cp.ndarray, (N, 3)
        """

        raise NotImplementedError

    @abstractmethod
    def get_nonconformity_residual(
        self,
        additional_Rs: cp.ndarray,  # (N, n_walks, 3, 3)
        additional_ts: cp.ndarray,  # (N, n_walks, 3)
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            inside: cp.ndarray[bool], (N, n_walks)
            min_residual: cp.ndarray[float], (N, n_walks)
        """
        raise NotImplementedError

    @abstractmethod
    def sample_rotation_boundary(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            successful_additional_Rs: cp.ndarray, (N, n_walks, 3, 3)
            successful_additional_ts: cp.ndarray, (N, n_walks, 3)
        """
        raise NotImplementedError

    @abstractmethod
    def sample_translation_boundary(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            successful_additional_Rs: cp.ndarray, (N, n_walks, 3, 3)
            successful_additional_ts: cp.ndarray, (N, n_walks, 3)
        """
        raise NotImplementedError

    def run_sampling(self):

        additional_Rs_r, additional_ts_r = self.sample_rotation_boundary()
        # additional_Rs_t, additional_ts_t = self.sample_translation_boundary()
        # additional_Rs = cp.concatenate((additional_Rs_r, additional_Rs_t), axis=1)
        # additional_ts = cp.concatenate((additional_ts_r, additional_ts_t), axis=1)
        additional_Rs = additional_Rs_r
        additional_ts = additional_ts_r
        # additional_Rs = additional_Rs_t
        # additional_ts = additional_ts_t
        return additional_Rs, additional_ts
