from typing import Tuple
from cupy._core import ndarray
from closure.gpu_utils import get_successful_idx, get_top_k_perturbation_indices
from closure_base import ClosureBase
import cupy as cp

import nonconformity_funcs as F


class ClosureFoundationPose(ClosureBase):

    def __init__(
        self,
        pred_Rs: cp.ndarray,  # (M, 3, 3)
        pred_ts: cp.ndarray,  # (M, 3)
        pred_scores: cp.ndarray,  # (M, )
        nonconformity_func_name: str,
        nonconformity_threshold: float,
        **kwargs
    ):
        self.pred_Rs = pred_Rs
        self.pred_ts = pred_ts
        self.pred_scores = pred_scores
        assert nonconformity_func_name in F.__dict__.keys()
        self.nonconformity_func = getattr(F, nonconformity_func_name)
        self.nonconformity_threshold = nonconformity_threshold

        super().__init__(**kwargs)

    def nonconformity_func(
        self,
        center_Rs: cp.ndarray,  # (K, 3, 3)
        center_ts: cp.ndarray,  # (K, 3)
        pred_Rs: cp.ndarray,  # (M, 3, 3)
        pred_ts: cp.ndarray,  # (M, 3)
        pred_scores: cp.ndarray,  # (M, )
    ) -> cp.ndarray: ...

    def check_final_poses(
        self,
        Rs: cp.ndarray,  # (N, 3, 3)
        ts: cp.ndarray,  # (N, 3)
    ) -> cp.ndarray:
        nonconformity_scores = self.nonconformity_func(
            Rs, ts, self.pred_Rs, self.pred_ts, self.pred_scores
        )
        inside = nonconformity_scores < self.nonconformity_threshold
        return inside

    def get_final_poses(
        self,
        additional_Rs: cp.ndarray,  # (N, n_walks, 3, 3)
        additional_ts: cp.ndarray,  # (N, n_walks, 3)
    ):
        final_Rs: cp.ndarray = cp.matmul(additional_Rs, self.init_Rs[:, None, :, :])
        final_ts: cp.ndarray = (
            cp.matmul(additional_Rs, self.init_ts[:, None, :, None]).squeeze()
            + additional_ts
        )
        return final_Rs.reshape(-1, 3, 3), final_ts.reshape(-1, 3)

    def check_additional_poses(
        self,
        additional_Rs: cp.ndarray,  # (N, n_walks, 3, 3)
        additional_ts: cp.ndarray,  # (N, n_walks, 3)
    ) -> cp.ndarray:
        inside = self.check_final_poses(
            *self.get_final_poses(additional_Rs, additional_ts)
        ).reshape(additional_Rs.shape[:-2])
        return inside

    def get_nonconformity_residual(
        self,
        additional_Rs: cp.ndarray,  # (N, n_walks, 3, 3)
        additional_ts: cp.ndarray,  # (N, n_walks, 3)
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        nonconformity_scores = self.nonconformity_func(
            *self.get_final_poses(additional_Rs, additional_ts),
            self.pred_Rs,
            self.pred_ts,
            self.pred_scores
        ).reshape(
            additional_Rs.shape[:-2]
        )  # (N, n_walks)
        inside = nonconformity_scores < self.nonconformity_threshold  # (N, n_walks)
        residuals = self.nonconformity_threshold - nonconformity_scores  # (N, n_walks)
        return inside, residuals

    def sample_rotation_boundary(self) -> Tuple[cp.ndarray, cp.ndarray]:

        N = self.init_Rs.shape[0]
        R_movements = self.get_rotation_movements()  # (N, n_walks, n_time_steps, 3, 3)
        t_perturbations = (
            self.get_translation_perturbation()
        )  # (n_iterations, n_perturbations, 3)

        successful_additional_Rs = cp.zeros((N, self.n_walks, 3, 3))
        successful_additional_Rs[:, :, :, :] = cp.eye(3)[None, None, :, :]
        successful_additional_ts = cp.zeros((N, self.n_walks, 3))

        for iter_cnt in range(self.n_iterations):
            t_perturbation = t_perturbations[iter_cnt].squeeze()  # (n_perturbations, 3)

            # First stage: perturb translation for each case to get a few best t_perturbations
            trial_additional_Rs = cp.repeat(
                successful_additional_Rs[:, :, None, :, :],
                self.n_perturbations,
                axis=2,
            )  # (N, n_walks, n_perturbations, 3, 3)

            trial_additional_ts = (
                successful_additional_ts[:, :, None, :]
                + t_perturbation[None, None, :, :]
            )  # (N, n_walks, n_perturbations, 3)

            inside, residuals = self.get_nonconformity_residual(
                trial_additional_Rs.reshape(N, -1, 3, 3),
                trial_additional_ts.reshape(N, -1, 3),
            )

            top_k_indices = get_top_k_perturbation_indices(
                inside.reshape(N * self.n_walks, -1),
                residuals.reshape(N * self.n_walks, -1),
                self.n_optimal_perturbations,
            )
            best_t_perturbations = t_perturbation[top_k_indices].reshape(
                N, self.n_walks, self.n_optimal_perturbations, 3
            )

            # Second stage: apply the best perturbations with rotation in different scale
            original_trial_additional_Rs: cp.ndarray = cp.matmul(
                R_movements, successful_additional_Rs[:, :, None, :, :]
            )  # (N, n_walks, n_time_steps, 3, 3)

            trial_additional_Rs = cp.repeat(
                original_trial_additional_Rs[:, :, :, None, :, :],
                self.n_optimal_perturbations,
                axis=3,
            )  # (N, n_walks, n_time_steps, n_optimal_perturbations, 3, 3)

            original_trial_additional_ts = cp.repeat(
                successful_additional_ts[:, :, None, :],
                self.n_time_steps,
                axis=2,
            )  # (N, n_walks, n_time_steps, 3)

            trial_additional_ts = (
                original_trial_additional_ts[:, :, :, None, :]
                + best_t_perturbations[:, :, None, :, :]
            )  # (N, n_walks, n_time_steps, n_optimal_perturbations, 3)

            inside = self.check_additional_poses(
                trial_additional_Rs.reshape(N, -1, 3, 3),
                trial_additional_ts.reshape(N, -1, 3),
            )  # (N, n_walks*n_time_steps*n_optimal_perturbations)
            inside = inside.reshape(
                N * self.n_walks, self.n_time_steps, self.n_optimal_perturbations
            )  # (N*n_walks, n_time_steps, n_optimal_perturbations)
            best_valid_scales, best_valid_perturbations = get_successful_idx(
                inside
            )  # both (N*n_walks, )

            successful_additional_Rs = original_trial_additional_Rs.reshape(
                -1, self.n_time_steps, 3, 3
            )[cp.arange(N * self.n_walks), best_valid_scales, :, :].reshape(
                -1, self.n_walks, 3, 3
            )  # (N, n_walks, 3, 3)
            successful_additional_ts = trial_additional_ts.reshape(
                -1, self.n_time_steps, self.n_optimal_perturbations, 3
            )[
                cp.arange(N * self.n_walks),
                best_valid_scales,
                best_valid_perturbations,
                :,
            ].reshape(
                -1, self.n_walks, 3
            )  # (N, n_walks, 3)

        return successful_additional_Rs, successful_additional_ts
