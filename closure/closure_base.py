from typing import Tuple, Union
import cupy as cp
import time
import numpy as np
import numpy.typing as npt


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


        assert cp.all(
            self.check_final_poses(self.init_Rs, self.init_ts)
        ), "Initial poses does not satisfy PURSE constraints"
        
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

        raise NotImplementedError
    
    def get_translation_movements(self) -> cp.ndarray:
        """
        Return:
            t_movements: cp.ndarray, (N, n_walks, n_time_steps, 3)
        """
        raise NotImplementedError

    def get_rotation_perturbations(self):
        """
        Return:
            R_perturbations: cp.ndarray, (N, n_walks, n_perturbations, 3, 3)
        """
        raise NotImplementedError
    
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

    def get_minimum_residual(
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


    def sample_rotation_boundary(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            successful_additional_Rs: cp.ndarray, (N, n_walks, 3, 3)
            successful_additional_ts: cp.ndarray, (N, n_walks, 3)
        """
        raise NotImplementedError
        

    def sample_translation_boundary(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return:
            successful_additional_Rs: cp.ndarray, (N, n_walks, 3, 3)
            successful_additional_ts: cp.ndarray, (N, n_walks, 3)
        """
        raise NotImplementedError


    def run_sampling(self):

        
        additional_Rs_r, additional_ts_r = self.sample_rotation_boundary()
        additional_Rs_t, additional_ts_t = self.sample_translation_boundary()
        additional_Rs = cp.concatenate((additional_Rs_r, additional_Rs_t), axis=1)
        additional_ts = cp.concatenate((additional_ts_r, additional_ts_t), axis=1)
        # additional_Rs = additional_Rs_r
        # additional_ts = additional_ts_r
        # additional_Rs = additional_Rs_t
        # additional_ts = additional_ts_t
        return additional_Rs, additional_ts
