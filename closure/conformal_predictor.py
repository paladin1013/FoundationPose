import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import cupy as cp
from closure.closure_foundation_pose import ClosureFoundationPose
from gpu_utils import get_rotation_dist, sample_convex_combination
from closure.miniball import miniball, rotation_miniball
import nonconformity_funcs as F


@dataclass
class Dataset:
    data_ids: npt.NDArray[np.int_]  # (N, )
    gt_Rs: npt.NDArray[np.float32]  # (N, 3, 3)
    gt_ts: npt.NDArray[np.float32]  # (N, 3)
    pred_Rs: npt.NDArray[np.float32]  # (N, M, 3, 3)
    pred_ts: npt.NDArray[np.float32]  # (N, M, 3)
    pred_scores: npt.NDArray[np.float32]  # (N, M)
    object_ids: npt.NDArray[np.int_]  # (N, )
    size: int


class ConfromalPredictor:
    def __init__(
        self,
        nonconformity_func_name: str,
        top_hypotheses_num: int = 10,
        seed=0,
        init_sample_num=1000,
    ):
        # dataset to be initialized in load_dataset
        self.dataset: Dataset
        self.calibration_set: Dataset
        self.test_set: Dataset

        assert nonconformity_func_name in F.__dict__.keys()
        print(f"Using {nonconformity_func_name} as nonconformity function")
        self.nonconformity_func_name = nonconformity_func_name
        self.nonconformity_func = getattr(F, nonconformity_func_name)
        self.top_hypotheses_num = top_hypotheses_num
        self.seed = 0
        np.random.seed(seed)
        cp.random.seed(seed)

        self.init_sample_num = init_sample_num

    def load_dataset(
        self,
        data_dir: str,
        dataset_name: str,
        object_ids: List[int],
        calibration_set_size: int,
    ):
        raw_dataset: Dict[int, Dict[str, npt.NDArray[np.float32]]] = {}
        for id in object_ids:
            data = {}
            data["gt_poses"] = np.load(f"{data_dir}/{dataset_name}_gt_poses_{id}.npy")
            data["pred_poses"] = np.load(
                f"{data_dir}/{dataset_name}_out_poses_{id}.npy"
            )[:, : self.top_hypotheses_num]
            data["pred_scores"] = np.load(
                f"{data_dir}/{dataset_name}_out_scores_{id}.npy"
            )[:, : self.top_hypotheses_num]
            raw_dataset[id] = data

        self.data_size = sum(
            [raw_dataset[id]["gt_poses"].shape[0] for id in object_ids]
        )
        gt_poses = np.concatenate([raw_dataset[id]["gt_poses"] for id in object_ids])
        pred_poses = np.concatenate(
            [raw_dataset[id]["pred_poses"] for id in object_ids]
        )
        self.dataset = Dataset(
            data_ids=np.arange(self.data_size),
            gt_Rs=gt_poses[:, :3, :3],
            gt_ts=gt_poses[:, :3, 3],
            pred_Rs=pred_poses[:, :, :3, :3],
            pred_ts=pred_poses[:, :, :3, 3],
            pred_scores=np.concatenate(
                [raw_dataset[id]["pred_scores"] for id in object_ids]
            ),
            object_ids=np.concatenate(
                [
                    np.ones(raw_dataset[id]["gt_poses"].shape[0]) * id
                    for id in object_ids
                ]
            ),
            size=self.data_size,
        )

        calibration_ids = np.random.choice(
            self.data_size, size=calibration_set_size, replace=False
        )
        self.calibration_set = Dataset(
            data_ids=calibration_ids,
            gt_Rs=self.dataset.gt_Rs[calibration_ids],
            gt_ts=self.dataset.gt_ts[calibration_ids],
            pred_Rs=self.dataset.pred_Rs[calibration_ids],
            pred_ts=self.dataset.pred_ts[calibration_ids],
            pred_scores=self.dataset.pred_scores[calibration_ids],
            object_ids=self.dataset.object_ids[calibration_ids],
            size=calibration_set_size,
        )

        test_ids = np.array(
            [i for i in range(self.data_size) if i not in calibration_ids]
        )
        self.test_set = Dataset(
            data_ids=test_ids,
            gt_Rs=self.dataset.gt_Rs[test_ids],
            gt_ts=self.dataset.gt_ts[test_ids],
            pred_Rs=self.dataset.pred_Rs[test_ids],
            pred_ts=self.dataset.pred_ts[test_ids],
            pred_scores=self.dataset.pred_scores[test_ids],
            object_ids=self.dataset.object_ids[test_ids],
            size=self.data_size - calibration_set_size,
        )

    def nonconformity_func(
        self,
        center_Rs: cp.ndarray,  # (K, 3, 3)
        center_ts: cp.ndarray,  # (K, 3)
        pred_Rs: cp.ndarray,  # (M, 3, 3)
        pred_ts: cp.ndarray,  # (M, 3)
        pred_scores: cp.ndarray,  # (M, )
    ) -> cp.ndarray:  # output: (K, )

        raise NotImplementedError("nonconformity_func is not initialized")

    def calibrate(self, epsilon: float):
        nonconformity_scores = np.zeros(self.calibration_set.size)
        for k in range(self.calibration_set.size):
            nonconformity_scores[k] = cp.asnumpy(
                self.nonconformity_func(
                    cp.array(self.calibration_set.gt_Rs[k][None, :, :]),
                    cp.array(self.calibration_set.gt_ts[k][None, :]),
                    cp.array(self.calibration_set.pred_Rs[k]),
                    cp.array(self.calibration_set.pred_ts[k]),
                    cp.array(self.calibration_set.pred_scores[k]),
                )
            )[0]
        nonconformity_threshold = float(np.quantile(nonconformity_scores, 1 - epsilon))

        print(f"{self.calibration_set.size=}, {nonconformity_threshold=}")
        return nonconformity_threshold

    def test_threshold(self, nonconformity_threshold: float):
        nonconformity_scores = np.zeros(self.test_set.size)
        for k in range(self.test_set.size):
            nonconformity_scores[k] = cp.asnumpy(
                self.nonconformity_func(
                    cp.array(self.test_set.gt_Rs[k][None, :, :]),
                    cp.array(self.test_set.gt_ts[k][None, :]),
                    cp.array(self.test_set.pred_Rs[k]),
                    cp.array(self.test_set.pred_ts[k]),
                    cp.array(self.test_set.pred_scores[k]),
                )
            )[0]
        test_epsilon = (
            np.sum(nonconformity_scores > nonconformity_threshold) / self.test_set.size
        )
        print(f"{self.test_set.size=}, {test_epsilon=}")
        return test_epsilon

    def predict(
        self,
        pred_Rs: npt.NDArray[np.float32],  # (M, 3, 3)
        pred_ts: npt.NDArray[np.float32],  # (M, 3)
        pred_scores: npt.NDArray[np.float32],  # (M)
        nonconformity_threshold: float,
    ) -> Tuple[npt.NDArray, npt.NDArray, float, float]:
        """
        Return:
            - minimax_center_R: (3, 3)
            - minimax_center_t: (3, )
            - max_rotation_error: float
            - max_translation_error: float
        """

        # First do sampling in the sets
        pred_Rs = cp.array(pred_Rs)
        pred_ts = cp.array(pred_ts)
        pred_scores = cp.array(pred_scores)

        center_Rs, center_ts = sample_convex_combination(
            cp.array(pred_Rs), cp.array(pred_ts), self.init_sample_num
        )  # (init_sample_num, 4, 4)
        nonconformity_scores = self.nonconformity_func(
            center_Rs, center_ts, pred_Rs, pred_ts, pred_scores
        )
        valid_center_Rs = center_Rs[nonconformity_scores < nonconformity_threshold]
        valid_center_ts = center_ts[nonconformity_scores < nonconformity_threshold]

        # Then do the rigid sim algorithms

        closure = ClosureFoundationPose(
            pred_Rs=pred_Rs,
            pred_ts=pred_ts,
            pred_scores=pred_scores,
            nonconformity_func_name=self.nonconformity_func_name,
            nonconformity_threshold=nonconformity_threshold,
            init_Rs=valid_center_Rs,
            init_ts=valid_center_ts,
            n_iterations=5,
            n_walks=20,
            base_ang_vel=0.5,
            base_lin_vel=0.2,
            decay_factor=0.5,
            n_time_steps=15,
            R_perturbation_scale=0.2,
            t_perturbation_scale=0.1,
            n_perturbations=150,
            n_optimal_perturbations=10,
            device_id=0,
        )
        Rs, ts = closure.run()
        Rs = cp.asnumpy(Rs)
        ts = cp.asnumpy(ts)

        # Finally get the minimax_center_R, minimax_center_t, max_rotation_errors, max_translation_errors
        minimax_center_R, max_rotation_error = rotation_miniball(Rs)
        minimax_center_t, max_translation_error = miniball(ts)
        return (
            minimax_center_R,
            minimax_center_t,
            max_rotation_error,
            max_translation_error,
        )

    def predict_testset(self, nonocnformaty_threshold: float):
        for k in range(self.test_set.size):
            (
                minimax_center_R,
                minimax_center_t,
                max_rotation_error,
                max_translation_error,
            ) = self.predict(
                self.test_set.pred_Rs[k],
                self.test_set.pred_ts[k],
                self.test_set.pred_scores[k],
                nonocnformaty_threshold,
            )

            # print("GT:")
            # print(self.test_set.gt_Rs[k])
            # print("Minimax center:")
            # print(minimax_center_R)
            minimax_center_err_R = get_rotation_dist(
                cp.array(minimax_center_R), cp.array(self.test_set.gt_Rs[k])
            )
            minimax_center_err_t = np.linalg.norm(
                minimax_center_t - self.test_set.gt_ts[k]
            )
            data = {}
            data["gt_Rs"] = self.test_set.gt_Rs[k]
            data["gt_ts"] = self.test_set.gt_ts[k]
            data["pred_Rs"] = self.test_set.pred_Rs[k]
            data["pred_ts"] = self.test_set.pred_ts[k]
            data["minimax_center_R"] = minimax_center_R
            data["minimax_center_t"] = minimax_center_t
            data["max_rotation_error"] = max_rotation_error
            data["max_translation_error"] = max_translation_error
            data["minimax_center_err_R"] = minimax_center_err_R
            data["minimax_center_err_t"] = minimax_center_err_t
            np.save(f"data/closure_test/test_result_{self.test_set.data_ids[k]}.npy", data, allow_pickle=True)
            print(
                f"{self.test_set.data_ids[k]=}, {max_rotation_error=:.4f}, {max_translation_error=:.4f}, {minimax_center_err_R=:.8f}, {minimax_center_err_t=:.8f}"
            )
            


if __name__ == "__main__":

    conformal_predictor = ConfromalPredictor(
        nonconformity_func_name="normalized_max_R",
        top_hypotheses_num=10,
        init_sample_num=1000,
    )
    conformal_predictor.load_dataset("data", "linemod", [1, 2, 4, 5, 6, 8, 9], 200)
    nonconformity_threshold = conformal_predictor.calibrate(epsilon=0.1)

    # test_epsilon = conformal_predictor.test_threshold(nonconformity_threshold)
    conformal_predictor.predict_testset(nonconformity_threshold)