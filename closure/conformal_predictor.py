import datetime
import json
import os
import sys
import time

from data_utils import convert_npy_to_mat
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import cupy as cp
from closure.closure_foundation_pose import ClosureFoundationPose
from gpu_utils import get_rotation_dist, sample_convex_combination
from closure.miniball import miniball, rotation_miniball
import nonconformity_funcs as F
import argparse
import pytz
@dataclass
class Dataset:
    data_ids: npt.NDArray[np.int_]  # (N, )
    gt_Rs: npt.NDArray[np.float32]  # (N, 3, 3)
    gt_ts: npt.NDArray[np.float32]  # (N, 3)
    pred_Rs: npt.NDArray[np.float32]  # (N, M, 3, 3)
    pred_ts: npt.NDArray[np.float32]  # (N, M, 3)
    pred_scores: npt.NDArray[np.float32]  # (N, M)
    object_ids: npt.NDArray[np.int_]  # (N, )
    image_ids: npt.NDArray[np.int_]  # (N, )
    size: int


class ConformalPredictor:
    def __init__(
        self,
        nonconformity_func_name: str,
        closure_params: Dict[str, Any],
        init_sample_num: int,
        top_hypotheses_num: int,
        seed: int,
    ):
        # dataset to be initialized in load_dataset
        self.dataset: Dataset
        self.calibration_set: Dataset
        self.test_set: Dataset

        self.nonconformity_func_name = nonconformity_func_name
        self.top_hypotheses_num = top_hypotheses_num
        self.seed = seed
        self.closure_params = closure_params
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
                    np.ones(raw_dataset[id]["gt_poses"].shape[0], dtype=np.int32) * id
                    for id in object_ids
                ]
            ),
            image_ids=np.concatenate(
                [
                    np.arange(raw_dataset[id]["gt_poses"].shape[0], dtype=np.int32)
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
            image_ids=self.dataset.image_ids[calibration_ids],
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
            image_ids=self.dataset.image_ids[test_ids],
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
        if "Rt" in self.nonconformity_func_name:
            R_ratio = self.calibrated_R_ratio
            t_ratio = self.calibrated_t_ratio
        elif "R" in self.nonconformity_func_name:
            R_ratio = 1
            t_ratio = 0
        elif "t" in self.nonconformity_func_name:
            R_ratio = 0
            t_ratio = 1
        else:
            raise ValueError("Invalid nonconformity function name")
        return F.nonconformity_func(
            center_Rs,
            center_ts,
            pred_Rs,
            pred_ts,
            pred_scores,
            aggregate_method="mean" if "mean" in self.nonconformity_func_name else "max",
            normalize=True if "normalized" in self.nonconformity_func_name else False,
            R_ratio = R_ratio,
            t_ratio = t_ratio,
        )

    def calibrate(self, epsilon: float):
        if "Rt" not in self.nonconformity_func_name:
            # Only single component
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
        else:
            # Two components
            nonconformity_scores_R = np.zeros(self.calibration_set.size)
            nonconformity_scores_t = np.zeros(self.calibration_set.size)
            for k in range(self.calibration_set.size):
                nonconformity_scores_R[k] = cp.asnumpy(
                    F.nonconformity_func(
                        cp.array(self.calibration_set.gt_Rs[k][None, :, :]),
                        cp.array(self.calibration_set.gt_ts[k][None, :]),
                        cp.array(self.calibration_set.pred_Rs[k]),
                        cp.array(self.calibration_set.pred_ts[k]),
                        cp.array(self.calibration_set.pred_scores[k]),
                        aggregate_method="mean" if "mean" in self.nonconformity_func_name else "max",
                        normalize=True if "normalized" in self.nonconformity_func_name else False,
                        R_ratio = 1,
                        t_ratio = 0,
                    )
                )[0]
                nonconformity_scores_t[k] = cp.asnumpy(
                    F.nonconformity_func(
                        cp.array(self.calibration_set.gt_Rs[k][None, :, :]),
                        cp.array(self.calibration_set.gt_ts[k][None, :]),
                        cp.array(self.calibration_set.pred_Rs[k]),
                        cp.array(self.calibration_set.pred_ts[k]),
                        cp.array(self.calibration_set.pred_scores[k]),
                        aggregate_method="mean" if "mean" in self.nonconformity_func_name else "max",
                        normalize=True if "normalized" in self.nonconformity_func_name else False,
                        R_ratio = 0,
                        t_ratio = 1,
                    )
                )[0]
            nonconformity_threshold_R = float(np.quantile(nonconformity_scores_R, 1 - epsilon))
            nonconformity_threshold_t = float(np.quantile(nonconformity_scores_t, 1 - epsilon))
            self.calibrated_R_ratio = 1/nonconformity_threshold_R
            self.calibrated_t_ratio = 1/nonconformity_threshold_t
            print(f"{self.calibrated_R_ratio=}, {self.calibrated_t_ratio=}")
            # Recalibrate with correct ratios
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
    ) -> Tuple[npt.NDArray, npt.NDArray, float, float, npt.NDArray, npt.NDArray]:
        """
        Return:
            - minimax_center_R: (3, 3)
            - minimax_center_t: (3, )
            - error_bound_R: float
            - error_bound_t: float
            - R_set: (N, 3, 3)
            - t_set: (N, 3)
        """

        # First do sampling in the sets
        pred_Rs = cp.array(pred_Rs)
        pred_ts = cp.array(pred_ts)
        pred_scores = cp.array(pred_scores)

        center_Rs, center_ts = sample_convex_combination(
            cp.array(pred_Rs), cp.array(pred_ts), self.init_sample_num
        )  # (init_sample_num, 3, 3), (init_sample_num, 3)
        nonconformity_scores = self.nonconformity_func(
            center_Rs, center_ts, pred_Rs, pred_ts, pred_scores
        )
        # print(f"{cp.sum(nonconformity_scores < nonconformity_threshold)=}, {nonconformity_scores.size=}")
        valid_center_Rs = center_Rs[nonconformity_scores < nonconformity_threshold]
        valid_center_ts = center_ts[nonconformity_scores < nonconformity_threshold]
        if valid_center_Rs.size == 0:
            raise ValueError("No valid center poses")
        if valid_center_Rs.size == 1:
            raise ValueError("Only 1 valid center poses")

        # Then do the rigid sim algorithms

        closure = ClosureFoundationPose(
            pred_Rs=pred_Rs,
            pred_ts=pred_ts,
            pred_scores=pred_scores,
            nonconformity_func_name=self.nonconformity_func_name,
            nonconformity_threshold=nonconformity_threshold,
            calibrated_R_ratio=self.calibrated_R_ratio,
            calibrated_t_ratio=self.calibrated_t_ratio,
            init_Rs=valid_center_Rs,
            init_ts=valid_center_ts,
            **self.closure_params,
        )
        
        Rs, ts = closure.run()
        Rs = cp.asnumpy(Rs)
        ts = cp.asnumpy(ts)

        # Finally get the minimax_center_R, minimax_center_t, error_bound_Rs, error_bound_ts
        minimax_center_R, error_bound_R = rotation_miniball(Rs)
        minimax_center_t, error_bound_t = miniball(ts)

        return (
            minimax_center_R,
            minimax_center_t,
            error_bound_R,
            error_bound_t,
            Rs,
            ts,
        )

    def predict_testset(self, nonocnformaty_threshold: float):

        time_zone = pytz.timezone("America/Los_Angeles")
        
        params = {}
        params.update(self.closure_params)
        params["nonconformity_threshold"] = nonocnformaty_threshold
        params["nonconformity_func_name"] = self.nonconformity_func_name
        params["top_hypotheses_num"] = self.top_hypotheses_num
        params["init_sample_num"] = self.init_sample_num
        params["seed"] = self.seed
        params["R_ratio"] = self.calibrated_R_ratio
        params["t_ratio"] = self.calibrated_t_ratio
        params["calibration_ids"] = self.calibration_set.data_ids.tolist()
        params["test_ids"] = self.test_set.data_ids.tolist()
        params["data_size"] = self.data_size

        time_stamp_str = datetime.datetime.fromtimestamp(time.time(), tz=time_zone).strftime("%Y%m%d_%H%M%S")
        json.dump(params, open(f"data/closure_data/{time_stamp_str}_predict_params.json", "w"))


        # for k in range(self.test_set.size):
        predict_data = []
        for object_id in np.unique(self.test_set.object_ids):
            indices = list(np.where(self.test_set.object_ids == object_id)[0])
            for k in indices:
                start_time = time.monotonic()
                try:
                    (
                        minimax_center_R,
                        minimax_center_t,
                        error_bound_R,
                        error_bound_t,
                        R_set,
                        t_set,
                    ) = self.predict(
                        self.test_set.pred_Rs[k],
                        self.test_set.pred_ts[k],
                        self.test_set.pred_scores[k],
                        nonocnformaty_threshold,
                    )
                except ValueError:
                    print(f"Obj {self.test_set.object_ids[k]} img {self.test_set.image_ids[k]}: No valid center poses")
                    continue

                time_cost = time.monotonic() - start_time
                minimax_center_err_R = get_rotation_dist(
                    minimax_center_R, self.test_set.gt_Rs[k]
                )
                minimax_center_err_t = np.linalg.norm(
                    minimax_center_t - self.test_set.gt_ts[k]
                )
                foundation_pose_err_R = get_rotation_dist(
                    self.test_set.pred_Rs[k, 0], self.test_set.gt_Rs[k]
                )
                foundation_pose_err_t = np.linalg.norm(
                    self.test_set.pred_ts[k, 0] - self.test_set.gt_ts[k]
                )
                data = {}
                data["object_id"] = self.test_set.object_ids[k]
                data["image_id"] = self.test_set.image_ids[k]
                data["data_id"] = self.test_set.data_ids[k]
                data["gt_Rs"] = self.test_set.gt_Rs[k]
                data["gt_ts"] = self.test_set.gt_ts[k]
                data["pred_Rs"] = self.test_set.pred_Rs[k]
                data["pred_ts"] = self.test_set.pred_ts[k]
                data["pred_scores"] = self.test_set.pred_scores[k]
                data["minimax_center_R"] = minimax_center_R
                data["minimax_center_t"] = minimax_center_t
                data["minimax_center_err_R"] = minimax_center_err_R
                data["minimax_center_err_t"] = minimax_center_err_t
                data["foundation_pose_err_R"] = foundation_pose_err_R
                data["foundation_pose_err_t"] = foundation_pose_err_t
                data["error_bound_R"] = error_bound_R
                data["error_bound_t"] = error_bound_t
                data["R_set"] = R_set
                data["t_set"] = t_set
                data["time_cost"] = time_cost

                nonconformity_score = cp.asnumpy(
                    self.nonconformity_func(
                        cp.array(self.test_set.gt_Rs[k][None, :, :]),
                        cp.array(self.test_set.gt_ts[k][None, :]),
                        cp.array(self.test_set.pred_Rs[k]),
                        cp.array(self.test_set.pred_ts[k]),
                        cp.array(self.test_set.pred_scores[k]),
                    )
                )[0]
                data["purse_covered"] = nonconformity_score < nonocnformaty_threshold

                predict_data.append(data)
                # np.save(f"data/closure_test/test_result_{self.test_set.data_ids[k]}.npy", data, allow_pickle=True)
                print(
                    f"Obj {self.test_set.object_ids[k]} img {self.test_set.image_ids[k]}: time {data['time_cost']:.3f}s, {error_bound_R=:.4f}, err_R: {minimax_center_err_R:.4f}/{foundation_pose_err_R:.4f}"
                )

            np.save(f"data/closure_data/{time_stamp_str}_object_{object_id}.npy", predict_data, allow_pickle=True)
            # convert_npy_to_mat(f"data/closure_data/{time_stamp_str}_object_{object_id}.npy", f"data/closure_data/{time_stamp_str}_object_{object_id}.mat")

            


if __name__ == "__main__":

    # for name in ["max_R", "max_t", "mean_R", "mean_t", "normalized_max_R", "normalized_max_t", "normalized_mean_R", "normalized_mean_t"]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonconformity_func", type=str)
    args = parser.parse_args()

    closure_params = {
        "n_iterations": 5,
        "n_walks": 20,
        "base_ang_vel": 0.5,
        "base_lin_vel": 0.2,
        "decay_factor": 0.5,
        "n_time_steps": 15,
        "R_perturbation_scale": 0.2,
        "t_perturbation_scale": 0.1,
        "n_perturbations": 150,
        "n_optimal_perturbations": 10,
        "device_id": 0,
    }
    conformal_predictor = ConformalPredictor(
        nonconformity_func_name=args.nonconformity_func,
        closure_params=closure_params,
        top_hypotheses_num=10,
        init_sample_num=200,
        seed=0,
    )

    # conformal_predictor.load_dataset("data", "linemod", [9], 200)

    conformal_predictor.load_dataset("data", "linemod", [1, 2, 4, 5, 6, 8, 9], 200)
    nonconformity_threshold = conformal_predictor.calibrate(epsilon=0.1)
    # test_epsilon = conformal_predictor.test_threshold(nonconformity_threshold)
    conformal_predictor.predict_testset(nonconformity_threshold)