from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import cupy as cp
from gpu_utils import sample_convex_combination
import nonconformity_funcs as F

@dataclass
class Dataset:
    data_ids: npt.NDArray[np.int_]  # (N, )
    gt_poses: npt.NDArray[np.float32]  # (N, 4, 4)
    pred_poses: npt.NDArray[np.float32]  # (N, M, 4, 4)
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
        self.nonconformity_func = getattr(F, nonconformity_func_name)
        self.top_hypotheses_num = top_hypotheses_num
        self.seed = 0
        np.random.seed(seed)

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
        self.dataset = Dataset(
            data_ids=np.arange(self.data_size),
            gt_poses=np.concatenate([raw_dataset[id]["gt_poses"] for id in object_ids]),
            pred_poses=np.concatenate(
                [raw_dataset[id]["pred_poses"] for id in object_ids]
            ),
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
            gt_poses=self.dataset.gt_poses[calibration_ids],
            pred_poses=self.dataset.pred_poses[calibration_ids],
            pred_scores=self.dataset.pred_scores[calibration_ids],
            object_ids=self.dataset.object_ids[calibration_ids],
            size=calibration_set_size,
        )

        test_ids = np.array(
            [i for i in range(self.data_size) if i not in calibration_ids]
        )
        self.test_set = Dataset(
            data_ids=test_ids,
            gt_poses=self.dataset.gt_poses[test_ids],
            pred_poses=self.dataset.pred_poses[test_ids],
            pred_scores=self.dataset.pred_scores[test_ids],
            object_ids=self.dataset.object_ids[test_ids],
            size=self.data_size - calibration_set_size,
        )

    def nonconformity_func(
        self,
        center_poses: cp.ndarray,  # (K, 4, 4)
        pred_poses: cp.ndarray,  # (M, 4, 4)
        pred_scores: cp.ndarray,  # (M, )
    ) -> cp.ndarray:  # output: (K, )

        raise NotImplementedError

    def calibrate(self, epsilon: float):
        nonconformity_scores = np.zeros(self.calibration_set.size)
        for k in range(self.calibration_set.size):
            nonconformity_scores[k] = cp.asnumpy(
                self.nonconformity_func(
                    cp.array(self.calibration_set.gt_poses[k][None, :, :]),
                    cp.array(self.calibration_set.pred_poses[k]),
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
                    cp.array(self.test_set.gt_poses[k][None, :, :]),
                    cp.array(self.test_set.pred_poses[k]),
                    cp.array(self.test_set.pred_scores[k]),
                )
            )[0]
        test_epsilon = np.sum(nonconformity_scores > nonconformity_threshold) / self.test_set.size
        print(f"{self.test_set.size=}, {test_epsilon=}")
        return test_epsilon

    def predict(
        self,
        pred_poses: npt.NDArray[np.float32],  # (M, 4, 4)
        pred_scores: npt.NDArray[np.float32],  # (M)
        nonconformity_threshold: float,
    ) -> Tuple[npt.NDArray[np.float32], float, float]:
        """
        Return:
            - minimax_center_pose: (4, 4)
            - max_rotation_error: float
            - max_translation_error: float
        """

        # First do sampling in the sets

        center_poses = sample_convex_combination(
            cp.array(pred_poses), self.init_sample_num
        )  # (init_sample_num, 4, 4)
        nonconformity_scores = self.nonconformity_func(
            center_poses, pred_poses, pred_scores
        )
        valid_center_poses = center_poses[
            nonconformity_scores < nonconformity_threshold
        ]

        print(f"{valid_center_poses.shape=}")

        # Then do the rigid sim algorithms

        # Finally get the minimax_center_poses, max_rotation_errors, max_translation_errors
        minimax_center_poses = np.zeros((4, 4), dtype=np.float32)
        max_rotation_error = 0
        max_translation_error = 0
        return minimax_center_poses, max_rotation_error, max_translation_error


if __name__ == "__main__":

    conformal_predictor = ConfromalPredictor(nonconformity_func_name="mean_R", top_hypotheses_num = 10, init_sample_num = 1000)
    conformal_predictor.load_dataset("data", "linemod", [1, 2, 4, 5, 6, 8, 9], 200)
    nonconformity_threshold = conformal_predictor.calibrate(0.1)

    test_epsilon = conformal_predictor.test_threshold(nonconformity_threshold)
    
