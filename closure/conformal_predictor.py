from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import cupy as cp

@dataclass
class Dataset:
    data_ids: npt.NDArray[np.int_] # (N, )
    gt_poses: npt.NDArray[np.float32] # (N, 4, 4)
    pred_poses: npt.NDArray[np.float32] # (N, M, 4, 4)
    pred_scores: npt.NDArray[np.float32] # (N, M)
    object_ids: npt.NDArray[np.int_] # (N, )
    size: int

class NonconformityFunc:
    @staticmethod
    def max_R(
        center_poses: cp.ndarray,
        pred_poses: cp.ndarray,
        pred_scores: cp.ndarray,
    ) -> cp.ndarray:
        
        assert center_poses.shape[0] == 4 & center_poses.shape[1] == 4
        assert pred_poses.shape[1] == 4 & pred_poses.shape[2] == 4
        assert pred_poses.shape[0] == pred_scores.shape[0]
        
        gt_R = center_poses[:3, :3]
        gt_t = center_poses[:3, 3]
        
        pred_prob = pred_scores/np.sum(pred_scores)
        pred_R = pred_poses[:,:3,:3]  
        pred_t = pred_poses[:,:3,3]
        
        R_diff = np.linalg.norm(gt_R - pred_R, axis=(1,2))
        t_diff = np.linalg.norm(gt_t - pred_t, axis=1)
        
        nonconformity = max(R_diff*pred_prob)
        
        return nonconformity

    @staticmethod
    def mean_R(
        center_poses: cp.ndarray,
        pred_poses: cp.ndarray,
        pred_scores: cp.ndarray,
    ) -> cp.ndarray:...

    @staticmethod
    def normalized_max_R(
        center_poses: cp.ndarray,
        pred_poses: cp.ndarray,
        pred_scores: cp.ndarray,
    ) -> cp.ndarray:...

    @staticmethod
    def normalized_mean_R(
        center_poses: cp.ndarray,
        pred_poses: cp.ndarray,
        pred_scores: cp.ndarray,
    ) -> cp.ndarray:...


class ConfromalPredictor:
    def __init__(self, nonconformity_func_name: str, top_hypotheses_num: int = 10):
        self.dataset: Dataset
        self.calibration_set: Dataset
        self.test_set: Dataset
        assert nonconformity_func_name in NonconformityFunc.__dict__.keys()
        self.nonconformity_func = getattr(NonconformityFunc, nonconformity_func_name)
        self.top_hypotheses_num = top_hypotheses_num


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
            )[:, :self.top_hypotheses_num]
            data["pred_scores"] = np.load(
                f"{data_dir}/{dataset_name}_out_scores_{id}.npy"
            )[:, :self.top_hypotheses_num]
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
            size = self.data_size,
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
            size = calibration_set_size,
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
            size = self.data_size - calibration_set_size,
        )

    def nonconformity_func(
        self,
        center_poses: cp.ndarray, # (K, 4, 4)
        pred_poses: cp.ndarray, # (M, 4, 4)
        pred_scores: cp.ndarray, # (M, )
    ) -> cp.ndarray: ...
    """
    output: (K, )
    """

    def calibrate(self, epsilon: float):
        nonconformity_scores = np.zeros(self.calibration_set.size)
        for k in range(self.calibration_set.size):
            nonconformity_scores[k] = cp.asnumpy(self.nonconformity_func(
                cp.array(self.calibration_set.gt_poses[k][None, :, :]),
                cp.array(self.calibration_set.pred_poses[k]),
                cp.array(self.calibration_set.pred_scores[k]),
            ))[0]

        nonconformity_threshold = np.quantile(nonconformity_scores, 1 - epsilon)

        return nonconformity_threshold

    def predict(
        self,
        pred_poses: npt.NDArray[np.float32], # (N, M, 4, 4)
        pred_scores: npt.NDArray[np.float32], # (N, M)
        nonconformity_threshold: float,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    """
    Return: 
        - minimax_center_poses: (N, 4, 4)
        - max_rotation_errors: (N, )
        - max_translation_errors: (N, )
    """
