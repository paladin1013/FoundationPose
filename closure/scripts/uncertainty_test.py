from typing import Dict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from conformal_predictor import Dataset
from gpu_utils import get_rotation_dist
from tqdm import tqdm



def load_dataset(
    object_ids,
    data_dir,
    dataset_name,
    top_hypotheses_num,
):

    raw_dataset: Dict[int, Dict[str, npt.NDArray[np.float32]]] = {}
    for id in object_ids:
        data = {}
        data["gt_poses"] = np.load(f"{data_dir}/{dataset_name}_gt_poses_{id}.npy")
        data["pred_poses"] = np.load(
            f"{data_dir}/{dataset_name}_out_poses_{id}.npy"
        )[:, :top_hypotheses_num]
        data["pred_scores"] = np.load(
            f"{data_dir}/{dataset_name}_out_scores_{id}.npy"
        )[:, :top_hypotheses_num]
        raw_dataset[id] = data



    data_size = sum(
        [raw_dataset[id]["gt_poses"].shape[0] for id in object_ids]
    )
    gt_poses = np.concatenate([raw_dataset[id]["gt_poses"] for id in object_ids])
    pred_poses = np.concatenate(
        [raw_dataset[id]["pred_poses"] for id in object_ids]
    )
    dataset = Dataset(
        data_ids=np.arange(data_size),
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
        size=data_size,
    )

    return dataset


dataset = load_dataset(object_ids=[1], data_dir='data', dataset_name='linemod', top_hypotheses_num=50)


top_score_cnts = np.zeros((dataset.size,))
for i in range(dataset.size):
    top_score_cnts[i] = np.sum(dataset.pred_scores[i] == dataset.pred_scores[i, 0])
print(f"Top score 1: {np.sum(top_score_cnts==1)}")

first_hypothesis_R = dataset.pred_Rs[:, 0]
first_hypothesis_t = dataset.pred_ts[:, 0]

first_hypothesis_error_R = get_rotation_dist(first_hypothesis_R, dataset.gt_Rs)
first_hypothesis_error_t = np.linalg.norm(first_hypothesis_t - dataset.gt_ts, axis=1)

mean_hypothesis_R = np.mean(dataset.pred_Rs, axis=1)
mean_hypothesis_t = np.mean(dataset.pred_ts, axis=1)

mean_hypothesis_error_R = get_rotation_dist(mean_hypothesis_R, dataset.gt_Rs)
mean_hypothesis_error_t = np.linalg.norm(mean_hypothesis_t - dataset.gt_ts, axis=1)




internal_mean_distance_R = np.zeros((dataset.size,))
internal_mean_distance_t = np.zeros((dataset.size,))

for i in tqdm(range(dataset.size)):
    hypotheses_R = dataset.pred_Rs[i][dataset.pred_scores[i] == dataset.pred_scores[i, 0], :, :]
    hypotheses_t = dataset.pred_ts[i][dataset.pred_scores[i] == dataset.pred_scores[i, 0], :]

    internal_mean_distance_R[i] = np.mean(
        get_rotation_dist(
            np.repeat(hypotheses_R[None, :, :, :], hypotheses_R.shape[0], axis=0),
            np.repeat(hypotheses_R[:, None, :, :], hypotheses_R.shape[0], axis=1)
        ))
    internal_mean_distance_t[i] = np.mean(
        np.linalg.norm(
            np.repeat(hypotheses_t[None, :, :], hypotheses_t.shape[0], axis=0) - 
            np.repeat(hypotheses_t[:, None, :], hypotheses_t.shape[0], axis=1),
            axis=2
        ))
    
    
plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(first_hypothesis_error_R, internal_mean_distance_R, label='First Hypothesis')
# plt.scatter(mean_hypothesis_error_R, internal_mean_distance_R, label='Mean Hypothesis')

# plt.xlabel('Rotation Error')
# plt.ylabel('Internal Rotation Distance')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.scatter(first_hypothesis_error_t, internal_mean_distance_t, label='First Hypothesis')
# plt.scatter(mean_hypothesis_error_t, internal_mean_distance_t, label='Mean Hypothesis')

# plt.xlabel('Translation Error')
# plt.ylabel('Internal Translation Distance')
# plt.legend()

plt.scatter(top_score_cnts, first_hypothesis_error_R, label = 'First Hypothesis')
plt.scatter(top_score_cnts, mean_hypothesis_error_R, label = 'Mean Hypothesis')
plt.xlabel('Top Score Count')
plt.ylabel('Rotation Error')
plt.legend()

plt.show()


