# import yaml
import numpy as np

print("Start reading\n")
# Open the YAML file for reading
# with open('../datas/linemod_res_scores.yml', 'r') as f:
#     # Load the YAML data
#     all_scores = yaml.safe_load(f)
# print("Score loaded")

# with open('../datas/linemod_res_poses.yml', 'r') as f:
#     # Load the YAML data
#     all_poses = yaml.safe_load(f)
# print("poses loaded")

# with open('../datas/linemod_res_gt_poses.yml', 'r') as f:
#     # Load the YAML data
#     all_gt_poses = yaml.safe_load(f)
# print("gt poses loaded")

# Now you can work with the loaded data
# For example, print the loaded data

# print(type(data))
# print(len(data))
# print(data.keys())
# print(data[1]['000002'][1])
# print(np.array(data[1]['000002'][1]).shape)

# for video_id in all_scores:
#     for id_str in all_scores[video_id]:
#         for ob_id in all_scores[video_id][id_str]:
#             print(all_scores[video_id][id_str][ob_id])
#             poses = all_poses[video_id][id_str][ob_id]
#             gt_pose = all_gt_poses[video_id][id_str][ob_id]
#             R = poses[:,:3,:3]
#             t = poses[:,:3,3]
#             R_err = np.linalg.norm(R - gt_pose[:3,:3],axis = (1,2))
#             t_err = np.linalg.norm(t - gt_pose[:3,3],axis = 1)
#             print(R_err)
#             print(t_err)
#             break
#         break
#     break

# data_dir = "/home/purse/FoundationPose/datas"

data_dir = "data"

scores = np.load(f"{data_dir}/linemod_out_scores_1.npy")
poses = np.load(f"{data_dir}/linemod_out_poses_1.npy")
gt_poses = np.load(f"{data_dir}/linemod_gt_poses_1.npy")

a = 1

# print(f"Done")
print(f"{scores.shape=}")
print(f"{poses.shape=}")
print(f"{gt_poses.shape=}")
