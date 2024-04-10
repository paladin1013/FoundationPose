
import numpy as np

# from gpu_utils import get_rotation_dist

if __name__ == "__main__":
    miniball_data = np.load("./data/closure_test/rotation_miniball.npy", allow_pickle=True).item()
    test_data = np.load("./data/closure_test/test_result_0.npy", allow_pickle=True).item()

    R_set = miniball_data["R_set"] # (N, 3, 3)

    err = np.zeros((R_set.shape[0], 1))
    for i in range(R_set.shape[0]):
        err[i] = np.linalg.norm(R_set[i].T@R_set[i] - np.eye(3))

    print(R_set[1,:,:])
    print(err)
    print(f"R_set rotation err: {max(err)}")
    # # R_set = R_set[10000:,:,:]
    R_center = miniball_data["R_center"] # (3, 3)
    print(R_center.T@R_center)
    # gt_Rs = test_data["gt_Rs"] # (3, 3)
    # print(gt_Rs.transpose(1, 0)@gt_Rs)
    # print(f"{R_set.shape=}, {gt_Rs.shape=}")
    
    # rot_dist = get_rotation_dist(R_set, np.repeat(gt_Rs[None, :, :], R_set.shape[0], axis=0))
    # print(f"R_set: {max(rot_dist)=}")
    # center_dist = get_rotation_dist(R_center, gt_Rs)
    # print(f"R_center: {center_dist=}")

    # center_rot_dist = get_rotation_dist(R_set, np.repeat(R_center[None, :, :], R_set.shape[0], axis=0))
    # print(f"R_set - center: {max(center_rot_dist)=}")

    # print(f'R_radius: {miniball_data["R_radius"]}')

    # # print(f'quant_radius: {miniball_data["quant_radius"]}')

    # print(f'angs: {min(abs(miniball_data["angs"]))}')

    # # N = R_set.shape[0]

    # # all_dist = get_rotation_dist(
    # #     np.repeat(R_set[:, None, :, :], R_set.shape[0], axis=1),
    # #     np.repeat(R_set[None, :, :, :], R_set.shape[0], axis=0),
    # # )  # (M, M)
    
    # # print(np.max(all_dist))
    # # print(f"all_dist: {min(all_dist)=}, {max(all_dist)=}")
