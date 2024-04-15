# import yaml
import numpy as np
import scipy


def convert_npy_to_mat(npy_path, mat_path):
    npy_data = np.load(npy_path, allow_pickle=True)
    keys = list(npy_data[0].keys())
    reorganized_data = {}
    for key in keys:
        if key in ["R_set", "t_set"]:
            reorganized_data[key] = [d[key] for d in npy_data]

        else:
            reorganized_data[key] = [d[key] for d in npy_data]
            if isinstance(reorganized_data[key][0], np.ndarray):
                reorganized_data[key] = np.stack(reorganized_data[key])
            if isinstance(reorganized_data[key], list):
                reorganized_data[key] = np.array(reorganized_data[key])


    scipy.io.savemat(mat_path, reorganized_data)


if __name__ == "__main__":
    data_dir = "data/closure_data"
    # convert_npy_to_mat(f"{data_dir}/predict_results.npy", f"{data_dir}/predict_results.mat")
    for object_id in [1, 2, 4, 5, 6, 8, 9]:
        convert_npy_to_mat(f"{data_dir}/20240412_230615_object_{object_id}.npy", f"{data_dir}/20240412_230615_object_{object_id}.mat")