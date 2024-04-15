import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_fig(
    error_bound_R: np.ndarray, 
    error_bound_t: np.ndarray, 
    minimax_center_err_R: np.ndarray, 
    minimax_center_err_t: np.ndarray,
    save_path: str,
    ):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Scatter plot: x-axis is the error bound, y-axis is the minimax error
    ax[0].scatter(error_bound_R, minimax_center_err_R, label="Rotation")
    ax[0].set_xlabel("Rotation error bound")
    ax[0].set_ylabel("Rotation minimax center error")

    ax[1].scatter(error_bound_t, minimax_center_err_t, label="Translation")
    ax[1].set_xlabel("Translation error bound")
    ax[1].set_ylabel("Translation minimax center error")

    # Save figure to save_path
    plt.savefig(save_path)

def load_data(data_dir: str, timestamp: str, object_id: List[int]):
    error_bound_R = []
    error_bound_t = []
    minimax_center_err_R = []
    minimax_center_err_t = []
    for i in object_id:
        # Load data from file
        data = np.load(data_dir + f"/{timestamp}_object_{i}.npy", allow_pickle=True)
        for data_item in data:
            error_bound_R.append(data_item["error_bound_R"])
            error_bound_t.append(data_item["error_bound_t"])
            minimax_center_err_R.append(data_item["minimax_center_err_R"])
            minimax_center_err_t.append(data_item["minimax_center_err_t"])
    return error_bound_R, error_bound_t, minimax_center_err_R, minimax_center_err_t
    
if __name__ == "__main__":
    # Load data
    data_dir = "data/closure_data"
    timestamp = "20240414_211057"
    object_id = [1]
    error_bound_R, error_bound_t, minimax_center_err_R, minimax_center_err_t = load_data(data_dir, timestamp, object_id)
    # Plot figure
    save_path = f"fig/{timestamp}.pdf"
    plot_fig(error_bound_R, error_bound_t, minimax_center_err_R, minimax_center_err_t, save_path)
