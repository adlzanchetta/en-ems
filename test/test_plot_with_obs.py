import matplotlib.pyplot as plt
import numpy as np
import enems


if __name__ == "__main__":

    # ## LOAD DATA ################################################################################################### #

    test_data_obs = enems.load_data_obs().values
    test_data_df = enems.load_data_75()
    test_data = test_data_df.to_dict("list")

    # ## PLOT FUNCTIONS ############################################################################################## #

    def plot_ensemble_members(ensemble_series: dict, observation_series: np.array, selected_series: set, 
                              plot_title: str, output_file_path: str) -> None:
        _, axs = plt.subplots(1, 1, figsize=(7, 2.5))
        axs.set_xlabel("Time")
        axs.set_ylabel("Value")
        axs.set_title(plot_title)
        axs.set_xlim(0, 143)
        axs.set_ylim(0, 5)
        [axs.plot(ensemble_series[series_id], color="#999999", zorder=3, alpha=0.33) for series_id in selected_series]
        axs.plot(observation_series, color="#000000", zorder=4)
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close()
        return None

    def plot_log(n_total_members: int, log: dict, output_file_path: str) -> None:
        _, axss = plt.subplots(1, 3, figsize=(10.0, 2.5))
        x_values=[n_total_members-i-1 for i in range(len(log["history"]["total_correlation"]))]
        axss[0].set_xlabel("")
        axss[0].set_ylabel("Total correlation")
        axss[0].plot(x_values, log["history"]["total_correlation"], color="#7777FF", zorder=3)
        axss[0].set_ylim(70, 140)
        axss[0].set_xlim(x_values[0], x_values[-1])
        axss[1].set_xlabel("# selected members")
        axss[1].set_ylabel("Joint entropy")
        axss[1].axhline(log["original_ensemble_joint_entropy"], color="#FF7777", zorder=3, label="Full set")
        axss[1].plot(x_values, log["history"]["joint_entropy"], color="#7777FF", zorder=3, label="Selected set")
        axss[1].set_ylim(6.3, 6.9)
        axss[1].set_xlim(x_values[0], x_values[-1])
        axss[1].legend()
        axss[2].set_xlabel("")
        axss[2].set_ylabel("Transinformation")
        axss[2].plot(x_values, log["history"]["transinformation"], color="#7777FF", zorder=3, label="Selected set")
        axss[2].axhline(log["original_ensemble_transinformation"], color="#FF7777", zorder=3, label="Full set")
        axss[2].set_xlim(x_values[0], x_values[-1])
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close()
        return None

    # ## FUNCTIONS CALL ############################################################################################## #

    cur_selection_log = enems.select_ensemble_members(test_data, test_data_obs, n_bins=10, bin_by="equal_intervals", 
                                                        beta_threshold=0.95, n_processes=1, verbose=False)

    plot_log(len(test_data.keys()), cur_selection_log, "test/log_obs.svg")
    plot_ensemble_members(test_data, test_data_obs, set(test_data.keys()),
                            "All members (%d)" % len(test_data.keys()),
                            "test/ensemble_all_obs.svg")
    plot_ensemble_members(test_data, test_data_obs, cur_selection_log["selected_members"], 
                            "Selected members (%d)" % len(cur_selection_log["selected_members"]),
                            "test/ensemble_selected_obs.svg")
    
    del test_data_obs, cur_selection_log
