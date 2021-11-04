import enems
import time


if __name__ == '__main__':

    # ## TEST CASE 1: NO OBSERVATIONS ################################################################################ #

    test_data_df = enems.load_data_75()
    test_data = test_data_df.to_dict('list')

    # ## TEST CASE 1.1: NO OBSERVATIONS, quantile individual ######################################################### #

    print("Test case 1.1: no obsv., quantile individual, sequential")
    start = time.time()
    selection_log = enems.select_ensemble_members(test_data, None, n_bins=10, bin_by='quantile_individual', 
                                                  beta_threshold=0.95, n_processes=1, verbose=False)
    print(selection_log)
    print(" In: %.02d seconds." % (time.time() - start))

    print("Test case 1.2: no obsv., quantile individual, parallel 10 processes")
    start = time.time()
    selection_log = enems.select_ensemble_members(test_data, None, n_bins=10, bin_by='quantile_individual', 
                                                  beta_threshold=0.95, n_processes=10, verbose=False)
    print(selection_log)
    print(" In: %.02d seconds." % (time.time() - start))

    # ## TEST CASE 2: NO OBSERVATIONS, quantile individual ########################################################### #

    print("Test case 2.1:")
    selection_log = enems.select_ensemble_members(test_data, None, n_bins=10, bin_by='quantile_total', 
                                                  beta_threshold=0.95, n_processes=1, verbose=False)
    print(selection_log)
