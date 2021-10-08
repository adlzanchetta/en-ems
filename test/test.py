from genericpath import isdir
from sys import path
import pandas as pd
import ebemse

'''

'''

# ## TEST CASE 1.1: NO OBSERVATIONS, quantile individual ############################################################# #

# load test data - no observations available
test_data = pd.read_csv("example_data/link-CJ48.92-Flow_velocity-df.csv")
test_data = test_data.to_dict('list')
if 'Unnamed: 0' in test_data:
    del test_data['Unnamed: 0']

# 
print("Test case 1.1:")
selection_log = ebemse.select_ensemble_members(test_data, None, n_bins=10, bin_by='quantile_individual', 
                                               beta_threshold=0.95, n_processes=1, verbose=False)
print(selection_log)

# ## TEST CASE 2: NO OBSERVATIONS, quantile individual ############################################################### #

print("Test case 1.2:")
selection_log = ebemse.select_ensemble_members(test_data, None, n_bins=10, bin_by='quantile_total', beta_threshold=0.95,
                                               n_processes=1, verbose=False)
print(selection_log)

# TODO
