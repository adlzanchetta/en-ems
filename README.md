# ebemse | *Entropy-Based Ensemble Members SElection*

*ebemse* is a Python library for the selection of a set of mutually exclusive, collectivelly exaustive (MECE) ensemble members.

The library implements the approach presented by [Darbandsari and Coulibaly (2021)](http://doi.org/https://doi.org/10.1016/j.jhydrol.2020.125577) as step that antecedes the further merging of a set of ensemble forecasts.


## Installing

The library can be installed using the traditional pip:

    pip install ebemse

And is listed on the Python Package Index (*pypi*) at []().


## Using

Suppose you have a file named ```example.csv``` with the following content:

```
Date,       Memb_A, Memb_B, ...,  Memb_Z, Obsv
2020/05/15, 1.12,   1.05,   ...,  0.5,    1.01
2020/05/16, 1.15,   1.12,   ...,  0.9,    1.10
2020/05/17, 1.13,   1.32,   ...,  1.1,    1.29
...         ...     ...     ...,  ...,    ...
2020/11/30, 1.22,   0.95,   ...,  0.3,    0.87
```

In which the columns starting with "Memb_" hold the realization of one ensemble member for the time interval and "Obsv" holds the observed values for the same time interval.

If your our objective is to select a MECE set considering obaservations, it can be done using the standard parameters by:

```
import pandas as pd
import ebemse

# read file
data_ensemble = pd.read_csv("example.csv").to_dict('list')
data_obsv = data_ensemble["Obsv"]
del data_ensemble["Obsv"], data_ensemble["Date"]

# perform selection
selected_members = ebemse.select_ensemble_members(data_ensemble, data_obsv)
```

The variable ```selected_members``` will be a dictionary with the following keys and values:

- **history**: dictionary with the following additional information related with the selection process:
	- **total_correlation**: list of floats
	- **joint_entropy**: list of floats
	- **transinformation**: list of floats or ```None```
- **selected_members**: list of string with the labels of the selected elements
- **original_ensemble_joint_entropy**: float


## Further information

### select\_ensemble\_members()

Arguments:
- all_ensemble_members: dict
- observations: Union[list, tuple, np.array, None] (default: *None*)
- n_bins: Union[int, None] (default: *10*)
- bin_by: str (default: *"quantile_individual"*)
- beta_threshold: float (default: *0.9*)
- n_processes: int (default: *1*)
- minimum_n_members: int (default: *2*)
- verbose: bool (default: *False*)