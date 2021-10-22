# En-EMS | *Entropy-based Ensemble Members Selection*

*en-ems* is a Python library for the selection of a set of mutually exclusive, collectivelly exaustive (MECE) ensemble members.

The library implements the approach presented by [Darbandsari and Coulibaly (2020)](http://doi.org/https://doi.org/10.1016/j.jhydrol.2020.125577) as step that antecedes the further merging of a set of ensemble forecasts.

The *en-ems* package is built over the [pyitlib](https://pypi.org/project/pyitlib/) package, which implements fundamental information theory methods.


## Installing

The library can be installed using the traditional pip:

    pip install en-ems

And is listed on the Python Package Index (*pypi*) as [en-ems](https://pypi.org/project/en-ems/).


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
import enems

# read file
data_ensemble = pd.read_csv("example.csv").to_dict('list')
data_obsv = data_ensemble["Obsv"]
del data_ensemble["Obsv"], data_ensemble["Date"]

# perform selection
selected_members = enems.select_ensemble_members(data_ensemble, data_obsv)
```

The variable ```selected_members``` will be a dictionary with the following keys and values:

- **history**: dictionary with the following additional information related with the selection process:
	- **total_correlation**: list of floats
	- **joint_entropy**: list of floats
	- **transinformation**: list of floats or ```None```
- **selected_members**: list of string with the labels of the selected elements
- **original_ensemble_joint_entropy**: float


## Further documentation

Further information can be found in the *docs* folder of this project.