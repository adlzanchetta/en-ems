# En-EMS documentation

## Public methods

Public methods are sorted by alphabetical order.

### load\_data\_75()

**About**

Loads a pre-established dataset that can be used for testing the package.

**Arguments**

*None*

**Return**

Pandas DataFrame with 75 columns, each representing an ensemble member series.

### select\_ensemble\_members()

**About**

TODO

**Arguments**

- all\_ensemble\_members: dict
- observations: Union[list, tuple, np.array, None] (default: *None*)
- n\_bins: Union[int, None] (default: *10*)
- bin\_by: str (default: *"quantile\_individual"*)
- beta\_threshold: float (default: *0.9*)
- n\_processes: int (default: *1*)
- minimum\_n\_members: int (default: *2*)
- verbose: bool (default: *False*)

**Return**

Set with the keys of the selected members.