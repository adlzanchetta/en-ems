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

Main function of the package. Performs the selection.

**Arguments**

- *all\_ensemble\_members: dict*

  Dictionaty with format:

      {
        "ensemble_member_01_id": np.array,
        "ensemble_member_02_id": np.array,
        ...
        "ensemble_member_NN_id": np.array
      }

- *observations: Union[list, tuple, np.array, None]* (default: *None*)

  If not provided (*None* value), applies the relaxed version of the algorithm.

- *n\_bins: Union[int, None]* (default: *10*)

- *bin\_by: str* (default: *"quantile\_individual"*)

  Defines which method will be used to discretize the data.

  - *"quantile\_individual"*:

    Bins will have equal (or nearly equal) number of elements (quantiles).
    The values of the quantiles are defined independently for each ensemble member data.

  - *"quantile\_total"*:

    Bins will have equal (or nearly equal) number of elements (quantiles).
    The values of the quantiles are defined considering all ensemble members data.

  - *"equal\_intervals"*:

    Bins will have equal interval values, regardless the number of records in each bin.
    The values of the quantiles are defined considering all ensemble members data.

- *beta\_threshold: float* (default: *0.9*)

- *n\_processes: int* (default: *1*)

- *minimum\_n\_members: int* (default: *2*)

- *verbose: bool* (default: *False*)

**Return**

A dictionary with the following structure concerning the progress of the members selection:

{

- *history*: {
  - *total\_correlation*: list of floats;
  - *joint_entropy*: list of floats;
  - *transinformation*: list of floats (if the *observations* argument is provided and not *None*) or *None* (otherwise)
- *selected\_members*: list of string with the labels of the selected elements;
- *original\_ensemble\_joint_entropy*: float

}
