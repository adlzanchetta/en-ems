from numpy.core.fromnumeric import argmin
from pyitlib import discrete_random_variable
from multiprocessing import Pool
from typing import Tuple, Union
import pkg_resources
import pandas as pd
import numpy as np
import copy

# ## PRIVATE METHODS ################################################################################################# #

def _categorize_by_intervals(values: list, intervals: list, labels: list) -> list:
    """
    Categorizes a list of continuous values.
    :param values: List of L continuos values to be classified.
    :param intervals: List of I intervals, each interval represented by a two-values (min, max) tuple.
    :param labels: List of size I with the labels associated to each interval.
    :return: List of size L only with the values contained in 'labels' parameter.
    """
    ret_list = []
    for cur_value in values:
        for cur_interval, cur_label in zip(intervals, labels):
            if _interval_contains(cur_interval, cur_value):
                ret_list.append(cur_label)
                break
        else:
            if cur_value <= intervals[0].left:
                ret_list.append(labels[0])
            elif cur_value >= intervals[-1].right:
                ret_list.append(labels[-1])
            else:
                raise ValueError("Value {0} not within any interval.".format(cur_value))
    return ret_list


def _interval_contains(interval: pd.Interval, value: float) -> bool:
    """
    Just creates a dictionary with one entry per column without index
    """
    if (value > interval.left) and (value < interval.right):
        return True
    elif ((value == interval.left) and interval.closed_left) or ((value == interval.right) and interval.closed_right):
        return True
    else:
        return False


def _discretize_ensemble_only(data: Union[dict, None], n_bins: Union[int, None], bin_by: str) -> dict:
    """
    Performs discretization of the continuous data considering only the ensemble members.
    """

    # basic checks
    if data is None:
        return None
    if n_bins is None:
        return copy.deepcopy(data)
    
    # applies bin approach
    labels = list(range(n_bins))

    if bin_by == 'quantile_individual':
        return dict([(k, list(pd.qcut(v, n_bins, labels=labels))) for k, v in data.items()])

    elif bin_by == 'quantile_total':
        # define intervals
        all_values = np.array([v for v in data.values()]).flatten()
        intervals = sorted(list(set(pd.qcut(all_values, n_bins))))
        del all_values

        # categorize
        return dict([(k, _categorize_by_intervals(v, intervals, labels)) for k, v in data.items()])
    
    elif bin_by in {'equal_intervals'}:
        # define intervals
        all_values = np.array([v for v in data.values()]).flatten()
        intervals = sorted(list(set(pd.cut(all_values, bins=n_bins))))
        print("Binning intervals:", intervals)
        del all_values

        # categorize
        return dict([(k, _categorize_by_intervals(v, intervals, labels)) for k, v in data.items()])

    else:
        raise ValueError("Binning by '%s' not supported." % bin_by)


def _discretize_ensemble_and_obsevations(ensemble_data: Union[dict, None], observed_data: Union[list, tuple, np.array, 
            None], n_bins: Union[int, None], bin_by: str) -> Tuple[dict, list]:
    """
    Performs discretization of the continuous data considering both ensemble members and observation.
    :return: dict with {ensemble_id: list(ensemble data categorized)} and list of observed data categorized
    """

    # basic check
    if n_bins is None:
        return copy.deepcopy(ensemble_data), observed_data
    
    # applies bin approach
    labels = list(range(n_bins))
    if bin_by == 'quantile_individual':
        ret_dict = dict([(k, list(pd.qcut(v, n_bins, labels=labels))) for k, v in ensemble_data.items()])
        ret_obsv = list(pd.qcut(observed_data, n_bins, labels=labels))
        return ret_dict, ret_obsv
    
    elif bin_by == 'quantile_total':
        # define intervals
        all_values = np.array([v for v in ensemble_data.values()] + [observed_data, ]).flatten()
        intervals = sorted(list(set(pd.qcut(all_values, n_bins))))
        del all_values

        # categorize
        return dict([(k, _categorize_by_intervals(v, intervals, labels)) for k, v in ensemble_data.items()]), \
                    _categorize_by_intervals(observed_data, intervals, labels)
    
    elif bin_by == 'equal_intervals':
        # define intervals
        all_values = np.array([v for v in ensemble_data.values()] + [observed_data, ]).flatten()
        intervals = sorted(list(set(pd.cut(all_values, bins=n_bins))))
        del all_values

        # categorize
        return dict([(k, _categorize_by_intervals(v, intervals, labels)) for k, v in ensemble_data.items()]), \
                    _categorize_by_intervals(observed_data, intervals, labels)

    else:
        raise ValueError("Binning by '%s' not supported." % bin_by)


def _rmsd(vals1: np.array, vals2: np.array) -> float:
    """
    Calculates the root mean squared distance between vals1 and vals2.
    """
    return np.sqrt(sum((vals1-vals2)**2) / vals1.size)


def _mad(vals1: np.array, vals2: np.array) -> float:
    """
    Calculates tge mean absolute distance between vals1 and vals2.
    """
    return sum(abs(vals1-vals2)) / vals1.size


def _get_minimal_distance(arg_tuple: tuple) -> float:
    """
    Calculates the minimum distance between specified ensemble member and all other.
    :param arg_tuple: A tuple with (dict of ensemble members, id of the specified ensemble member, distance method).
    :return: Minimum distance found.
    """

    # splits the arguments
    member_out, ensemble_members, deviation = arg_tuple

    # splits the ensemble members
    all_ensemble_values = [v for k, v in ensemble_members.items() if k != member_out]
    member_out_values = ensemble_members[member_out]

    # get the distance function
    dev_func = {
        "mad": _mad,
        "rmsd": _rmsd
    }[deviation]
    
    # get the minimum distance value
    return min([dev_func(m, member_out_values) for m in all_ensemble_values])


def _get_total_correlation(arg_list: tuple) -> float:
    """
    Calculates the total correlation between all ensemble members (excluding one).
    :param arg_list: Tuple with: (id of the excluded ensemble member, dict with all ensemble members)
    :return: Total correlation value.
    """

    cur_member_out, ensemble_members = arg_list

    # calculate total correlation
    all_ensemble_values = [v for k, v in ensemble_members.items() if k != cur_member_out]
    cur_total_correlation = discrete_random_variable.information_multi(all_ensemble_values)
    del all_ensemble_values

    return cur_total_correlation


def _stop_criteria(ensemble_members: dict, observations: Union[list, tuple, np.array, None],
                   full_ensemble_joint_entropy: float, full_ensemble_transinformation: Union[float, None],
                   beta_threshold: float, max_n_members: Union[int, float], verbose: bool = False) -> tuple:
    """
    Evaluate if: joint entropy > beta_threshold, transinformation > beta_threshold, size(selected) > max_n_members
    :return: Tuple with: (joint entropy: float, transinformation: float or None, allow_to_stop: boolean)
    """

    ensemble_members_values = list(ensemble_members.values())
    
    # calculate joint entropy
    joint_entropy = discrete_random_variable.entropy_joint(ensemble_members_values)
    joint_entropy_ratio = joint_entropy/full_ensemble_joint_entropy
    stop_due_joint_entropy = True if joint_entropy_ratio <= beta_threshold else False
    print("  With %2d members. Current Joint Entropy: %.05f. Full Joint Entropy: %.05f. Ratio: %.05f." % 
        (len(ensemble_members), joint_entropy, full_ensemble_joint_entropy, joint_entropy_ratio)) if verbose else None

    # calculate transinformation if needed
    if (observations is None) or (full_ensemble_transinformation is None):
        transinformation, transinfo_ratio = None, 0
    else:
        transinformation = discrete_random_variable.information_mutual(ensemble_members_values, observations,
                                                                       cartesian_product=True)
        transinformation = np.mean(transinformation)
        transinfo_ratio = transinformation / full_ensemble_transinformation

    # stop_due_transinformation = True if ((transinformation is None) or (transinfo_ratio <= beta_threshold)) else False  # >= case (as in work)
    stop_due_transinformation = False if ((transinformation is None) or (transinfo_ratio > beta_threshold)) else True     #  < case (as interpreted)

    can_stop_due_max_number = True if len(ensemble_members_values) <= max_n_members else False

    # define final stop decision
    # stop = True if (stop_due_joint_entropy and stop_due_transinformation) else False                                    # AND case (as in work)
    stop = True if (stop_due_joint_entropy or stop_due_transinformation) else False                                       #  OR case (as interpreted)

    stop = stop if can_stop_due_max_number else False

    return joint_entropy, transinformation, stop


def _select_winner_ensemble_set(ensemble_members: dict, n_processes: int) -> Tuple[dict, float]:
    """
    Select the one-element-removed subset with minimum total correlation
    """

    if n_processes == 1:
        
        min_total_correlation, total_correlations, all_member_ids = np.inf, [], sorted(list(ensemble_members.keys()))
        for cur_member_out in all_member_ids:
            total_correlations.append(_get_total_correlation((cur_member_out, ensemble_members)))

    else:

        # adjust parallel args and call it using the pool of processes
        parallel_args = [(cur_member_id, ensemble_members) for cur_member_id in ensemble_members.keys()]
        with Pool(n_processes) as processes_pool:
            total_correlations = processes_pool.map(_get_total_correlation, parallel_args)
        all_member_ids = [v[0] for v in parallel_args]
        del parallel_args
        
    # identify the winner
    min_total_correlation, min_correlation_idx = min(total_correlations), argmin(total_correlations)
    winner_ensemble_subset = copy.deepcopy(ensemble_members)

    del winner_ensemble_subset[all_member_ids[min_correlation_idx]]

    return winner_ensemble_subset, min_total_correlation


def _select_winner_ensemble_set_by_distance(deviation: str, ensemble_members: dict, n_processes: int) -> \
        Tuple[dict, float, float]:
    """
    
    """

    # calculate minimal deviations
    parallel_args = [(cur_member_id, ensemble_members, deviation) for cur_member_id in ensemble_members.keys()]
    if n_processes == 1:
        minimal_deviations = [_get_minimal_distance(p_arg) for p_arg in parallel_args]
    else:
        with Pool(n_processes) as processes_pool:
            minimal_deviations = processes_pool.map(_get_minimal_distance, parallel_args)
    all_member_ids = [v[0] for v in parallel_args]
    del parallel_args
    
    # identify the winner
    min_dev, min_i = np.min(minimal_deviations), np.argmin(minimal_deviations)
    winner_ensemble_subset = copy.deepcopy(ensemble_members)
    del winner_ensemble_subset[all_member_ids[min_i]]

    return winner_ensemble_subset, min_dev, -1


# ## PUBLIC METHODS ################################################################################################## #

def load_data_75() -> pd.DataFrame:
    """
    Gets a Pandas DataFrame with series of 75 ensemble members.
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'example_data/ensemble_set_75.pickle')
    return pd.read_pickle(stream)


def load_data_obs(suffix: str = 'a') -> pd.Series:
    """
    Gets a Pandas Series with the same number of records than the embedded 75 ensemble members.
    :param suffix: A letter corresponding to which observation use. 'a' is default and recommended.
    """
    
    stream = pkg_resources.resource_stream(__name__, 'example_data/ensemble_set_75_obs_%s.pickle' % suffix)
    return pd.read_pickle(stream)
    

def select_ensemble_members(all_ensemble_members: dict, observations: Union[list, tuple, np.array, None] = None,
                            n_bins: int = 10, bin_by: str = 'quantile_individual',
                            beta_threshold: float = 0.9, n_processes: int = 1, minimum_n_members: int = 2, 
                            maximum_n_members: Union[int, None] = None, verbose: bool = False) -> dict:
    """
    Performs the ensemble selection considering (or not) observations.
    :param all_ensemble_members: Dictionary in the form of {'ensemble_member_id': pandas.Series of size N}.
    :param observations: Pandas.Series of size N with observation data.
    :param n_bins: Number of bins into which the continuos data will be discretized.
    :param bin_by: How performing the binning: 'quantile_individual', 'quantile_total' or 'equal_intervals'.
    :param beta_threshold: Expects a value between higher than 0, lower than 1.
    :param n_processes: Number of parallel processes to be used. If value is 1, no paralelization is performed.
    :param minimum_n_members: Minimum number of ensemble members in the selected set. No minimum if not provided.
    :param maximum_n_members: Maximum number of ensemble members in the selected set. No maximum if not provided.
    :param verbose: If True, prints significant innter information. Does not print anything otherwise.
    :return: Dictionary with the id of the 'selected members', 'history' of the selection variables, etc.
    """

    # basic checks
    if (type(n_bins) is not int) or (n_bins <= 0):
        raise ValueError("Argument 'n_bins' must be a positive integer. Got: {0} ({1}).".format(n_bins, type(n_bins)))
    elif (beta_threshold < 0) or (beta_threshold > 1):
        raise ValueError("Argument 'beta_threshold' must be a float between 0 and 1. Got: {0}.".format(beta_threshold))
    elif (type(n_processes) is not int) or (n_processes < 1):
        raise ValueError("Argument 'n_processes' must be a positive integer. Got: {0} ({1}).".format(n_processes,
            type(n_processes)))
    elif (type(minimum_n_members) is not int) or (minimum_n_members < 2):
        raise ValueError("Argument 'minimum_n_members' must be a integer equal or bigger than 2. Got: {0} ({1}).".format(
            minimum_n_members, type(minimum_n_members)))
    elif (maximum_n_members is not None) and (type(maximum_n_members) is not int):
        raise ValueError("Argument 'maximum_n_members' must be a non-zero positive integer or None. Got: {0}.".format(
            type(maximum_n_members)))
    elif (type(maximum_n_members) is int) and (maximum_n_members < 1):
        raise ValueError("Argument 'maximum_n_members' must be a non-zero positive integer or None. Got: {0}.".format(
            type(maximum_n_members)))

    # discretize data
    if observations is None:
        disc_ensemble_members_remaining = _discretize_ensemble_only(all_ensemble_members, n_bins, bin_by)
        disc_observations, full_ensemble_transinformation = None, None
    else:
        disc_ensemble_members_remaining, disc_observations = _discretize_ensemble_and_obsevations(all_ensemble_members,
            observations, n_bins, bin_by)
        ensemble_members_values = list(disc_ensemble_members_remaining.values())
        full_ensemble_transinformation = discrete_random_variable.information_mutual(ensemble_members_values, 
            disc_observations, cartesian_product=True)
        full_ensemble_transinformation = np.mean(full_ensemble_transinformation)
        del ensemble_members_values

    # calculate joint entropy of the original ensemble set
    full_ensemble_joint_entropy = discrete_random_variable.entropy_joint(list(disc_ensemble_members_remaining.values()))

    # create empty accumulator variables
    total_correlations, joint_entropies, transinformations = [], [], []

    # defines the effective maximum number of elements to consider
    max_n_members = np.inf if maximum_n_members is None else maximum_n_members

    # enters iterative loop
    print("Starting with %d ensemble members." % len(disc_ensemble_members_remaining)) if verbose else None
    print("Using %d processes." % n_processes) if verbose else None
    while len(disc_ensemble_members_remaining) > minimum_n_members:

        # identifies winner
        print(" Identifying winner set.") if verbose else None
        disc_ensemble_members_remaining, total_corr = _select_winner_ensemble_set(disc_ensemble_members_remaining,
                                                                                  n_processes)
        total_correlations.append(total_corr)

        # check stopping criteria
        joint_entropy, transinformation, stop = _stop_criteria(disc_ensemble_members_remaining, disc_observations,
            full_ensemble_joint_entropy, full_ensemble_transinformation, beta_threshold, max_n_members,
            verbose=verbose)
        joint_entropies.append(joint_entropy)
        transinformations.append(transinformation) if transinformation is not None else None
        if stop:
            break

        # debug if verbose
        print(" Selected %d ensemble members." % len(disc_ensemble_members_remaining)) if verbose else None
        total_corr = "%.02f" % total_corr
        joint_entropy = None if joint_entropy is None else "%.02f" % joint_entropy
        transinformation = None if transinformation is None else "%.02f" % transinformation
        print("  Total correlation: {0}. Joint Entropy: {1}. Transinformation: {2}.".format(total_corr, joint_entropy,
            transinformation)) if verbose else None
        del total_corr, joint_entropy, transinformation

    # return data
    return {
        "history": {
            "total_correlation": total_correlations,
            "joint_entropy": joint_entropies,
            "transinformation": transinformations if len(transinformations) > 0 else None
        },
        "selected_members": set(disc_ensemble_members_remaining.keys()),
        "original_ensemble_joint_entropy": full_ensemble_joint_entropy,
        "original_ensemble_transinformation": full_ensemble_transinformation
    }


def select_mece_by_distance(all_ensemble_members: dict, n_selected_members: int, distance: str = 'rmsd',
                            n_processes: int = 1) -> dict:
    """
    Performs the ensemble selection considering the maximization of a distance criteria between the ensemble members.
    :param all_ensemble_members: Dictionary in the form of {'ensemble_member_id': pandas.Series of size N}.
    :param distance: Which distance to be used. 'rmsd' (root mean squared) or 'mad' (mean absolute)
    :param n_processes: Number of parallel processes to be used. If value is 1, no paralelization is performed.
    :param n_selected_members: Total number of ensemble members to be selected.
    :return: Dictionary with the id of the 'selected members', 'history' of the selection variables, etc.
    """

    # basic checks
    if n_selected_members < 2:
        raise ValueError("Number of selected members must be 2 or more.")
    if n_selected_members >= len(all_ensemble_members.keys()):
        raise ValueError("Number of selected members must be lower then the total number of ensemble members.")
    if n_processes < 1:
        raise ValueError("Number of parallel processes to be used.")

    # calculate the mean mutual RMSD of the entire set
    full_ensemble_mean_dev, full_ensemble_min_dev = None, None

    # cont_ensemble_members_remaining = copy.deepcopy(all_ensemble_members)
    cont_ensemble_members_remaining = dict([(k, s.values) for k, s in all_ensemble_members.items()])
    mean_devs, min_devs = [], []

    while (len(cont_ensemble_members_remaining) > n_selected_members):

        # identifies winner
        cont_ensemble_members_remaining, min_rmsd, mean_rmsd = _select_winner_ensemble_set_by_distance(distance,
            cont_ensemble_members_remaining, n_processes)
        min_devs.append(min_rmsd)
        mean_devs.append(mean_rmsd)

    # return data
    return {
        "history": {
            "mean_distance": mean_devs,
            "min_distance": min_devs
        },
        "selected_members": set(cont_ensemble_members_remaining.keys()),
        "original_ensemble_mean_deviation": full_ensemble_mean_dev,
        "original_ensemble_min_deviation": full_ensemble_min_dev
    }
