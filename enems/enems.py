from numpy.core.fromnumeric import argmin
from pyitlib import discrete_random_variable
from multiprocessing import Pool
from typing import Tuple, Union
import pkg_resources
import pandas as pd
import numpy as np
import string
import copy

# ## PRIVATE METHODS ################################################################################################# #

def _categorize_by_intervals(values: list, intervals: list, labels: list) -> list:
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
    elif (value == interval.left) and interval.closed_left:
        return True
    elif (value == interval.right) and interval.closed_right:
        return True
    else:
        return False


def _discretize_if_needed(data: Union[dict, None], n_bins: Union[int, None], bin_by: str) -> dict:
    """
    
    """

    # basic check
    if data is None:
        return None

    # basic check
    if n_bins is None:
        return copy.deepcopy(data)
    
    # applies bin approach
    # labels = [int(l) for l in list(string.ascii_uppercase[0:n_bins])]
    labels = list(range(n_bins))

    if bin_by == 'quantile_individual':
        ret_dict = dict([(k, list(pd.qcut(v, n_bins, labels=labels))) for k, v in data.items()])
        return ret_dict

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
        del all_values

        # categorize
        return dict([(k, _categorize_by_intervals(v, intervals, labels)) for k, v in data.items()])

    else:
        raise ValueError("Binning by '%s' not supported." % bin_by)


def _get_total_correlation(arg_list) -> float:
    """
    """

    cur_member_out, ensemble_members = arg_list

    # calculate total correlation
    all_ensemble_values = [v for k, v in ensemble_members.items() if k != cur_member_out]
    cur_total_correlation = discrete_random_variable.information_multi(all_ensemble_values)
    del all_ensemble_values

    return cur_total_correlation


def _stop_criteria(ensemble_members: dict, observations: Union[list, tuple, np.array, None],
                   full_ensemble_joint_entropy: float, beta_threshold: float, verbose: bool = False) -> bool:
    """
    Evaluate if both joint entropy and transinformation are higher than the beta_threshold
    """

    ensemble_members_values = list(ensemble_members.values())
    
    # calculate joint entropy
    joint_entropy = discrete_random_variable.entropy_joint(ensemble_members_values)
    joint_entropy_ratio = joint_entropy/full_ensemble_joint_entropy
    stop_due_joint_entropy = True if joint_entropy_ratio <= beta_threshold else False
    print("  With %2d members. Current Joint Entropy: %.05f. Full Joint Entropy: %.05f. Ratio: %.05f." % 
        (len(ensemble_members), joint_entropy, full_ensemble_joint_entropy, joint_entropy_ratio)) if verbose else None

    # calculate transinformation if needed
    transinformation = None if observations is None else \
        discrete_random_variable.information_mutual(ensemble_members_values, observations)
    stop_due_transinformation = True if ((transinformation is None) or (transinformation <= beta_threshold)) else False

    # define final stop decision
    stop = True if (stop_due_joint_entropy and stop_due_transinformation) else False

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


# ## PUBLIC METHODS ################################################################################################## #

def load_data_75():
    """
    Return a Pandas DataFrame with series of 75 ensemble members.
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'example_data/ensemble_set_75.pickle')
    return pd.read_pickle(stream)
    

def select_ensemble_members(all_ensemble_members: dict, observations: Union[list, tuple, np.array, None] = None,
                            n_bins: int = 10, bin_by: str = 'quantile_individual',
                            beta_threshold: float = 0.9, n_processes: int = 1, minimum_n_members: int = 2, 
                            verbose: bool = False) -> dict:
    """
    Performs the ensemble selection considering (or not) observations.
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

    # discretize data if needed
    disc_ensemble_members_remaining = _discretize_if_needed(all_ensemble_members, n_bins, bin_by)
    disc_observations = _discretize_if_needed(observations, n_bins, bin_by)

    # calculate joint entropy of the original ensemble set
    full_ensemble_joint_entropy = discrete_random_variable.entropy_joint(list(disc_ensemble_members_remaining.values()))

    # create empty accumulator variables
    total_correlations, joint_entropies, transinformations = [], [], []

    # enters iterative loop
    print("Starting with %d ensemble members." % len(disc_ensemble_members_remaining)) if verbose else None
    while len(disc_ensemble_members_remaining) > minimum_n_members:
        # identifies winner
        print(" Identifying winner set.") if verbose else None
        disc_ensemble_members_remaining, total_corr = _select_winner_ensemble_set(disc_ensemble_members_remaining,
                                                                                  n_processes)
        total_correlations.append(total_corr)

        # check stopping criteria
        joint_entropy, transinformation, stop = _stop_criteria(disc_ensemble_members_remaining, disc_observations,
                                                               full_ensemble_joint_entropy, beta_threshold,
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
        "original_ensemble_joint_entropy": full_ensemble_joint_entropy
    }
