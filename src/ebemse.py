from pyitlib import discrete_random_variable
from typing import Tuple, Union
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
    labels = list(string.ascii_uppercase[0:n_bins])
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
            
    else:
        raise ValueError("Binning by '%s' not supported." % bin_by)


def _stop_criteria(ensemble_members: dict, observations: Union[list, tuple, np.array, None],
                   full_ensemble_joint_entropy: float, beta_threshold: float, verbose: bool = False) -> \
    bool:
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
        min_total_correlation, winner_ensemble_subset = np.inf, None
        for cur_member_out in ensemble_members.keys():
            # 
            cur_ensemble_subset = copy.deepcopy(ensemble_members)
            del cur_ensemble_subset[cur_member_out]
            a = list(cur_ensemble_subset.values())
            cur_total_correlation = discrete_random_variable.information_multi(a)
            # 
            if (winner_ensemble_subset is None) or (cur_total_correlation < min_total_correlation ):
                min_total_correlation, winner_ensemble_subset = cur_total_correlation, cur_ensemble_subset
            del cur_member_out, cur_ensemble_subset, cur_total_correlation
    else:
        raise NotImplementedError("No support for parallel processing yet.")

    return winner_ensemble_subset, min_total_correlation


# ## PUBLIC METHODS ################################################################################################## #

def select_ensemble_members(all_ensemble_members: dict, observations: Union[list, tuple, np.array, None] = None,
                            n_bins: Union[int, None] = 10, bin_by: str = 'quantile_individual',
                            beta_threshold: float = 0.9, n_processes: int = 1, minimum_n_members: int = 2, 
                            verbose: bool = False) -> dict:
    """
    Performs the ensemble selection considering (or not) observations.
    """

    # basic checks
    # TODO: n_bins must be a positive integer
    # TODO: beta threshold should be between 0 and 1
    # TODO: n_processes should be and integer greater than 0

    # discretize data if needed
    disc_ensemble_members_remaining = _discretize_if_needed(all_ensemble_members, n_bins, bin_by)
    disc_observations = _discretize_if_needed(observations, n_bins, bin_by)

    # calculate joint entropy of the original ensemble set
    full_ensemble_joint_entropy = discrete_random_variable.entropy_joint(list(all_ensemble_members.values()))

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
