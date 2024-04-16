from enum import Enum

import numpy


class DecisionPolicy(Enum):
    """
    Class for guiding the final decision on anomalies
    """
    NONE = 1
    TWO_ROW = 2
    THREE_ROW = 3
    TWO_IN_THREE = 4
    THREE_IN_FOUR = 5
    TWO_IN_FOUR = 6


def policy_from_string(p_str):
    if p_str in ['none', '', None, 'classifier', 'clf']:
        return DecisionPolicy.NONE
    elif p_str in ['2', 'two', 'double']:
        return DecisionPolicy.TWO_ROW
    elif p_str in ['3', 'three', 'triple']:
        return DecisionPolicy.THREE_ROW
    elif p_str in ['2oo3', 'two three', 'two in three', '2 in 3']:
        return DecisionPolicy.TWO_IN_THREE
    elif p_str in ['2oo4', 'two four', 'two in four', '2 in 4']:
        return DecisionPolicy.TWO_IN_FOUR
    elif p_str in ['3oo4', 'three four', 'three in four', '3 in 4']:
        return DecisionPolicy.THREE_IN_FOUR
    else:
        print('Unable to recognize policy \'%s\', using NONE as default', p_str)
        return DecisionPolicy.NONE


def policy_to_string(p_obj):
    if p_obj == DecisionPolicy.TWO_ROW:
        return '2'
    elif p_obj == DecisionPolicy.THREE_ROW:
        return '3'
    elif p_obj == DecisionPolicy.TWO_IN_THREE:
        return '2oo3'
    elif p_obj == DecisionPolicy.TWO_IN_FOUR:
        return '2oo4'
    elif p_obj == DecisionPolicy.THREE_IN_FOUR:
        return '3oo4'
    else:
        return 'none'


def apply_policy(clf_y, p_obj, default_tag: str = 'normal'):
    """
    Returns predictions according to a specific policy
    :param clf_y: classifier predictions
    :param p_obj: policy
    :param default_tag: tag that indicates normal data
    :return: a numpy array of predictions
    """
    p_y = []
    for i in range(0, len(clf_y)):
        an_last_2 = (clf_y[i-1:i+1] != default_tag).sum() if i > 0 else 0
        an_last_3 = (clf_y[i-2:i+1] != default_tag).sum() if i > 1 else 0
        an_last_4 = (clf_y[i-3:i+1] != default_tag).sum() if i > 2 else 0
        if p_obj == DecisionPolicy.TWO_ROW and an_last_2 == 2:
            new_y = clf_y[i]
        elif p_obj == DecisionPolicy.THREE_ROW and an_last_3 == 3:
            new_y = clf_y[i]
        elif p_obj == DecisionPolicy.TWO_IN_THREE and an_last_3 >= 2:
            new_y = clf_y[i]
        elif p_obj == DecisionPolicy.TWO_IN_FOUR and an_last_4 >= 2:
            new_y = clf_y[i]
        elif p_obj == DecisionPolicy.THREE_IN_FOUR and an_last_4 >= 3:
            new_y = clf_y[i]
        elif p_obj == DecisionPolicy.NONE:
            new_y = clf_y[i]
        else:
            new_y = 'normal'
        p_y.append(new_y)
    return numpy.asarray(p_y)
