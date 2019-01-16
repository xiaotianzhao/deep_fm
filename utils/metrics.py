#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/15/19
"""


def auc_score(preds):
    """

    :param preds: list obj， contains tuple; likes: [(y_pred, y_label), ...]
    :return: auc 的值
    """
    rank = len(preds)
    pos_threshold = 0.5  #: label is 0 and 1 so label>pos_threshold is positive
    nums_pos, pos_rank_sum = 0, 0
    preds.sort(key=lambda x: x[0], reverse=True)
    for _, label in preds:
        if label > pos_threshold:
            nums_pos += 1
            pos_rank_sum += rank
        rank -= 1
    return ((pos_rank_sum - nums_pos * (nums_pos - 1) / 2.0) / (nums_pos * (len(preds) - nums_pos))) \
        if nums_pos > 0 and (len(preds) - nums_pos) > 0 else 0