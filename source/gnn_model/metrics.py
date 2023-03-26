'''
@author: 孙嘉良
@purpose: construct metrics function
'''

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to("cuda")
    # return F.binary_cross_entropy_with_logits(scores, labels)
    return nn.BCELoss()(scores, labels)

def compute_accuracy(pos_score, neg_score):
    # print(sum(pos_score >= 0.5), len(pos_score), sum(neg_score < 0.5), len(neg_score))
    return sum(torch.cat([pos_score >= 0.5, neg_score < 0.5]))/(len(pos_score)+len(neg_score))

def compute_auc(pos_score, neg_score):
    with torch.no_grad():
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return roc_auc_score(labels, scores)

