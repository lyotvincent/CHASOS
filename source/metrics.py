'''
@author: 孙嘉良
@purpose: construct metrics function
'''

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def compute_one_loss(pred_scores, labels):
    loss_a = nn.BCELoss()(pred_scores, labels)
    return loss_a

def compute_two_loss(pred_scores, labels):
    loss_a = nn.BCELoss()(pred_scores[:, 0], labels[:, 0])
    loss_b = nn.BCELoss()(pred_scores[:, 1], labels[:, 1])
    loss = 0.5 * (loss_a + loss_b)
    return loss

def compute_accuracy(pred_scores, labels):
    """计算精度
    Args:
        pred_scores (torch.Tensor): 预测分数，形状为 [batch_size, 1]
        labels (torch.Tensor): 真实标签，形状为 [batch_size, 1]
    Returns:
        float: 精度值(0到1之间的浮点数)
    """
    def compute_one_acc(pred1, label1):
        # 将预测分数四舍五入为0或1，表示预测标签
        pred1 = torch.round(pred1)
        # 计算预测标签与真实标签相同的数量
        correct = (pred1 == label1).sum().item()
        # 计算精度值
        accuracy = correct / len(label1)
        return accuracy
    anchor_acc = compute_one_acc(pred_scores[:, 0], labels[:, 0])
    ocr_acc = compute_one_acc(pred_scores[:, 1], labels[:, 1])

    return anchor_acc, ocr_acc

def compute_auc(pos_score, neg_score):
    with torch.no_grad():
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return roc_auc_score(labels, scores)
