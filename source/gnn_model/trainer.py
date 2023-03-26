'''
@author: 孙嘉良
@purpose: load loop data & construct DGL graph & load model & train model
'''

import itertools
import torch
import sys, time, os

# my modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import *
from graph_constructor import construct_graph
from model import GraphSAGE, MLPPredictor
from scheduler import EarlyStopping
from metrics import compute_loss, compute_accuracy

# * prepare log
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_out = open(ROOT_DIR+r'/source/gnn_model/log/log_'+time.strftime('%Y_%m_%d_%H_%M_%S.txt', time.localtime()), 'w', buffering=8)

# * load data
train_pos_g, train_neg_g, val_pos_g, val_neg_g = construct_graph(["GM12878"])
train_pos_g, train_neg_g, val_pos_g, val_neg_g = train_pos_g.to(device), train_neg_g.to(device), val_pos_g.to(device), val_neg_g.to(device)


# * load model
model = GraphSAGE(train_pos_g.ndata['sequence_features'].shape[1], 2000, 1000)
model.to(device)
pred = MLPPredictor(1000)
pred.to(device)

log_out.write(str(model)+'\n')
log_out.write(str(pred)+'\n')

# * scheduler
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.00001)
early_stopping = EarlyStopping(mode="max", patience=40, verbose=True, delta=1E-5, start_epoch=100)

# * train
for e in range(201):
    # forward
    model.train()
    h = model(train_pos_g, train_pos_g.ndata["sequence_features"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = compute_accuracy(pos_score, neg_score)
        if e % 10 == 0:
            val_pos_score = pred(val_pos_g, h)
            val_neg_score = pred(val_neg_g, h)
            val_loss = compute_loss(val_pos_score, val_neg_score)
            val_acc = compute_accuracy(val_pos_score, val_neg_score)
            log_info = f'In epoch {e},\tloss: {loss:.4f},\tacc: {acc:.4f},\tval_loss: {val_loss:.4f},\tval_acc: {val_acc:.4f},\tpos: {min(pos_score):.3f}~{torch.mean(pos_score):.3f}~{max(pos_score):.3f},\tneg: {min(neg_score):.3f}~{torch.mean(neg_score):.3f}~{max(neg_score):.3f},\tval_pos: {min(val_pos_score):.3f}~{torch.mean(val_pos_score):.3f}~{max(val_pos_score):.3f},\tval_neg: {min(val_neg_score):.3f}~{torch.mean(val_neg_score):.3f}~{max(val_neg_score):.3f}'
            print(log_info)
            log_out.write(log_info + '\n')
    # 早停止
    early_stopping(metric=acc, current_epoch=e)
    #达到早停止条件时，early_stop会被置为True
    if early_stopping.early_stop:
        print("Early stopping")
        log_out.write('Early stopping\n')
        break #跳出迭代，结束训练

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()
