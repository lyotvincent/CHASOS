'''
@author: 孙嘉良
@purpose: construct model
'''


import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        # init.normal_(self.conv1.fc_self.weight, 0.5, 0.01)
        # init.normal_(self.conv1.fc_neigh.weight, 0.5, 0.01)
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=out_feats, aggregator_type='mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.linear_list = [
            nn.Linear(h_feats * 2, h_feats * 4),
            nn.ReLU(),
            nn.Linear(h_feats * 4, h_feats * 4),
            nn.ReLU(),
            nn.Linear(h_feats * 4, h_feats * 2),
            nn.ReLU(),
            nn.Linear(h_feats * 2, h_feats * 1),
            nn.ReLU(),
            nn.Linear(h_feats, 1),
            nn.Sigmoid()
        ]
        self.sequential = nn.Sequential(*self.linear_list)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        # 在这里聚合
        return {"score": self.sequential(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

class AvgDistanceConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, h):
        with g.local_scope():
            pos = h[:, 0]  # 特征的第一维是节点位置
            g.ndata['pos'] = pos
            g.update_all(self.message_func, self.reduce_func)
            h_dst = g.ndata.pop('h_dst')
            pos_dist = torch.stack([pos, h_dst], dim=1)
            return pos_dist

    def message_func(self, edges):
        pos_i = edges.src['pos'] # size: [edge_num, 1]
        pos_j = edges.dst['pos'] # size: [edge_num, 1]
        dist = torch.abs(pos_i - pos_j) # distance between a list of u and a list of v
        return {'dist': dist}

    def reduce_func(self, nodes):
        '''
        这个函数每次每次处理的是某一个入度的所有节点
        '''
        # print(-1, nodes.nodes())
        dist = nodes.mailbox['dist']
        h = dist.mean(dim=1)
        return {'h_dst': h}

if __name__ == "__main__":
    g = dgl.graph(([0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9], [1, 0, 2, 1, 4, 5, 3, 6, 9, 3, 5, 8, 7, 5]))  # 构建一个简单的图
    # pos = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).unsqueeze(-1)  # 创建节点位置特征
    pos = torch.tensor([10000, 15000, 20000, 40000, 45000, 50000, 55000, 80000, 85000, 60000]).unsqueeze(-1)  # 创建节点位置特征
    # h = torch.cat([pos, torch.randn((10, 32))], dim=-1)  # 将位置特征和其他特征拼接起来
    h = torch.cat([pos, torch.randn((10, 2))], dim=-1)  # 将位置特征和其他特征拼接起来
    print(h.shape)
    conv = AvgDistanceConv()
    h_dst = conv(g, h)
    print(h_dst)

