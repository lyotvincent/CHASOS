'''
@author: 孙嘉良
@purpose: construct graph
'''

from gnn_data_loader import NodeLoader, PositiveSampleLoader, NegativeSampleLoader
import torch
import dgl


def construct_graph(cell_line_list: list):
    nl = NodeLoader(cell_line_list)
    nl.load_nodes()
    node_num = len(nl.cell_line_node_dict["GM12878"]["ndata_sequence_features"])

    psl = PositiveSampleLoader(cell_line_list, nl.node_dict)
    psl.load_positive_samples()
    psl.split_train_val()
    # 训练集 正例
    train_pos_g = dgl.graph(psl.positive_data_dict["GM12878"]["train_pos_edges"], num_nodes=node_num)
    train_pos_g.ndata["chr_features"] = torch.tensor(nl.cell_line_node_dict["GM12878"]["ndata_chr_features"])
    train_pos_g.ndata["orientation_strand_features"] = torch.tensor(nl.cell_line_node_dict["GM12878"]["ndata_orientation_strand_features"])
    train_pos_g.ndata["mid_point_features"] = torch.tensor(nl.cell_line_node_dict["GM12878"]["ndata_mid_point_features"])
    train_pos_g.ndata["mid_point_features"] = train_pos_g.ndata["mid_point_features"].unsqueeze(-1)
    train_pos_g.ndata["sequence_features"] = torch.tensor(nl.cell_line_node_dict["GM12878"]["ndata_sequence_features"])
    train_pos_g.ndata["sequence_features"] = train_pos_g.ndata["sequence_features"].reshape([-1, 4000])
    # 验证集 正例
    val_pos_g = dgl.graph(psl.positive_data_dict["GM12878"]["val_pos_edges"], num_nodes=node_num)
    # val_pos_g.ndata["sequence_features"] = torch.tensor(nl.cell_line_node_dict["GM12878"]["ndata_sequence_features"])
    # val_pos_g.ndata["sequence_features"] = val_pos_g.ndata["sequence_features"].reshape([-1, 4000])

    nsl = NegativeSampleLoader(cell_line_list, nl.node_dict)
    nsl.load_negative_samples()
    nsl.split_train_val()
    # 训练集 反例
    train_neg_g = dgl.graph(nsl.negative_data_dict["GM12878"]["train_neg_edges"], num_nodes=node_num)
    # 验证集 反例
    val_neg_g = dgl.graph(nsl.negative_data_dict["GM12878"]["val_neg_edges"], num_nodes=node_num)

    return train_pos_g, train_neg_g, val_pos_g, val_neg_g


if __name__ == "__main__":

    train_pos_g, train_neg_g, val_pos_g, val_neg_g = construct_graph(["GM12878"])
    # print(train_pos_g)
    # print(len(train_pos_g.ndata["mid_point_features"]))
    # print(train_neg_g)
    # print(val_pos_g)
    # print(val_neg_g)
    _, _, edge_ids = train_pos_g.edges("all")
    print(len(edge_ids))

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    # dataloader = dgl.dataloading.EdgeDataLoader(
    #     train_pos_g, train_seeds, sampler,
    #     batch_size=8,
    #     shuffle=True,
    #     drop_last=False,
    #     pin_memory=True,
    #     num_workers=args.num_workers)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([5,4,3])
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(
        train_pos_g, edge_ids, sampler,
        batch_size=2, shuffle=True, drop_last=False, num_workers=4)

    print(f'len(dataloader): {len(dataloader)}')
    for input_nodes, positive_graph, blocks in dataloader:
        print(input_nodes)
        print(positive_graph)
        print(blocks)
        print(len(blocks))

        print("1="*7)
        print(blocks[0])
        induced_src = blocks[0].srcdata[dgl.NID]
        induced_dst = blocks[0].dstdata[dgl.NID]
        print(f"induced_src, induced_dst: {induced_src}, {induced_dst}")
        print(blocks[0].edata[dgl.EID])
        src, dst = blocks[0].edges(order='eid')
        print(src, dst)
        print(induced_src[src], induced_dst[dst])

        print("2="*7)
        print(blocks[1])
        induced_src = blocks[1].srcdata[dgl.NID]
        induced_dst = blocks[1].dstdata[dgl.NID]
        print(f"induced_src, induced_dst: {induced_src}, {induced_dst}")
        print(blocks[1].edata[dgl.EID])
        src, dst = blocks[1].edges(order='eid')
        print(src, dst)
        print(induced_src[src], induced_dst[dst])

        print("3="*7)
        print(blocks[2])
        induced_src = blocks[2].srcdata[dgl.NID]
        induced_dst = blocks[2].dstdata[dgl.NID]
        print(f"induced_src, induced_dst: {induced_src}, {induced_dst}")
        print(blocks[2].edata[dgl.EID])
        src, dst = blocks[2].edges(order='eid')
        print(src, dst)
        print(induced_src[src], induced_dst[dst])
        exit()
