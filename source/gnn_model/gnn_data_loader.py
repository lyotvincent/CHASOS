'''
@author: 孙嘉良
@purpose: load loop data from ROOT_DIR/data/positive_loop & ROOT_DIR/data/negative_loop
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import *
import numpy as np


chr_dict = dict()
def init_chr_dict():
    for i in range(len(CHROMOSOME_LIST)):
        chr_dict[CHROMOSOME_LIST[i]] = [0.] * len(CHROMOSOME_LIST)
        chr_dict[CHROMOSOME_LIST[i]][i] = 1.

def encode_chr(chr):
    '''
    @purpose: return chromosome one-hot encoding
    @input: chr: "chr1", "chr2", ...
    '''
    return chr_dict[chr]

def encode_orientation_strand(ori_strand):
    '''
    @purpose: return orientation_strand one-hot encoding
    @input: orientation: f/r
            strand: +/-
    '''
    ori_strand_dict = {"f,+":[1., 0., 0.], "f,-":[0., 1., 0.], "r,+":[0., 1., 0.], "r,-":[1., 0., 0.], ".,.":[0., 0., 1.]}
    return ori_strand_dict[ori_strand]

def encode_sequence(seq):
    '''
    @purpose: return chromosome one-hot encoding
    '''
    ATCG_dict = {"A": [1., 0., 0., 0.], "T": [0., 1., 0., 0.], "C": [0., 0., 1., 0.], "G": [0., 0., 0., 1.]}
    return [ATCG_dict[base] for base in seq]

class NodeLoader:
    '''
    @purpose: load nodes & node features
    @output: cell_line_node_dict: {"cell_line": {'ndata_chr_features': list() # size: [node_num, 24]
                                                 'ndata_orientation_strand_features': list() # size: [node_num, 5]
                                                 'ndata_mid_point_features': list() # size: [node_num]
                                                 'ndata_sequence_features': list() # size: [node_num, 1000, 4]}, ...
             }
             node_dict: {
                key: "chr1\tstart\tend\torientation\tstrand\tmid_point\tseq_1",
                value: ordinal number from 0
             }
    '''
    def __init__(self, cell_line_list):
        self.cell_line_list = cell_line_list
        init_chr_dict()

    def load_nodes(self):
        '''
        @purpose: load all pos & neg nodes
        '''
        self.cell_line_node_dict = dict() # key is cell line name
        self.node_dict = dict() # node_str to node id
        for cell_line in self.cell_line_list:
            node_chr_features = list()
            node_orientation_strand_features = list()
            node_mid_point_features = list()
            node_sequence_features = list()
            for loop_type in range(1, 6):
                f = open(POS_LOOP_PATH.format(cell_line, loop_type), 'r')
                lines = f.readlines()
                f.close()
                # 在此处，一行就是一个边的两个节点。这个循环先记录一个节点id的列表，顺便记录节点的特征。
                node_list = list()
                for line in lines:
                    fields = line.strip().split()
                    if "N" in fields[5] or "N" in fields[11]: continue
                    node_1 = fields[:6]
                    node_2 = fields[6:]
                    node_list.append(node_1)
                    node_list.append(node_2)
                node_list = ["\t".join(map(str, i)) for i in node_list]
                node_list = list(set(node_list))
                node_list = [i.split('\t') for i in node_list]
                node_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
                for node in node_list:
                    node_str = "\t".join(map(str, node))
                    if node_str in self.node_dict.keys(): continue
                    # * 节点数据和节点号的对应
                    self.node_dict[node_str] = len(self.node_dict)
                    # * 节点特征1，节点所在的染色体号，人类22条染色体+XY共24维one-hot
                    node_chr_features.append(encode_chr(node[0]))
                    # * 节点特征2，节点所在的方向和正反链，编码为3维one-hot
                    node_orientation_strand_features.append(encode_orientation_strand(node[3]))
                    # * 节点特征3，节点序列的中点
                    node_mid_point_features.append(float(node[4]))
                    # * 节点特征4，节点序列的序列 【1000，4】的one-hot
                    node_sequence_features.append(encode_sequence(node[5]))
            for loop_type in range(6):
                for sub_set in range(2):
                    f = open(NEG_LOOP_PATH.format(cell_line, loop_type, sub_set), 'r')
                    lines = f.readlines()
                    f.close()
                    # 在此处，一行就是一个边的两个节点。这个循环先记录一个节点id的列表，顺便记录节点的特征。
                    node_list = list()
                    edge_str_pairs = []
                    for line in lines:
                        fields = line.strip().split()
                        if "N" in fields[5] or "N" in fields[11]: continue
                        node_1 = fields[:6]
                        node_2 = fields[6:]
                        node_list.append(node_1)
                        node_list.append(node_2)
                        edge_str_pairs.append(("\t".join(map(str, node_1)), "\t".join(map(str, node_2))))
                    node_list = ["\t".join(map(str, i)) for i in node_list]
                    node_list = list(set(node_list))
                    node_list = [i.split('\t') for i in node_list]
                    node_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
                    for node in node_list:
                        node_str = "\t".join(map(str, node))
                        if node_str in self.node_dict.keys(): continue
                        self.node_dict[node_str] = len(self.node_dict)
                        node_chr_features.append(encode_chr(node[0]))
                        node_orientation_strand_features.append(encode_orientation_strand(node[3]))
                        node_mid_point_features.append(float(node[4]))
                        node_sequence_features.append(encode_sequence(node[5]))
            self.cell_line_node_dict[cell_line] = {
                "ndata_chr_features": node_chr_features,
                "ndata_orientation_strand_features": node_orientation_strand_features,
                "ndata_mid_point_features": node_mid_point_features,
                "ndata_sequence_features": node_sequence_features,
            }

class PositiveSampleLoader:
    '''
    @output: {"cell_line": {
                'edges': dict() # keys: [1,5] 对应五个Loop type的边list，每个list的元素是edge，每个edge格式是(node1,node2)
                'train_pos_edges': list() # size: [train edge_num, 2], 这个2是一个edge的(node1, node2)
                'val_pos_edges': same as train_pos_edges}, ...
             }
    '''
    def __init__(self, cell_line_list, node_dict):
        self.cell_line_list = cell_line_list
        self.node_dict = node_dict
        np.random.seed(RANDOM_SEED)

    def load_positive_samples(self):
        '''
        @data_format:
        # * chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # * |-----------------------chr1---------------------|------------------------chr2--------------------|
        @return: several node features
        '''
        self.positive_data_dict = dict()
        for cell_line in self.cell_line_list:
            edges = {1:list(), 2:list(), 3:list(), 4:list(), 5:list()}
            for loop_type in range(1, 6):
                f = open(POS_LOOP_PATH.format(cell_line, loop_type), 'r')
                lines = f.readlines()
                f.close()
                edge_str_pairs = []
                for line in lines:
                    fields = line.strip().split()
                    if "N" in fields[5] or "N" in fields[11]: continue
                    node_1 = fields[:6]
                    node_2 = fields[6:]
                    edge_str_pairs.append(("\t".join(map(str, node_1)), "\t".join(map(str, node_2))))
                for edge_str_1, edge_str_2 in edge_str_pairs:
                    edges[loop_type].append((self.node_dict[edge_str_1], self.node_dict[edge_str_2]))
            self.positive_data_dict[cell_line] = {
                "edges": edges
            }

    def split_train_val(self):
        for cell_line in self.cell_line_list:
            train_pos_edges = list()
            val_pos_edges = list()
            for loop_type in range(1, 6):
                current_pos_edges = np.array(self.positive_data_dict[cell_line]["edges"][loop_type])
                edges_num = len(current_pos_edges)
                eids = np.arange(edges_num)
                np.random.shuffle(eids)
                train_num = round(edges_num*TRAIN_SET_RATIO)
                train_pos_edges.extend(current_pos_edges[eids[:train_num]])
                val_pos_edges.extend(current_pos_edges[eids[train_num:]])
            self.positive_data_dict[cell_line]["train_pos_edges"] = train_pos_edges
            self.positive_data_dict[cell_line]["val_pos_edges"] = val_pos_edges

class NegativeSampleLoader:
    '''
    @output: {"cell_line": {
                'edges': dict() # keys: [0,11] 对应6个Loop type和每个loop type的2种subset, 共12个边list，每个list的元素是edge，每个edge格式是(node1,node2)
                'train_pos_edges': list() # size: [train edge_num, 2], 这个2是一个edge的(node1, node2)
                'val_pos_edges': same as train_pos_edges}, ...
             }
    '''
    def __init__(self, cell_line_list, node_dict):
        self.cell_line_list = cell_line_list
        self.__count_pos_sample_num()
        self.node_dict = node_dict
        np.random.seed(RANDOM_SEED)

    def __count_pos_sample_num(self):
        self.positive_sample_num = dict()
        for cell_line in self.cell_line_list:
            self.positive_sample_num[cell_line] = 0
            for loop_type in range(1, 6):
                f = open(POS_LOOP_PATH.format(cell_line, loop_type), 'r')
                self.positive_sample_num[cell_line] += len(f.readlines())
                f.close()

    def load_negative_samples(self):
        self.negative_data_dict = dict()
        for cell_line in self.cell_line_list:
            sub_set_length = self.positive_sample_num[cell_line]/12
            edges = {i:list() for i in range(12)}
            for loop_type in range(6):
                for sub_set in range(2):
                    f = open(NEG_LOOP_PATH.format(cell_line, loop_type, sub_set), 'r')
                    lines = f.readlines()
                    f.close()
                    edge_str_pairs = []
                    for line in lines:
                        fields = line.strip().split()
                        if "N" in fields[5] or "N" in fields[11]: continue
                        node_1 = fields[:6]
                        node_2 = fields[6:]
                        edge_str_pairs.append(("\t".join(map(str, node_1)), "\t".join(map(str, node_2))))
                    for edge_str_1, edge_str_2 in edge_str_pairs:
                        if len(edges[loop_type*2+sub_set]) < sub_set_length:
                            edges[loop_type*2+sub_set].append((self.node_dict[edge_str_1], self.node_dict[edge_str_2]))
            self.negative_data_dict[cell_line] = {
                "edges": edges
            }

    def split_train_val(self):
        for cell_line in self.cell_line_list:
            train_neg_edges = list()
            val_neg_edges = list()
            for loop_type in range(6):
                for sub_set in range(2):
                    current_pos_edges = np.array(self.negative_data_dict[cell_line]["edges"][loop_type*2+sub_set])
                    edges_num = len(current_pos_edges)
                    eids = np.arange(edges_num)
                    np.random.shuffle(eids)
                    train_num = round(edges_num*TRAIN_SET_RATIO)
                    train_neg_edges.extend(current_pos_edges[eids[:train_num]])
                    val_neg_edges.extend(current_pos_edges[eids[train_num:]])
            self.negative_data_dict[cell_line]["train_neg_edges"] = train_neg_edges
            self.negative_data_dict[cell_line]["val_neg_edges"] = val_neg_edges

if __name__ == '__main__':
    custom_cell_line_list = ["GM12878"]

    nl = NodeLoader(custom_cell_line_list)
    nl.load_nodes()

    psl = PositiveSampleLoader(custom_cell_line_list, nl.node_dict)
    psl.load_positive_samples()
    psl.split_train_val()
    print(len(psl.positive_data_dict["GM12878"]["train_pos_edges"]))
    print(len(psl.positive_data_dict["GM12878"]["val_pos_edges"]))

    nsl = NegativeSampleLoader(custom_cell_line_list, nl.node_dict)
    nsl.load_negative_samples()
    nsl.split_train_val()
    # print(len(nsl.negative_data_dict["GM12878"]["edges"]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][0]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][1]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][2]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][3]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][4]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][5]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][6]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][7]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][8]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][9]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][10]))
    # print(len(nsl.negative_data_dict["GM12878"]["edges"][11]))
    print(len(nsl.negative_data_dict["GM12878"]["train_neg_edges"]))
    print(len(nsl.negative_data_dict["GM12878"]["val_neg_edges"]))




