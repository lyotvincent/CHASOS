'''
@author: 孙嘉良
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from parameters import GLOVE_PATH
from positional_encoding import PositionalEncoding
from custom_layer import *



class PretrainedModel_v20230228(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with a simple MLP model
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.fc1 = nn.Linear(996*128, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        # self.fc1 = nn.Linear(332*128, 2048) # a 1000bp sequence has 332 stride=3 5-mers
        # self.fc2 = nn.Linear(2048, 4096)
        # self.fc3 = nn.Linear(4096, 1024)
        # self.fc4 = nn.Linear(1024, 128)
        # self.fc5 = nn.Linear(128, 2)

        # load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h)
        h = nn.Flatten()(h)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230302_v1(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with conv layers
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(4, 4), stride=4)
        self.fc1 = nn.Linear(31744, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        # load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]])
        h = self.conv1(h) # output shape: (batch_size, 16, 249, 32)
        h = nn.ReLU()(h)
        h = self.conv2(h) # output shape: (batch_size, 64, 62, 8)
        h = nn.ReLU()(h)
        h = nn.Flatten()(h) # output shape: (batch_size, 64*62*8)=(batch_size, 31744)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230302_v2(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with conv layers
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 4), stride=(3, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 4), stride=(3, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 4), stride=(3, 2))
        self.fc1 = nn.Linear(16896, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)
        h = self.conv1(h) # output shape: (batch_size, 16, 331, 63)
        h = nn.ReLU()(h)
        h = self.conv2(h) # output shape: (batch_size, 64, 109, 30)
        h = nn.ReLU()(h)
        h = self.conv3(h) # output shape: (batch_size, 128, 35, 14)
        h = nn.ReLU()(h)
        h = self.conv4(h) # output shape: (batch_size, 256, 11, 6)
        h = nn.ReLU()(h)
        h = nn.Flatten()(h) # output shape: (batch_size, 256*11*6)=(batch_size, 16896)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230306_v1(nn.Module):
    '''
    @description: first model combine 5-mers and 1-mers, but its preformance is not good.
                however, its anchor_acc achieve 1.00 in epoch 14, and ocr_acc achieve 0.96.
                to this extent, the model need to add pooling and resnet block.
    @Input: two seqs, one is 996 5-mers, the other is 1000 1-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.embedding_2 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 4), stride=(3, 2))
        self.conv_2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv_2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(3, 1))
        self.conv_2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(3, 1))
        self.fc1 = nn.Linear(19200, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data, monomer_data):
        # 5-mer
        h1 = self.embedding_1(kmer_data)
        h1 = self.pos_encoding(h1) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h1.reshape([-1, 1, h1.shape[1], h1.shape[2]]) # output shape: (batch_size, 1, 996, 128)
        h1 = self.conv_1_1(h1) # output shape: (batch_size, 16, 331, 63)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_2(h1) # output shape: (batch_size, 64, 109, 30)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_3(h1) # output shape: (batch_size, 128, 35, 14)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_4(h1) # output shape: (batch_size, 256, 11, 6)
        h1 = nn.ReLU()(h1)
        h1 = nn.Flatten()(h1) # output shape: (batch_size, 256*11*6)=(batch_size, 16896)
        # 1-mer
        h2 = self.embedding_2(monomer_data)
        h2 = self.pos_encoding(h2)
        h2 = h2.reshape([-1, 1, h2.shape[1], h2.shape[2]])
        h2 = self.conv_2_1(h2) # output shape: (batch_size, 16, 332, 3)
        h2 = nn.ReLU()(h2)
        h2 = self.conv_2_2(h2) # output shape: (batch_size, 32, 110, 1)
        h2 = nn.ReLU()(h2)
        h2 = self.conv_2_3(h2) # output shape: (batch_size, 64, 36, 1)
        h2 = nn.ReLU()(h2)
        h2 = nn.Flatten()(h2) # output shape: (batch_size, 64*36*1)=(batch_size, 2304)

        h = torch.cat((h1, h2), dim=1)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230306_v2(nn.Module):
    '''
    @Input: two seqs, one is 996 5-mers, the other is 1000 1-mers
    此模型未成功运行
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=996)
        # self.embedding_2 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=996, max_seq_len=1000)
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_1 = nn.MaxPool2d((4, 4))
        self.conv_1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_2 = nn.MaxPool2d((4, 4))
        self.conv_1_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_3 = nn.MaxPool2d((4, 4))
        self.conv_1_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_4 = nn.MaxPool2d((4, 4))
        # self.conv_2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        # self.conv_2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(3, 1))
        # self.conv_2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(3, 1))
        self.fc1 = nn.Linear(256*3*3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        # 5-mer
        h1 = self.embedding_1(kmer_data)
        h1 = self.pos_encoding(h1) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h1.reshape([-1, 1, h1.shape[1], h1.shape[2]]) # output shape: (batch_size, 1, 996, 996)
        h1 = self.conv_1_1(h1) # output shape: (batch_size, 64, 996, 996)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_2(h1) # output shape: (batch_size, 64, 996, 996)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_1(h1) # output shape: (batch_size, 64, 249, 249)
        h1 = self.conv_1_3(h1) # output shape: (batch_size, 128, 249, 249)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_4(h1) # output shape: (batch_size, 128, 249, 249)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_2(h1) # output shape: (batch_size, 128, 62, 62)
        h1 = self.conv_1_5(h1) # output shape: (batch_size, 256, 62, 62)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_6(h1) # output shape: (batch_size, 256, 62, 62)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_3(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = self.conv_1_7(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_8(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_4(h1) # output shape: (batch_size, 256, 3, 3)
        h1 = nn.Flatten()(h1) # output shape: (batch_size, 256*3*3)
        # 1-mer
        # h2 = self.embedding_2(monomer_data)
        # h2 = self.pos_encoding(h2)
        # h2 = h2.reshape([-1, 1, h2.shape[1], h2.shape[2]]) # output shape: (batch_size, 1, 1000, 8)
        # h2 = self.conv_2_1(h2) # output shape: (batch_size, 16, 332, 3)
        # h2 = nn.ReLU()(h2)
        # h2 = self.conv_2_2(h2) # output shape: (batch_size, 32, 110, 1)
        # h2 = nn.ReLU()(h2)
        # h2 = self.conv_2_3(h2) # output shape: (batch_size, 64, 36, 1)
        # h2 = nn.ReLU()(h2)
        # h2 = nn.Flatten()(h2) # output shape: (batch_size, 64*36*1)=(batch_size, 2304)

        # h = torch.cat((h1, h2), dim=1)
        h = self.fc1(h1)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230307(nn.Module):
    '''
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()
        self.sequential_2 = self.__get_sequential()
        self.sequential_3 = self.__get_sequential()
        self.sequential_4 = self.__get_sequential()
        self.sequential_5 = self.__get_sequential()
        self.sequential_6 = self.__get_sequential()
        self.sequential_7 = self.__get_sequential()
        self.sequential_8 = self.__get_sequential()
        self.sequential_9 = self.__get_sequential()
        self.sequential_10 = self.__get_sequential()
        self.sequential_11 = self.__get_sequential()

        self.fc1 = nn.Linear(2816, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        # 5-mer
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]])
        h1 = h[:, :, 0:166, :]
        h2 = h[:, :, 83:249, :]
        h3 = h[:, :, 166:332, :]
        h4 = h[:, :, 249:415, :]
        h5 = h[:, :, 332:498, :]
        h6 = h[:, :, 415:581, :]
        h7 = h[:, :, 498:664, :]
        h8 = h[:, :, 581:747, :]
        h9 = h[:, :, 664:830, :]
        h10 = h[:, :, 747:913, :]
        h11 = h[:, :, 830:996, :]

        h = torch.cat([self.sequential_1(h1), self.sequential_2(h2), self.sequential_3(h3), self.sequential_4(h4), self.sequential_5(h5), self.sequential_6(h6), self.sequential_7(h7), self.sequential_8(h8), self.sequential_9(h9), self.sequential_10(h10), self.sequential_11(h11)], dim=1)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230308(nn.Module):
    '''
    @description: stack 11 subgraphs model
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230309(nn.Module):
    '''
    @description: stack 11 subgraphs, with group convolution
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()

        self.fc1 = nn.Linear(352, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=88, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=88, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=176, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=176, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230313(nn.Module):
    '''
    @description: stack 11 subgraphs, with group convolution * 2
                                         + random group convolution * 2
                                         + conv * 2
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential_1()
        self.sequential_2 = self.__get_sequential_2()

        self.fc1 = nn.Linear(352, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential_1(self):
        # nn.GroupNorm(11, 88),
        # import torchvision.models.shufflenetv2
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=88, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )

    def __get_sequential_2(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=176, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=176, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)
        h = channel_shuffle(h, 11)
        h = self.sequential_2(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230315(nn.Module):
    '''
    @description: Embedding + PositionalEncoding + (group FireBlock_v1+SEBlock_v1)*2 + SKBlock_v1 + DenseOutLayer
    @model parameter number: 502730
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.group_convolution = nn.Sequential(
            FireBlock_v1(input_channels=11, squeeze_channels=22, expand_channels=44, groups=11),
            SEBlock_v1(h_channels=44, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.shuffle_group_convolution = nn.Sequential(
            FireBlock_v1(input_channels=44, squeeze_channels=44, expand_channels=88, groups=11),
            SEBlock_v1(h_channels=88, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block = SKBlock_v1(h_channels=88, out_channels=176, reduction=44)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(176, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h)
        # stack
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]
        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        # group convolution
        h = self.group_convolution(h)
        # shuffle group convolution
        # h = channel_shuffle(h, 11)
        h = self.shuffle_group_convolution(h)
        # SKBlock_v1
        h = self.sk_block(h)
        h = self.out_layer(h)
        return h

class PretrainedModel_v20230318(nn.Module):
    '''
    @description: Embedding + PositionalEncoding + (FireBlock_v2+SEBlock_v1)*2 + SKBlock_v1 + DenseOutLayer
    @model parameter number: 502730
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v1(h_channels=11, out_channels=88, reduction=44)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v2(input_channels=88, squeeze_channels=22, e_1_channels=44, e_3_channels=44, groups=1),
            SEBlock_v1(h_channels=88, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block_2 = SKBlock_v1(h_channels=88, out_channels=176, reduction=44)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v2(input_channels=176, squeeze_channels=44, e_1_channels=88, e_3_channels=88, groups=1),
            SEBlock_v1(h_channels=176, reduction=44),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block_3 = SKBlock_v1(h_channels=176, out_channels=352, reduction=44)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(352, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h)
        # stack
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]
        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sk_block_1(h)
        h = self.squeeze_convolution_1(h)
        h = self.sk_block_2(h)
        h = self.squeeze_convolution_2(h)
        h = self.sk_block_3(h)
        h = self.out_layer(h)
        return h

class PretrainedModel_v20230319(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
    @thop.profile: macs=3840937152.0, params=667010.0
    @model parameter number: 798082
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230320(nn.Module):
    """
    @description: model for 996bp, no slice, and use n*1 & 1*n conv (Asymmetric Convolution)
                  add SAOL
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        # Spatial Attention Map
        self.sam_layer = SpatialAttentionMapBlock_v1(h=16, w=2, in_c=256, mid_c=32)
        # Spatial Logits
        self.sl_layer = SpatialLogitsBlock_v1(h=16, w=2, in_c=(64, 128, 256), mid_c=16, out_c=2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h1 = self.sk_block_1(h)                 # output shape: (batch_size, 32, 996, 128)
        h1 = self.squeeze_convolution_1(h1)     # output shape: (batch_size, 64, 250, 32)
        h1 = self.sk_block_2(h1)                # output shape: (batch_size, 64, 250, 32)
        h2 = self.squeeze_convolution_2(h1)     # output shape: (batch_size, 128, 63, 8)
        h2 = self.sk_block_3(h2)                # output shape: (batch_size, 128, 63, 8)
        h3 = self.squeeze_convolution_3(h2)     # output shape: (batch_size, 256, 16, 2)

        sam = self.sam_layer(h3)                # output shape: (batch_size, 1, 16, 2)
        sl = self.sl_layer(h1, h2, h3)          # output shape: (batch_size, 2, 16, 2)
        # Spatial Weighted Sum
        sws = torch.sum(torch.mul(sam, sl), dim=(2,3))
        return sws

class PretrainedModel_v20230324(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  use new layer FireBlock_v4，SEBlock_v2，SKBlock_v3 which reduce some BN and ReLU
    @thop.profile: macs=3794374976.0, params=665442.0
    @model parameter number: 796514
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v3(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v4(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v2(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v3(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v4(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v2(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v3(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v4(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v2(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230325(nn.Module):
    """
    @description: model for 996bp, no slice, and use n*1 & 1*n conv (Asymmetric Convolution)
                  add more channels SAOL
    @thop.profile: macs=3842888224.0, params=711809.0
    @model parameter number: 842881
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        # Spatial Attention Map
        self.sam_layer = SpatialAttentionMapBlock_v1(h=16, w=2, in_c=256, mid_c=32)
        # Spatial Logits
        self.sl_layer = SpatialLogitsBlock_v1(h=16, w=2, in_c=(64, 128, 256), mid_c=16, out_c=16)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h1 = self.sk_block_1(h)                     # output shape: (batch_size, 32, 996, 128)
        h1 = self.squeeze_convolution_1(h1)         # output shape: (batch_size, 64, 250, 32)
        h1 = self.sk_block_2(h1)                    # output shape: (batch_size, 64, 250, 32)
        h2 = self.squeeze_convolution_2(h1)         # output shape: (batch_size, 128, 63, 8)
        h2 = self.sk_block_3(h2)                    # output shape: (batch_size, 128, 63, 8)
        h3 = self.squeeze_convolution_3(h2)         # output shape: (batch_size, 256, 16, 2)

        sam = self.sam_layer(h3)                    # output shape: (batch_size, 1, 16, 2)
        sl = self.sl_layer(h1, h2, h3)              # output shape: (batch_size, 16, 16, 2)
        # Spatial Weighted Sum
        weighted_spatial_sum = torch.mul(sam, sl)   # output shape: (batch_size, 16, 16, 2)
        weighted_spatial_sum = weighted_spatial_sum.reshape((weighted_spatial_sum.shape[0], 2, -1, weighted_spatial_sum.shape[2], weighted_spatial_sum.shape[3]))
        sws = torch.sum(weighted_spatial_sum, dim=(2,3,4))
        return sws

class PretrainedModel_v20230326(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                    ,replace BN with LN
    @thop.profile: macs=3840937152.0, params=51881954.0
    @model parameter number: 52013026
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v4(h_channels=1, out_channels=32, reduction=16, ln_size=(996, 128))
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v5(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, ln_size=(996, 128), groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v4(h_channels=64, out_channels=64, reduction=32, ln_size=(250, 32))
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v5(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, ln_size=(250, 32), groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v4(h_channels=128, out_channels=128, reduction=64, ln_size=(63, 8))
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v5(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, ln_size=(63, 8), groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h


if __name__=="__main__":
    model = PretrainedModel_v20230326()

    # from thop import profile
    # input = torch.randint(4**5, (1, 996))
    # macs, params = profile(model, inputs=(input, ))
    # print(f"thop.profile: macs={macs}, params={params}")

    # from torchstat import stat
    # stat(model, (1, 1000, 4))

    print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

