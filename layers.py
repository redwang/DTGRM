import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.special import softmax

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import copy

def exchange_time(x, exchange_rate=0.2):
    exchange_label = torch.zeros(x.shape[0], x.shape[2], dtype=torch.long).to(x.device)
    seq_length = (x.shape[2]//2)*2
    exchange_num = int(seq_length/2*exchange_rate)
    randn_pair = torch.randperm(x.shape[2])[:seq_length].reshape(2,-1)
    exchange_pair = randn_pair[:,:exchange_num]

    exchange_index = torch.arange(start=0, end=x.shape[2])
    exchange_index[exchange_pair[0]] = exchange_pair[1]
    exchange_index[exchange_pair[1]] = exchange_pair[0]

    exchange_x = x[:,:,exchange_index]
    exchange_label[:,exchange_pair[0]]=1
    exchange_label[:,exchange_pair[1]]=1

    return exchange_x, exchange_label

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.exchange_out = nn.Conv1d(num_f_maps, 2, 1)

    def forward(self, x, mask, ex_x, ex_label):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        pred = self.conv_out(out) * mask[:, 0:1, :]

        ex_out = self.conv_1x1(ex_x)
        for layer in self.layers:
            ex_out = layer(ex_out, mask)
        ex_pred = self.exchange_out(ex_out) * mask[:, 0:1, :]
        ex_clspred = self.conv_out(ex_out) * mask[:, 0:1, :]

        return pred, ex_pred, ex_label, ex_clspred


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class DRCGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True, kernel_size=3, dilation=1, padding=1):
        super(DRCGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def cosine_pairwise(self, x):
        x = x.permute((1, 3, 0, 2))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-3)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 3, 0, 1))
        batch_size, seq_length, kernel_size, _ = cos_sim_pairwise.shape
        adj = cos_sim_pairwise.reshape(batch_size, seq_length, -1)
        adj = F.softmax(adj, dim=1).reshape(batch_size, seq_length, kernel_size, kernel_size)
        return adj

    def forward(self, x, adj=None):
        batch_size, feat_dim, L = x.shape
        x = x.unsqueeze(3)
        input = F.unfold(x, kernel_size=(self.kernel_size,1), dilation=(self.dilation,1), padding=(self.padding,0))
        input = input.reshape(batch_size, feat_dim, self.kernel_size, L).permute(0,2,3,1)
        if adj is None:
            adj = self.cosine_pairwise(input)
        support = torch.matmul(input, self.weight)
        support = support.permute(0,2,1,3)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = output.permute(0,3,2,1).reshape((batch_size, feat_dim*self.kernel_size, L))
        output = F.fold(output, (1,L), (self.kernel_size,1), dilation=(self.dilation,1), padding=(self.padding,0))
        output = output.squeeze(2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResidualLayer(nn.Module):
    def __init__(self, dilation, df_size, in_channels, out_channels):
        super(GCNResidualLayer, self).__init__()
        padding = int((dilation*(df_size-1))/2)
        self.df_size = df_size
        self.gcn_dilated1 = DRCGraphConvolution(in_channels, out_channels, kernel_size=df_size, dilation=dilation, padding=padding)
        self.gcn_dilated2 = DRCGraphConvolution(in_channels, out_channels, kernel_size=df_size, dilation=dilation, padding=padding)
        self.conv_dilated_adj = nn.Conv1d(out_channels, df_size*df_size, df_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        batch_size, _, seq_length = x.shape
        adj = self.conv_dilated_adj(x)
        adj = F.softmax(adj, dim=1).permute(0,2,1)
        adj = adj.reshape(batch_size, seq_length, self.df_size, self.df_size)
        out = F.relu(self.gcn_dilated1(x)) + F.relu(self.gcn_dilated2(x, adj))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class GCNStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, df_size, dim, num_classes):
        super(GCNStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GCNResidualLayer(2 ** i, df_size, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.exchange_out = nn.Conv1d(num_f_maps, 2, 1)

    def forward(self, x, mask, ex_x, ex_label):
        out = self.conv_1x1(x)
        for layer in self.gcn_layers:
            out = layer(out, mask)
        pred = self.conv_out(out) * mask[:, 0:1, :]

        ex_out = self.conv_1x1(ex_x)
        for layer in self.gcn_layers:
            ex_out = layer(ex_out, mask)
        ex_pred = self.exchange_out(ex_out) * mask[:, 0:1, :]
        ex_clspred = self.conv_out(ex_out) * mask[:, 0:1, :]

        return pred, ex_pred, ex_label, ex_clspred