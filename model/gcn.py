import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Gcn(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Gcn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_params()
        
    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        # input: (seq_len x in_features)
        # adj: (seq_len x seq_len)
        support = torch.mm(input, self.weight)
        
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output