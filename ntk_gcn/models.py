import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN_deep(nn.Module):
    def __init__(self, nfeat, nhid, nclass, layers=1, linear=True, sigma=1):
        super(GCN_deep, self).__init__()

        '''
        nfeat : number of features
        nhid : size of hidden layers
        nclass : number of classes
        layers : number of hidden layers
        linear : linear or relu GCN
        sigma : normalisation constant c_sigma
        '''
        self.layers = layers - 1
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc = nn.ModuleList()
        for i in range(self.layers):
            self.gc.append(GraphConvolution(nhid, nhid))
        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear = linear
        self.sigma = sigma

    def forward(self, x, adj):
        x = self.gc1(x,adj)
        if self.linear == False:
            x = F.relu(x)

        for i,gc_layer in enumerate(self.gc):
            k = (torch.sqrt(torch.tensor(self.sigma).type(torch.float)) / (torch.sqrt(torch.tensor(gc_layer.in_features).type(torch.float))))
            if self.linear:
                x = k * gc_layer(x, adj)
            else:
                x = k * F.relu(gc_layer(x, adj))

        k = (torch.sqrt(torch.tensor(self.sigma).type(torch.float)) / (torch.sqrt(torch.tensor(self.gc2.in_features).type(torch.float))))
        x = k * self.gc2(x, adj)
        return x

class GCN_skip(nn.Module):
    def __init__(self, nfeat, nhid, nclass, layers=1, linear=False, seed=42, skip_formulation='gcn', alpha=0.2, sigma=1, relu_h0=False):
        super(GCN_skip, self).__init__()

        '''
        nfeat : number of features
        nhid : size of hidden layers
        nclass : number of classes
        layers : number of hidden layers
        linear : linear or relu GCN
        seed : seed for weight initialisation for the input transformation
        skip_formulation : gcn (skip-pc) or gcnii (skip-alpha)
        alpha : alpha in skip-alpha
        sigma : normalisation constant c_sigma
        '''
        self.layers = layers -1
        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nhid),requires_grad=False)
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc = nn.ModuleList()
        for i in range(self.layers):
            self.gc.append(GraphConvolution(nhid, nhid))
        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear = linear
        torch.manual_seed(seed)
        self.weight.data.normal_(0,1)
        self.formulation = skip_formulation
        self.alpha = alpha
        self.sigma = sigma
        self.relu_h0 = relu_h0
        self.nhidden = nhid

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        if self.relu_h0:
            h0 = F.relu(x)
        else:
            h0 = x

        k =  (torch.sqrt(torch.tensor(self.sigma).type(torch.float)) / (torch.sqrt(torch.tensor(self.nhidden).type(torch.float))))
        if self.formulation == 'gcn':
            x = k * self.gc1(h0,adj)
            if self.linear == False:
                x = F.relu(x)
        # didn't do extensive experiments with gcnii training! we used NTK as surrogate.
        # approximate implementation is provided here
        else:
            x = k * ((1-self.alpha)*self.gc1(h0,adj) + self.alpha*h0) #h0 should be multiplied with W
            if self.linear == False:
                x = F.relu(x)

        for i,gc_layer in enumerate(self.gc):
            if self.formulation == 'gcn':
                x = k * gc_layer(x, adj)
                if self.linear == False:
                    x = F.relu(x)

            else:
                x = k * ((1-self.alpha)*gc_layer(x,adj) + self.alpha*h0)
                if self.linear == False:
                    x = F.relu(x)

        if self.formulation == 'gcn':
            x = k * self.gc2(x+h0, adj)
        else:
            x = k * self.gc2(x, adj)
        return x


