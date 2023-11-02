# taken from https://github.com/tkipf/pygcn and modified for our case
import numpy as np
import scipy.sparse as sp
import torch
import sys
from sklearn.kernel_ridge import KernelRidge

sys.setrecursionlimit(99999)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def dc_sbm(n=1000, p=0.8, q=0.1, r=-1, sample=True):
    # adj sampled from sbm, features = I
    if r != -1:
        q = ((1 - r) * p) / (1 + r)
    print('prob ', p, q)

    np.random.seed(10)
    deg_cor = np.random.uniform(0.1, 1, n)
    deg_cor = torch.tensor(deg_cor)
    pi_sum = int(n) / 2  # int(n)/2
    deg_cor[:int(n / 2)] = (pi_sum * deg_cor[:int(n / 2)]) / (2 * torch.sum(deg_cor[:int(n / 2)]))
    deg_cor[int(n / 2):] = (pi_sum * deg_cor[int(n / 2):]) / (2 * torch.sum(deg_cor[int(n / 2):]))
    deg_cor_mat = deg_cor.reshape(-1, 1) @ deg_cor.reshape(1, -1)

    sbm = q * torch.ones((n, n), dtype=torch.float64)
    sbm[:int(n / 2), :int(n / 2)] = p
    sbm[int(n / 2):, int(n / 2):] = p

    dc_sbm = deg_cor_mat * sbm

    if sample == False:
        adj = dc_sbm
    else:
        adj = torch.distributions.binomial.Binomial(1, dc_sbm).sample()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
    y = np.zeros((n, 2), dtype=np.int64)
    y[:int(n / 2), 0] = 1
    y[int(n / 2):, 1] = 1
    labels = torch.LongTensor(y)
    features = torch.FloatTensor(torch.eye(n))
    adj = sp.coo_matrix(adj)
    return adj, features, labels

def load_data(dataset="cora", self_loop=1, feature_normalisation=1, adj_norm='row_norm', order_by_cls=False):

    if dataset == "dc_sbm":
        n = 1000
        adj, features, labels = dc_sbm(n=n, p=0.8, q=0.2, r=-1, sample=True)

        if adj_norm == 'sym_norm':  ## D^-0.5 adj D^-0.5
            adj = normalize_adj_symmertric(adj, self_loop=self_loop)
        elif adj_norm == 'row_norm':  ## D^-1 adj
            if self_loop:
                adj = normalize(adj + sp.eye(adj.shape[0]))
            else:
                adj = normalize(adj)
        elif adj_norm == 'col_norm':  ## adj D^-1
            if self_loop:
                adj = normalize(adj + sp.eye(adj.shape[0]), False)
            else:
                adj = normalize(adj, False)
        elif adj_norm == 'unnorm':  ## adj
            if self_loop:
                adj = adj + sp.eye(adj.shape[0])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = range(int(0.7 * n))
        idx_val = range(int(0.8 * n), n)
        idx_test = range(int(0.7 * n), n)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return adj, features, labels, idx_train, idx_val, idx_test

    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    path = "../data/" + dataset + "/"

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    np.random.seed(25)
    if order_by_cls:
        indices = np.argsort(idx_features_labels[:, -1])
    else:
        indices = np.random.permutation(idx_features_labels.shape[0])
    idx_features_labels = idx_features_labels[indices]
    print(np.unique(idx_features_labels[:, -1], return_counts=True))
    print("Total instances ", idx_features_labels.shape)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print("Labels shape ", labels.shape)

    # build graph
    if dataset == "cora":
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
    elif dataset == "WebKB" or dataset == "citeseer":
        idx = np.array(idx_features_labels[:, 0], dtype=str)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=str)

    # pick only the edges with chosen nodes
    s = [edges_unordered[i] for i in range(edges_unordered.shape[0]) if
         edges_unordered[i][0] in idx and edges_unordered[i][1] in idx]
    s = np.array(s)
    edges_unordered = s

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if adj_norm == 'sym_norm':  ## D^-0.5 adj D^-0.5
        adj = normalize_adj_symmertric(adj, self_loop=self_loop)
    elif adj_norm == 'row_norm':  ## D^-1 adj
        if self_loop:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)
    elif adj_norm == 'col_norm': ## adj D^-1
        if self_loop:
            adj = normalize(adj + sp.eye(adj.shape[0]), False)
        else:
            adj = normalize(adj, False)
    elif adj_norm == 'unnorm':  ## adj
        if self_loop:
            adj = adj + sp.eye(adj.shape[0])

    if feature_normalisation == 1:
       features = normalize(features)

    if dataset == "cora":
        idx_train = range(1000)
        idx_val = range(1000, 1208)
        idx_test = range(1208, 2708)
    elif dataset == "WebKB":
        idx_train = range(377)
        idx_val = range(350, 377)
        idx_test = range(377, 877)
    elif dataset == "citeseer":
        idx_train = range(1312)
        idx_val = range(1200, 1312)
        idx_test = range(1312, 3312)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test



def normalize(mx, row_norm=True):  # D^-1 mx or mx D^-1
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # D^-1
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    if row_norm:
        mx = r_mat_inv.dot(mx)  # D^-1 mx
    else:
        mx = mx.dot(r_mat_inv)  # mx D^-1
    return mx


def normalize_adj_symmertric(mx, self_loop=1):  # D^-0.5 A D^-0.5
    if self_loop == 1:
        mx = mx + sp.eye(mx.shape[0])  # add self loops
    print('self loop added? ', self_loop)
    """Symmetric normalization sparse matrix"""
    rowsum = np.array(mx.sum(1))  # deg
    r_inv = np.power(rowsum, -0.5).flatten()  # deg^-0.5
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)  # D^-0.5
    mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)  # D^-0.5 mx D^-0.5
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    true_labels = labels.max(1)[1].type_as(labels)
    correct = preds.eq(true_labels).double()
    correct = correct.sum()
    return correct / len(labels)

def kernel_ridge_reg(kernel_train, kernel_test, labels_train, alpha=0):
    krr = KernelRidge(alpha=alpha, kernel='precomputed')
    krr.fit(kernel_train, labels_train)
    output = torch.tensor(krr.predict(kernel_test))
    return output

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)