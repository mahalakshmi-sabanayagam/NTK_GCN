# taken from https://github.com/tkipf/pygcn and modified for our case
import numpy as np
import scipy.sparse as sp
import torch
import sys

sys.setrecursionlimit(99999)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset="WebKB", self_loop=1, feature_normalisation=1):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    path = "../data/" + dataset + "/"

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # combine classes to get binary cls dataset
    l = idx_features_labels[:,-1]
    if dataset == "cora":
        nn = np.where(l == 'Neural_Networks')[0]
        rein_l = np.where(l == 'Reinforcement_Learning')[0]
        ga = np.where(l == 'Genetic_Algorithms')[0]
        cb = np.where(l == 'Case_Based')[0]
        pm = np.where(l == 'Probabilistic_Methods')[0]
        rl = np.where(l == 'Rule_Learning')[0]
        th = np.where(l == 'Theory')[0]
        cls1 = np.concatenate((nn,th,pm), axis=0)
        cls2 = np.concatenate((cb,rl,rein_l,ga), axis=0)
        idx_features_labels[cls2,-1] = 'cls_2'
        idx_features_labels[cls1, -1] = 'cls_1'
        all_chosen_idx = np.concatenate((cb,pm,rl,th,nn,rein_l,ga), axis=0)
    elif dataset == "WebKB":
        course = np.where(l == 'course')[0]
        project = np.where(l == 'project')[0]
        student = np.where(l == 'student')[0]
        faculty = np.where(l == 'faculty')[0]
        staff = np.where(l == 'staff')[0]
        cls2 = np.concatenate((course, project,faculty, staff), axis=0)
        idx_features_labels[cls2, -1] = 'cls_2'
        all_chosen_idx = np.concatenate((student, course, project, faculty, staff), axis=0)
    elif dataset == "citeseer":
        ai = np.where(l == 'AI')[0]
        ml = np.where(l == 'ML')[0]
        db = np.where(l == 'DB')[0]
        ir = np.where(l == 'IR')[0]
        hci = np.where(l == 'HCI')[0]
        agents = np.where(l == 'Agents')[0]
        cls1 = np.concatenate((ai,ml,agents), axis=0)
        idx_features_labels[cls1, -1] = 'cls_1'
        cls2 = np.concatenate((db, ir,hci), axis=0)
        idx_features_labels[cls2, -1] = 'cls_2'
        all_chosen_idx = np.concatenate((ai, ml, db, ir, hci, agents), axis=0)

    idx_features_labels = idx_features_labels[all_chosen_idx]
    np.random.seed(25)
    indices = np.random.permutation(idx_features_labels.shape[0])
    idx_features_labels = idx_features_labels[indices]
    print("Total instances for binary cls ", idx_features_labels.shape)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # dont call onehot as it sometimes create [0,1] and sometimes [1,0] -- leading to non reproducible results.
    # so, assign 0 for cls_2, 1 for cls_1
    labels = np.array([0 if i == 'cls_2' else 1 for i in idx_features_labels[:,-1]])
    print("Labels shape ", labels.shape)

    # build graph
    if dataset == "cora":
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
    elif dataset == "WebKB" or dataset == "citeseer":
        idx = np.array(idx_features_labels[:, 0], dtype=np.str)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.str)

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

    adj = normalize_adj_symmertric(adj, self_loop=self_loop)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
    if feature_normalisation == 1:
       features = normalize(features)

    if dataset == "cora":
        idx_train = range(708)
        idx_val = range(500, 708)
        idx_test = range(708, 2708)
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


def normalize(mx):  # D^-1 mx
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # D^-1
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)  # D^-1 mx
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
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)